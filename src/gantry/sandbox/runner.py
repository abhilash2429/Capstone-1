"""Executing commands under containment.

Two layers, because they defend against different things.

The **subprocess layer** is always on. It confines paths, strips the
environment down to an explicit allowlist, applies resource limits through a
re-exec wrapper, runs the command in its own process group, and bounds output.
It is a good guardrail and an honest one: it stops an agent damaging the
workspace or the developer's machine by accident.

The **container layer** is opt-in and is the actual boundary. When the work is
untrusted - a fixture repository from the internet, code the agent wrote itself
- ``--network none``, ``--cap-drop ALL`` and a disposable filesystem are what
containment means. The subprocess layer cannot offer that, and pretending
otherwise would be the dangerous kind of security theatre.

Three details that are easy to get wrong and expensive to get wrong:

* **Kill the process group, not the process.** ``pytest`` that spawned a server
  leaves the server running when you kill only the child, and the next run
  fails on a port already in use.
* **Keep draining the pipes.** Capping captured output by *not reading* fills
  the pipe buffer and blocks the child forever. Output is drained in full and
  only retained up to the cap.
* **Scrub the environment.** A subprocess inherits every credential in the
  parent's environment by default, and an agent that greps its own environment
  finds your cloud keys.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from gantry.config import SandboxConfig
from gantry.errors import CommandDenied, SandboxError
from gantry.sandbox.jail import PathJail
from gantry.sandbox.policy import CommandPolicy, Decision, PolicyCounters
from gantry.telemetry import metrics
from gantry.telemetry import semconv as sc
from gantry.telemetry.tracer import Tracer, get_tracer

#: Environment variables a command may see. Everything else is dropped, so a
#: credential has to be added deliberately rather than inherited silently.
ENV_ALLOWLIST: tuple[str, ...] = (
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "TZ",
    "PYTHONHASHSEED",
    "PYTHONDONTWRITEBYTECODE",
)

#: Pointed at a black hole when the network is meant to be off. Honest about
#: what it is: a hint that well-behaved clients respect, not containment.
NETWORK_OFF_ENV = {
    "http_proxy": "http://127.0.0.1:9",
    "https_proxy": "http://127.0.0.1:9",
    "HTTP_PROXY": "http://127.0.0.1:9",
    "HTTPS_PROXY": "http://127.0.0.1:9",
    "no_proxy": "",
    "NO_PROXY": "",
}


@dataclass
class SandboxResult:
    """The outcome of one sandboxed command."""

    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    duration_ms: float = 0.0
    timed_out: bool = False
    truncated: bool = False
    killed_group: bool = False
    command: tuple[str, ...] = ()
    mode: str = "subprocess"
    denied: Decision | None = None

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and self.denied is None

    def combined_output(self, max_chars: int = 16_000) -> str:
        """Model-facing rendering: what happened, then what it printed."""
        if self.denied is not None:
            return f"Command refused ({self.denied.rule}): {self.denied.reason}"
        parts = [f"$ {shlex.join(self.command)}"]
        if self.timed_out:
            parts.append(f"[timed out after {self.duration_ms / 1000:.1f}s and was killed]")
        parts.append(f"[exit code {self.exit_code}]")
        if self.stdout.strip():
            parts.append(f"--- stdout ---\n{self.stdout.rstrip()}")
        if self.stderr.strip():
            parts.append(f"--- stderr ---\n{self.stderr.rstrip()}")
        if self.truncated:
            parts.append("[output was truncated]")
        text = "\n".join(parts)
        return text if len(text) <= max_chars else text[:max_chars] + "\n[...truncated]"

    def as_attributes(self) -> dict[str, Any]:
        return {
            sc.SANDBOX_EXIT_CODE: self.exit_code,
            sc.SANDBOX_MODE: self.mode,
            "gantry.sandbox.timed_out": self.timed_out,
            "gantry.sandbox.truncated": self.truncated,
            "gantry.sandbox.killed_group": self.killed_group,
        }


def _drain(stream: IO[bytes] | None, cap: int, sink: dict[str, Any], key: str) -> None:
    """Read a pipe to EOF, keeping at most ``cap`` bytes.

    Draining in full matters more than the cap: a reader that stops reading
    fills the pipe buffer and the child blocks on its next write, forever. The
    process then looks hung when it is merely unheard.
    """
    if stream is None:
        return
    kept = bytearray()
    dropped = 0
    try:
        for chunk in iter(lambda: stream.read(8192), b""):
            room = cap - len(kept)
            if room > 0:
                kept.extend(chunk[:room])
            dropped += max(0, len(chunk) - max(0, room))
    except (OSError, ValueError):
        pass
    finally:
        sink[key] = kept.decode("utf-8", errors="replace")
        sink[f"{key}_dropped"] = dropped


class SandboxRunner:
    """Runs commands inside the workspace, under policy and resource limits."""

    def __init__(
        self,
        jail: PathJail,
        policy: CommandPolicy | None = None,
        config: SandboxConfig | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.jail = jail
        self.policy = policy or CommandPolicy()
        self.config = config or SandboxConfig()
        self.tracer = tracer or get_tracer()
        self.counters = PolicyCounters()

    # -- environment -----------------------------------------------------
    def build_env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        """A minimal environment, built by allowlist rather than by removal.

        Denying known-secret names would miss the next one. Starting from
        nothing means a new credential in the parent's environment is invisible
        to commands by default.
        """
        env = {name: os.environ[name] for name in ENV_ALLOWLIST if name in os.environ}
        env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
        env["HOME"] = str(self.jail.root)
        env["TMPDIR"] = str(self.jail.root)
        env["PWD"] = str(self.jail.root)
        # Deterministic hashing, so a re-run of a failing command reproduces.
        env["PYTHONHASHSEED"] = "0"
        if not self.config.allow_network:
            env.update(NETWORK_OFF_ENV)
        env.update(extra or {})
        return env

    def _wrap_with_limits(self, argv: list[str]) -> list[str]:
        """Prepend the re-exec wrapper that applies rlimits."""
        return [
            sys.executable,
            "-m",
            "gantry.sandbox._limits",
            "--cpu",
            str(self.config.max_cpu_seconds),
            "--mem-mb",
            str(self.config.max_memory_mb),
            "--fsize-mb",
            str(self.config.max_file_size_mb),
            "--nproc",
            str(self.config.max_processes),
            "--",
            *argv,
        ]

    # -- execution -------------------------------------------------------
    def run(
        self,
        command: str | list[str],
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        timeout_s: float | None = None,
        stdin: str | None = None,
    ) -> SandboxResult:
        """Run one command. Never raises for a command that merely failed."""
        decision = self.policy.check(command)
        self.counters.record(decision)
        with self.tracer.span("sandbox.run", kind=sc.SpanKind.SANDBOX) as span:
            span.set_attributes(
                {
                    sc.SANDBOX_DECISION: "allowed" if decision.allowed else "denied",
                    sc.SANDBOX_MODE: "container" if self.config.use_container else "subprocess",
                }
            )
            if not decision.allowed:
                span.set_attribute("gantry.sandbox.rule", decision.rule)
                metrics.SANDBOX_DECISIONS.inc(decision="denied", rule=decision.rule)
                return SandboxResult(exit_code=126, denied=decision, command=tuple(decision.argv))

            metrics.SANDBOX_DECISIONS.inc(decision="allowed", rule="default")
            workdir = self.jail.resolve(cwd) if cwd is not None else self.jail.root
            if not workdir.is_dir():
                raise SandboxError(f"working directory does not exist: {workdir}")

            argv = list(decision.argv)
            timeout = timeout_s if timeout_s is not None else self.config.timeout_s
            runner = self._run_container if self.config.use_container else self._run_subprocess
            result = runner(argv, workdir, self.build_env(env), timeout, stdin)
            span.set_attributes(result.as_attributes())
            if not result.ok:
                span.set_status("error", f"exit {result.exit_code}")
            return result

    def _run_subprocess(
        self,
        argv: list[str],
        workdir: Path,
        env: dict[str, str],
        timeout: float,
        stdin: str | None,
    ) -> SandboxResult:
        started = time.monotonic()
        try:
            process = subprocess.Popen(  # noqa: S603 - argv list, never a shell
                self._wrap_with_limits(argv),
                cwd=str(workdir),
                env=env,
                stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                # Its own process group, so a timeout can kill the whole tree.
                # process_group is used rather than preexec_fn=os.setsid because
                # the latter runs after a fork in a threaded process.
                process_group=0,
                close_fds=True,
            )
        except OSError as exc:
            return SandboxResult(
                exit_code=127,
                stderr=f"could not start {argv[0]!r}: {exc}",
                command=tuple(argv),
                duration_ms=(time.monotonic() - started) * 1000.0,
            )

        sink: dict[str, Any] = {}
        readers = [
            threading.Thread(
                target=_drain,
                args=(process.stdout, self.config.max_output_bytes, sink, "stdout"),
                daemon=True,
            ),
            threading.Thread(
                target=_drain,
                args=(process.stderr, self.config.max_output_bytes, sink, "stderr"),
                daemon=True,
            ),
        ]
        for reader in readers:
            reader.start()

        if stdin is not None and process.stdin is not None:
            try:
                process.stdin.write(stdin.encode())
                process.stdin.close()
            except OSError:
                pass  # the child may have exited before reading its input

        timed_out = killed_group = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            killed_group = self._kill_group(process)

        for reader in readers:
            reader.join(timeout=5)

        return SandboxResult(
            exit_code=process.returncode if process.returncode is not None else -1,
            stdout=sink.get("stdout", ""),
            stderr=sink.get("stderr", ""),
            duration_ms=(time.monotonic() - started) * 1000.0,
            timed_out=timed_out,
            truncated=bool(sink.get("stdout_dropped") or sink.get("stderr_dropped")),
            killed_group=killed_group,
            command=tuple(argv),
            mode="subprocess",
        )

    @staticmethod
    def _kill_group(process: subprocess.Popen) -> bool:
        """Terminate the whole process group, escalating if it ignores SIGTERM.

        Killing only the child leaves whatever it spawned running - a test
        server holding a port, a watcher holding a file - and the next run
        fails for reasons that have nothing to do with the next run.
        """
        try:
            group = os.getpgid(process.pid)
        except (ProcessLookupError, PermissionError):
            process.kill()
            return False
        for sig, grace in ((signal.SIGTERM, 2.0), (signal.SIGKILL, 3.0)):
            try:
                os.killpg(group, sig)
            except (ProcessLookupError, PermissionError):
                return True
            try:
                process.wait(timeout=grace)
                return True
            except subprocess.TimeoutExpired:
                continue
        return True

    def _run_container(
        self,
        argv: list[str],
        workdir: Path,
        env: dict[str, str],
        timeout: float,
        stdin: str | None,
    ) -> SandboxResult:
        """Run inside a disposable container. The real isolation boundary."""
        relative = workdir.relative_to(self.jail.root)
        docker = [
            # `docker run` attaches stdout and stderr by default; passing
            # --attach explicitly overrides that set and silently loses output.
            "docker",
            "run",
            "--rm",
            *(["--interactive"] if stdin is not None else []),
            # Drop everything not needed, then add nothing back.
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(self.config.max_processes),
            "--memory",
            f"{self.config.max_memory_mb}m",
            "--cpus",
            "1",
            "--workdir",
            f"/workspace/{relative}" if str(relative) != "." else "/workspace",
            "--volume",
            f"{self.jail.root}:/workspace",
        ]
        if not self.config.allow_network:
            # Actual network isolation, as opposed to the proxy hint the
            # subprocess layer can offer.
            docker += ["--network", "none"]
        for key in ("PATH", "LANG", "HOME", "PYTHONHASHSEED"):
            if key in env:
                docker += ["--env", f"{key}={env[key]}"]
        docker += [self.config.container_image, *argv]

        started = time.monotonic()
        try:
            completed = subprocess.run(  # noqa: S603 - argv list, never a shell
                docker,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                input=stdin,
            )
        except FileNotFoundError:
            raise SandboxError(
                "container mode is enabled but docker was not found",
                hint="install docker, or set GANTRY_SANDBOX_CONTAINER=0",
            ) from None
        except subprocess.TimeoutExpired as exc:
            return SandboxResult(
                exit_code=124,
                stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
                stderr="[container timed out]",
                duration_ms=(time.monotonic() - started) * 1000.0,
                timed_out=True,
                command=tuple(argv),
                mode="container",
            )

        cap = self.config.max_output_bytes
        stdout, stderr = completed.stdout or "", completed.stderr or ""
        return SandboxResult(
            exit_code=completed.returncode,
            stdout=stdout[:cap],
            stderr=stderr[:cap],
            duration_ms=(time.monotonic() - started) * 1000.0,
            truncated=len(stdout) > cap or len(stderr) > cap,
            command=tuple(argv),
            mode="container",
        )

    def describe(self) -> dict[str, Any]:
        return {
            "root": str(self.jail.root),
            "mode": "container" if self.config.use_container else "subprocess",
            "network": "allowed" if self.config.allow_network else "blocked",
            "limits": {
                "wall_clock_s": self.config.timeout_s,
                "cpu_s": self.config.max_cpu_seconds,
                "memory_mb": self.config.max_memory_mb,
                "processes": self.config.max_processes,
                "file_size_mb": self.config.max_file_size_mb,
                "output_bytes": self.config.max_output_bytes,
            },
            "env_allowlist": list(ENV_ALLOWLIST),
            "policy": self.policy.describe(),
            "denials": dict(self.counters.denials),
        }


__all__ = [
    "ENV_ALLOWLIST",
    "CommandDenied",
    "SandboxResult",
    "SandboxRunner",
]
