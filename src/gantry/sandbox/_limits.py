"""Apply resource limits, then become the target command.

Run as ``python -m gantry.sandbox._limits --cpu 30 -- pytest -x``.

The obvious way to set rlimits on a child is ``subprocess``'s ``preexec_fn``,
and it is a trap in this codebase. ``preexec_fn`` runs between ``fork`` and
``exec`` in a process that has just inherited every lock held by every other
thread at the moment of the fork. The dispatcher is threaded, so a hook that
allocates or logs there can deadlock a child permanently and intermittently -
the worst failure mode there is, because it reproduces once a week.

Doing it in a real process that then ``exec``s the target avoids the fork
hazard entirely: limits are set in a single-threaded process, and the
``execvp`` replaces it, so the final process really is the requested command
with the limits applied. The cost is one interpreter start-up per command.
"""

from __future__ import annotations

import contextlib
import os
import resource
import sys


def _set(limit: int, value: int) -> None:
    """Apply a soft limit, clamped to the hard limit we are allowed to set."""
    try:
        _soft, hard = resource.getrlimit(limit)
    except (OSError, ValueError):
        return
    if hard != resource.RLIM_INFINITY:
        value = min(value, hard)
    # A limit the platform will not accept must not stop the command: the
    # caller still has the wall-clock timeout and the process-group kill.
    with contextlib.suppress(OSError, ValueError):
        resource.setrlimit(limit, (value, hard))


def main(argv: list[str]) -> int:
    options: dict[str, int] = {}
    rest: list[str] = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--":
            rest = argv[index + 1 :]
            break
        if argument.startswith("--") and index + 1 < len(argv):
            options[argument[2:].replace("-", "_")] = int(argv[index + 1])
            index += 2
            continue
        index += 1

    if not rest:
        print("usage: _limits [--cpu N] [--mem-mb N] ... -- command args", file=sys.stderr)
        return 2

    if "cpu" in options:
        # A hard CPU ceiling that survives a command ignoring SIGTERM, which
        # wall-clock timeouts alone do not give you.
        _set(resource.RLIMIT_CPU, options["cpu"])
    if "mem_mb" in options:
        _set(resource.RLIMIT_AS, options["mem_mb"] * 1024 * 1024)
    if "fsize_mb" in options:
        _set(resource.RLIMIT_FSIZE, options["fsize_mb"] * 1024 * 1024)
    if "nproc" in options:
        _set(resource.RLIMIT_NPROC, options["nproc"])
    if "nofile" in options:
        _set(resource.RLIMIT_NOFILE, options["nofile"])
    # Never write a core dump: they are large, and they contain whatever was in
    # memory at the time.
    _set(resource.RLIMIT_CORE, 0)

    try:
        # Replacing this process is the entire point: the limits set above stay
        # in force, and the resulting process really is the requested command
        # rather than a wrapper holding it.
        os.execvp(rest[0], rest)  # noqa: S606 - argv list from a parsed policy decision
    except FileNotFoundError:
        print(f"{rest[0]}: command not found", file=sys.stderr)
        return 127
    except PermissionError:
        print(f"{rest[0]}: permission denied", file=sys.stderr)
        return 126
    except OSError as exc:
        print(f"{rest[0]}: {exc}", file=sys.stderr)
        return 126
    return 0  # pragma: no cover - execvp does not return on success


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
