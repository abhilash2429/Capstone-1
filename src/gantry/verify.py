"""Verification gates: nothing is done because the agent says so.

A model asked whether it finished will usually say yes. That is not
dishonesty, it is the absence of a way to check, and the harness is where the
checking belongs. A gate is a post-condition the work has to satisfy before
the loop accepts a completion claim: the tests pass, the file exists, the
module still imports.

Two properties make gates worth having rather than merely reassuring.

**A failed gate is feedback, not a verdict.** The failure is rendered as a
message and handed back, so the next turn gets the compiler error rather than
a run marked failed with no explanation. Most gate failures are recoverable in
one more step.

**Gates run in the sandbox.** A gate that shells out unconfined is a hole
straight through the containment the rest of the harness provides, and it would
be the obvious place to put one.
"""

from __future__ import annotations

import ast
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from gantry.sandbox import PathJail, SandboxRunner
from gantry.telemetry import semconv as sc
from gantry.telemetry.tracer import Tracer, get_tracer

#: How much gate output to hand back to the model. Enough for a stack trace,
#: not enough for a full test log.
MAX_EVIDENCE_CHARS = 4_000


@dataclass
class GateContext:
    """What a gate is allowed to look at."""

    jail: PathJail
    runner: SandboxRunner | None = None
    task: str = ""
    #: Files the agent touched this run, when the caller tracked them.
    changed_files: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Verification:
    """The result of one gate."""

    name: str
    passed: bool
    detail: str = ""
    evidence: str = ""
    duration_ms: float = 0.0
    #: A gate that could not run at all, as distinct from one that ran and
    #: failed. Conflating the two turns a broken harness into a failing agent.
    inconclusive: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "inconclusive": self.inconclusive,
            "detail": self.detail,
            "duration_ms": round(self.duration_ms, 2),
        }


class Gate(ABC):
    """A post-condition the work must satisfy."""

    name: str = "gate"
    #: Shown to the model when the gate fails, so it knows what is expected.
    description: str = ""

    @abstractmethod
    def check(self, context: GateContext) -> Verification: ...

    def _timed(self, started: float, **kwargs: Any) -> Verification:
        return Verification(
            name=self.name, duration_ms=(time.monotonic() - started) * 1000.0, **kwargs
        )


class CommandGate(Gate):
    """Passes when a command exits with the expected status.

    The workhorse: ``pytest -q``, ``ruff check``, ``npm test``, ``go build``.
    Runs through the sandbox, so a gate cannot become the unconfined hole in
    the middle of a confined system.
    """

    def __init__(
        self,
        name: str,
        command: str | list[str],
        expect_exit: int = 0,
        description: str = "",
        timeout_s: float | None = None,
        cwd: str | None = None,
    ) -> None:
        self.name = name
        self.command = command
        self.expect_exit = expect_exit
        self.description = description or f"`{command}` must exit {expect_exit}"
        self.timeout_s = timeout_s
        self.cwd = cwd

    def check(self, context: GateContext) -> Verification:
        started = time.monotonic()
        if context.runner is None:
            return self._timed(
                started,
                passed=False,
                inconclusive=True,
                detail="no sandbox runner is available to execute this gate",
            )
        result = context.runner.run(self.command, cwd=self.cwd, timeout_s=self.timeout_s)
        if result.denied is not None:
            return self._timed(
                started,
                passed=False,
                inconclusive=True,
                detail=f"the gate command was refused by policy: {result.denied.reason}",
            )
        passed = result.exit_code == self.expect_exit and not result.timed_out
        detail = (
            f"timed out after {self.timeout_s or 'the default'}s"
            if result.timed_out
            else f"exit code {result.exit_code}, expected {self.expect_exit}"
        )
        return self._timed(
            started,
            passed=passed,
            detail="" if passed else detail,
            evidence="" if passed else result.combined_output(MAX_EVIDENCE_CHARS),
        )


class FileExistsGate(Gate):
    """Passes when a path exists inside the workspace."""

    def __init__(self, path: str, name: str = "", should_exist: bool = True) -> None:
        self.path = path
        self.should_exist = should_exist
        self.name = name or f"{'exists' if should_exist else 'absent'}:{path}"
        self.description = (
            f"{path} must {'exist' if should_exist else 'not exist'} in the workspace"
        )

    def check(self, context: GateContext) -> Verification:
        started = time.monotonic()
        try:
            exists = context.jail.resolve(self.path).exists()
        except Exception as exc:  # noqa: BLE001 - a bad path is a failed gate
            return self._timed(started, passed=False, detail=str(exc))
        passed = exists is self.should_exist
        return self._timed(
            started,
            passed=passed,
            detail=""
            if passed
            else f"{self.path} {'is missing' if self.should_exist else 'still exists'}",
        )


class FileContainsGate(Gate):
    """Passes when a file matches (or does not match) a pattern."""

    def __init__(self, path: str, pattern: str, name: str = "", should_match: bool = True) -> None:
        self.path = path
        self.pattern = re.compile(pattern)
        self.should_match = should_match
        self.name = name or f"contains:{path}"
        self.description = f"{path} must {'contain' if should_match else 'not contain'} /{pattern}/"

    def check(self, context: GateContext) -> Verification:
        started = time.monotonic()
        try:
            content = context.jail.read_text(self.path)
        except Exception as exc:  # noqa: BLE001 - unreadable is a failed gate
            return self._timed(started, passed=False, detail=f"could not read {self.path}: {exc}")
        matched = bool(self.pattern.search(content))
        passed = matched is self.should_match
        return self._timed(
            started,
            passed=passed,
            detail=""
            if passed
            else f"pattern /{self.pattern.pattern}/ "
            f"{'was not found' if self.should_match else 'is still present'} in {self.path}",
        )


class PythonSyntaxGate(Gate):
    """Passes when every Python file in the workspace still parses.

    Cheap, and it catches the failure mode that wastes the most turns: an edit
    that leaves a file unparseable, after which every other tool reports a
    confusing secondary error and the agent chases the wrong problem.
    """

    name = "python-syntax"
    description = "every .py file in the workspace must parse"

    def __init__(self, limit: int = 2_000) -> None:
        self.limit = limit

    def check(self, context: GateContext) -> Verification:
        started = time.monotonic()
        broken: list[str] = []
        for path in context.jail.iter_files("**/*.py", limit=self.limit):
            try:
                ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError as exc:
                broken.append(f"{context.jail.relative(path)}:{exc.lineno}: {exc.msg}")
            except (OSError, UnicodeDecodeError):
                continue  # unreadable is a different problem, not a syntax error
        return self._timed(
            started,
            passed=not broken,
            detail="" if not broken else f"{len(broken)} file(s) do not parse",
            evidence="\n".join(broken[:20]),
        )


class PredicateGate(Gate):
    """Passes when a Python callable returns True. For fixtures and tests."""

    def __init__(
        self,
        name: str,
        predicate: Callable[[GateContext], bool | tuple[bool, str]],
        description: str = "",
    ) -> None:
        self.name = name
        self.predicate = predicate
        self.description = description or name

    def check(self, context: GateContext) -> Verification:
        started = time.monotonic()
        try:
            outcome = self.predicate(context)
        except Exception as exc:  # noqa: BLE001 - a raising predicate is inconclusive
            return self._timed(
                started,
                passed=False,
                inconclusive=True,
                detail=f"the gate itself raised {type(exc).__name__}: {exc}",
            )
        passed, detail = outcome if isinstance(outcome, tuple) else (outcome, "")
        return self._timed(started, passed=bool(passed), detail=detail)


@dataclass
class GateReport:
    """The outcome of running every gate."""

    verifications: list[Verification] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(v.passed for v in self.verifications)

    @property
    def failures(self) -> list[Verification]:
        return [v for v in self.verifications if not v.passed]

    @property
    def inconclusive(self) -> list[Verification]:
        return [v for v in self.verifications if v.inconclusive]

    def feedback(self) -> str:
        """Render failures as an instruction the next turn can act on.

        This is the whole point of gating rather than merely judging: the agent
        gets the compiler error, not a verdict.
        """
        if self.passed:
            return ""
        lines = ["The work is not finished. These checks did not pass:", ""]
        for failure in self.failures:
            lines.append(f"- {failure.name}: {failure.detail or 'failed'}")
            if failure.inconclusive:
                lines.append("  (this check could not run; it is a harness problem, not yours)")
            if failure.evidence:
                lines.append("")
                lines.append("  " + failure.evidence.replace("\n", "\n  "))
                lines.append("")
        lines.append("Fix the cause and continue. Do not claim completion until they pass.")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [v.as_dict() for v in self.verifications],
            "failed": [v.name for v in self.failures],
        }


class GateSet:
    """Runs a collection of gates and reports on them."""

    def __init__(self, gates: Sequence[Gate] = (), tracer: Tracer | None = None) -> None:
        self.gates = list(gates)
        self.tracer = tracer or get_tracer()

    def __len__(self) -> int:
        return len(self.gates)

    def add(self, gate: Gate) -> GateSet:
        self.gates.append(gate)
        return self

    def run(self, context: GateContext, stop_on_first_failure: bool = False) -> GateReport:
        """Run every gate. All of them, unless asked otherwise.

        Running the whole set by default means one turn can fix several
        problems. Stopping at the first failure makes the agent play
        whack-a-mole, one turn and one provider call per issue.
        """
        report = GateReport()
        with self.tracer.span("verify", kind=sc.SpanKind.VERIFY) as span:
            for gate in self.gates:
                verification = gate.check(context)
                report.verifications.append(verification)
                if stop_on_first_failure and not verification.passed:
                    break
            span.set_attributes(
                {
                    "gantry.verify.total": len(report.verifications),
                    "gantry.verify.failed": len(report.failures),
                    "gantry.verify.passed": report.passed,
                }
            )
            if not report.passed:
                span.set_status("error", ",".join(v.name for v in report.failures))
        return report

    def describe(self) -> list[dict[str, str]]:
        return [{"name": g.name, "description": g.description} for g in self.gates]


def default_python_gates(test_command: str = "python -m pytest -q") -> GateSet:
    """A sensible starting set for a Python workspace."""
    return GateSet([PythonSyntaxGate(), CommandGate("tests", test_command)])


def gates_from_spec(spec: Sequence[dict[str, Any]]) -> GateSet:
    """Build gates from declarative data, for fixtures and configuration."""
    builders: dict[str, Callable[..., Gate]] = {
        "command": lambda **kw: CommandGate(**kw),
        "file_exists": lambda **kw: FileExistsGate(**kw),
        "file_contains": lambda **kw: FileContainsGate(**kw),
        "python_syntax": lambda **kw: PythonSyntaxGate(**kw),
    }
    gates: list[Gate] = []
    for entry in spec:
        options = dict(entry)
        kind = options.pop("type")
        if kind not in builders:
            raise ValueError(f"unknown gate type {kind!r}; expected one of {sorted(builders)}")
        gates.append(builders[kind](**options))
    return GateSet(gates)


__all__ = [
    "CommandGate",
    "FileContainsGate",
    "FileExistsGate",
    "Gate",
    "GateContext",
    "GateReport",
    "GateSet",
    "PredicateGate",
    "PythonSyntaxGate",
    "Verification",
    "default_python_gates",
    "gates_from_spec",
]
