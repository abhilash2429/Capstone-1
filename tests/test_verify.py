"""Verification gates.

A model asked whether it finished will usually say yes. These tests cover the
mechanism that decides instead of asking, and the property that makes it
useful: a failure comes back as something the next turn can act on.
"""

from __future__ import annotations

import sys

import pytest

from gantry.config import SandboxConfig
from gantry.sandbox import CommandPolicy, PathJail, SandboxRunner
from gantry.verify import (
    CommandGate,
    FileContainsGate,
    FileExistsGate,
    GateContext,
    GateReport,
    GateSet,
    PredicateGate,
    PythonSyntaxGate,
    Verification,
    default_python_gates,
    gates_from_spec,
)


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "good.py").write_text("def add(a, b):\n    return a + b\n")
    (root / "notes.md").write_text("# Notes\nTODO: fix the parser\n")
    return root


@pytest.fixture
def context(workspace) -> GateContext:
    jail = PathJail(workspace)
    return GateContext(
        jail=jail,
        runner=SandboxRunner(
            jail, CommandPolicy(), SandboxConfig(root=str(workspace), timeout_s=60)
        ),
    )


# --- command gate ----------------------------------------------------------
def test_a_passing_command_passes(context):
    assert CommandGate("ok", [sys.executable, "-c", "pass"]).check(context).passed


def test_a_failing_command_reports_its_output(context):
    """The agent needs the stack trace, not a verdict."""
    gate = CommandGate("boom", [sys.executable, "-c", "raise SystemExit(2)"])
    result = gate.check(context)
    assert not result.passed
    assert "exit code 2" in result.detail
    assert result.evidence


def test_a_non_zero_expectation_is_honoured(context):
    gate = CommandGate("fails", [sys.executable, "-c", "raise SystemExit(3)"], expect_exit=3)
    assert gate.check(context).passed


def test_a_gate_runs_inside_the_sandbox(context):
    """A gate that shells out unconfined is a hole straight through the
    containment the rest of the harness provides."""
    result = CommandGate("escape", "sudo ls").check(context)
    assert not result.passed
    assert result.inconclusive
    assert "refused by policy" in result.detail


def test_a_gate_with_no_runner_is_inconclusive_not_failed(workspace):
    """A broken harness must not be reported as a failing agent."""
    context = GateContext(jail=PathJail(workspace), runner=None)
    result = CommandGate("tests", "pytest").check(context)
    assert not result.passed
    assert result.inconclusive


def test_a_timeout_is_a_failure_with_an_explanation(context):
    gate = CommandGate("slow", [sys.executable, "-c", "import time;time.sleep(30)"], timeout_s=1.0)
    result = gate.check(context)
    assert not result.passed
    assert "timed out" in result.detail


# --- file gates ------------------------------------------------------------
def test_file_existence_both_ways(context):
    assert FileExistsGate("good.py").check(context).passed
    assert not FileExistsGate("missing.py").check(context).passed
    assert FileExistsGate("missing.py", should_exist=False).check(context).passed


def test_a_path_outside_the_workspace_fails_the_gate(context):
    result = FileExistsGate("/etc/passwd").check(context)
    assert not result.passed


def test_file_content_both_ways(context):
    assert FileContainsGate("good.py", r"a \+ b").check(context).passed
    assert FileContainsGate("notes.md", "TODO", should_match=False).check(context).detail
    assert FileContainsGate("notes.md", "DONE", should_match=False).check(context).passed


def test_an_unreadable_file_fails_with_a_reason(context):
    result = FileContainsGate("nope.txt", "x").check(context)
    assert not result.passed
    assert "could not read" in result.detail


# --- syntax gate -----------------------------------------------------------
def test_syntax_gate_passes_on_valid_files(context):
    assert PythonSyntaxGate().check(context).passed


def test_syntax_gate_catches_an_edit_that_broke_a_file(context, workspace):
    """The failure that wastes the most turns: after it, every other tool
    reports a confusing secondary error and the agent chases the wrong bug."""
    (workspace / "broken.py").write_text("def f(:\n    pass\n")
    result = PythonSyntaxGate().check(context)
    assert not result.passed
    assert "broken.py" in result.evidence
    assert "1 file(s) do not parse" in result.detail


def test_syntax_gate_ignores_unreadable_files(context, workspace):
    (workspace / "binary.py").write_bytes(b"\xff\xfe\x00not utf-8")
    assert PythonSyntaxGate().check(context).passed


# --- predicate gate --------------------------------------------------------
def test_a_predicate_gate_can_return_a_reason(context):
    gate = PredicateGate("custom", lambda ctx: (False, "the invariant does not hold"))
    result = gate.check(context)
    assert not result.passed
    assert result.detail == "the invariant does not hold"


def test_a_raising_predicate_is_inconclusive(context):
    def broken(ctx):
        raise RuntimeError("the gate itself is buggy")

    result = PredicateGate("broken", broken).check(context)
    assert result.inconclusive
    assert "the gate itself raised" in result.detail


# --- gate sets -------------------------------------------------------------
def test_all_gates_run_so_one_turn_can_fix_several_problems(context, workspace):
    """Stopping at the first failure makes the agent play whack-a-mole, one
    turn and one provider call per issue."""
    (workspace / "broken.py").write_text("def f(:\n")
    gates = GateSet(
        [
            PythonSyntaxGate(),
            FileExistsGate("missing.py"),
            CommandGate("fails", [sys.executable, "-c", "raise SystemExit(1)"]),
        ]
    )
    report = gates.run(context)
    assert len(report.failures) == 3


def test_stopping_early_is_available_when_asked(context, workspace):
    (workspace / "broken.py").write_text("def f(:\n")
    gates = GateSet([PythonSyntaxGate(), FileExistsGate("missing.py")])
    report = gates.run(context, stop_on_first_failure=True)
    assert len(report.verifications) == 1


def test_feedback_names_every_failure_and_asks_for_a_fix(context):
    gates = GateSet([FileExistsGate("a.py"), FileExistsGate("b.py")])
    feedback = gates.run(context).feedback()
    assert "exists:a.py" in feedback and "exists:b.py" in feedback
    assert "Do not claim completion" in feedback


def test_feedback_marks_an_inconclusive_check_as_a_harness_problem(workspace):
    context = GateContext(jail=PathJail(workspace), runner=None)
    feedback = GateSet([CommandGate("tests", "pytest")]).run(context).feedback()
    assert "harness problem, not yours" in feedback


def test_a_passing_report_has_nothing_to_say(context):
    assert GateSet([PythonSyntaxGate()]).run(context).feedback() == ""


def test_an_empty_gate_set_passes(context):
    assert GateSet().run(context).passed


def test_a_gate_run_is_traced(context, tracer, exporter):
    gates = GateSet([PythonSyntaxGate(), FileExistsGate("missing.py")], tracer=tracer)
    with tracer.trace("agent.run"):
        gates.run(context)
    span = next(s for s in exporter.traces[0].spans if s.kind == "verify")
    assert span.attributes["gantry.verify.failed"] == 1
    assert span.status == "error"


# --- construction ----------------------------------------------------------
def test_gates_can_be_built_from_declarative_data():
    gates = gates_from_spec(
        [
            {"type": "command", "name": "tests", "command": "pytest -q"},
            {"type": "file_exists", "path": "README.md"},
            {"type": "python_syntax"},
        ]
    )
    assert [g.name for g in gates.gates] == ["tests", "exists:README.md", "python-syntax"]


def test_an_unknown_gate_type_is_refused_with_the_valid_ones():
    with pytest.raises(ValueError, match="unknown gate type"):
        gates_from_spec([{"type": "telepathy"}])


def test_the_default_python_set_checks_syntax_and_tests():
    assert [g.name for g in default_python_gates().gates] == ["python-syntax", "tests"]


def test_a_report_serialises_for_the_dashboard():
    report = GateReport(
        [
            Verification("a", passed=True),
            Verification("b", passed=False, detail="nope"),
        ]
    )
    data = report.as_dict()
    assert data["passed"] is False
    assert data["failed"] == ["b"]
