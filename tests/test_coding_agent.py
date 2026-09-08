"""End-to-end: a real agent, real tools, a real bug, a real test suite.

Everything below runs offline. The provider is deterministic, so these are
tests rather than demonstrations, but the path they exercise is the whole
product: the model asks for a file, the sandbox reads it, the model edits it,
the sandbox runs pytest, and the verification gates decide whether the claim
of completion is true.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gantry.config import BudgetConfig, Config, SandboxConfig
from gantry.contract import StopReason
from gantry.messages import Role
from gantry.providers.base import CompletionRequest
from gantry.providers.offline import OfflineProvider, Turn
from gantry.toolkit import build_coding_agent
from gantry.tools import Grant
from gantry.verify import CommandGate, GateSet, PythonSyntaxGate

CALC = "def add(a, b):\n    return a - b\n"
TEST = "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "calc.py").write_text(CALC)
    (tmp_path / "test_calc.py").write_text(TEST)
    return tmp_path


def config() -> Config:
    return Config(
        sandbox=SandboxConfig(timeout_s=60.0),
        budget=BudgetConfig(max_steps=10, wall_clock_s=180.0),
    )


def gates() -> GateSet:
    return GateSet([PythonSyntaxGate(), CommandGate("tests", "python -m pytest -q")])


def last_tool_text(request: CompletionRequest) -> str:
    for message in reversed(request.messages):
        if message.role is Role.TOOL:
            return message.content or ""
    return ""


def transcript(request: CompletionRequest) -> str:
    return "\n".join(m.content or "" for m in request.messages)


def test_the_agent_fixes_a_failing_test_end_to_end(project: Path):
    """The headline path, driven by a policy that reacts to tool output."""

    def policy(request: CompletionRequest, index: int) -> Turn:
        seen = last_tool_text(request)
        if not seen:
            return Turn(calls=(("read_file", {"path": "calc.py", "offset": None, "limit": None}),))
        if "return a - b" in seen:
            return Turn(
                calls=(
                    (
                        "edit_file",
                        {
                            "path": "calc.py",
                            "old_string": "return a - b",
                            "new_string": "return a + b",
                            "replace_all": None,
                        },
                    ),
                )
            )
        if seen.startswith("Edited"):
            return Turn(
                calls=(
                    ("bash", {"command": "python -m pytest -q", "cwd": None, "timeout_s": None}),
                )
            )
        return Turn(
            calls=(("finish", {"status": "completed", "summary": "add() now returns the sum."}),)
        )

    agent = build_coding_agent(
        OfflineProvider(policy=policy), project, config=config(), gates=gates()
    )
    result = agent.run("The test in test_calc.py fails. Fix calc.py.")

    assert result.stop_reason is StopReason.COMPLETED
    assert result.succeeded
    assert (project / "calc.py").read_text() == "def add(a, b):\n    return a + b\n"
    assert result.gate_report is not None and result.gate_report.passed
    # The whole run: read, edit, run the tests, claim completion.
    assert result.usage["tool_calls"] >= 3


def test_a_premature_claim_is_caught_and_the_feedback_names_the_failure(project: Path):
    """The agent says it is done before doing anything. The gate disagrees."""
    seen_feedback: list[str] = []

    def policy(request: CompletionRequest, index: int) -> Turn:
        if index == 0:
            return Turn(
                calls=(("finish", {"status": "completed", "summary": "Looks fine to me."}),)
            )
        if index == 1:
            seen_feedback.append(transcript(request))
            return Turn(calls=(("read_file", {"path": "calc.py", "offset": None, "limit": None}),))
        if index == 2:
            return Turn(
                calls=(
                    (
                        "edit_file",
                        {
                            "path": "calc.py",
                            "old_string": "return a - b",
                            "new_string": "return a + b",
                            "replace_all": None,
                        },
                    ),
                )
            )
        return Turn(calls=(("finish", {"status": "completed", "summary": "Fixed the operator."}),))

    agent = build_coding_agent(
        OfflineProvider(policy=policy), project, config=config(), gates=gates()
    )
    result = agent.run("Make the test pass.")

    assert seen_feedback, "the agent was never told its claim failed"
    feedback = seen_feedback[0]
    assert "tests" in feedback
    assert "test_add" in feedback or "assert" in feedback
    assert result.stop_reason is StopReason.COMPLETED
    assert (project / "calc.py").read_text().endswith("a + b\n")


def test_a_read_only_grant_refuses_the_edit_rather_than_performing_it(project: Path):
    """Capability enforcement is the dispatcher's, not the tool's."""

    def policy(request: CompletionRequest, index: int) -> Turn:
        if index == 0:
            return Turn(
                calls=(
                    (
                        "edit_file",
                        {
                            "path": "calc.py",
                            "old_string": "return a - b",
                            "new_string": "return a + b",
                            "replace_all": None,
                        },
                    ),
                )
            )
        return Turn(calls=(("finish", {"status": "blocked", "summary": "No write access."}),))

    agent = build_coding_agent(
        OfflineProvider(policy=policy),
        project,
        config=config(),
        gates=GateSet(),
        grant=Grant.read_only(),
    )
    result = agent.run("Fix the bug.")

    assert (project / "calc.py").read_text() == CALC
    assert result.stop_reason is StopReason.REFUSED
    assert "No write access" in result.summary


def test_the_tools_offered_to_the_model_match_the_grant(project: Path):
    agent = build_coding_agent(
        OfflineProvider(script=None), project, config=config(), grant=Grant.read_only()
    )
    offered = {t["function"]["name"] for t in agent.registry.to_openai_tools(agent.grant)}
    assert "bash" not in offered and "edit_file" not in offered
    assert {"read_file", "grep", "glob", "finish"} <= offered


def test_a_run_produces_a_trace_covering_tools_and_the_sandbox(project: Path, tracer):
    def policy(request: CompletionRequest, index: int) -> Turn:
        if index == 0:
            return Turn(
                calls=(
                    ("bash", {"command": 'python -c "print(1)"', "cwd": None, "timeout_s": None}),
                )
            )
        return Turn(calls=(("finish", {"status": "completed", "summary": "ran it"}),))

    agent = build_coding_agent(
        OfflineProvider(policy=policy), project, config=config(), gates=GateSet(), tracer=tracer
    )
    result = agent.run("Run something.")

    traces = tracer.exporter.traces
    assert len(traces) == 1
    names = {span.name for span in traces[0].spans}
    # The sandbox span proves the runner traces into the agent's tracer rather
    # than into whichever one it happened to be constructed with.
    assert "sandbox.run" in names, "the sandbox span is missing from the run's trace"
    assert any("bash" in name for name in names)
    assert traces[0].cost_usd == 0.0  # the offline model is priced at zero
    assert result.trace_id == traces[0].trace_id
