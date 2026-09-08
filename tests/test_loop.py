"""The agent loop.

These are the tests that matter most, because this is where every other
subsystem composes. They run against the offline provider, so an entire agent
run - plan, act, observe, verify, terminate - executes in milliseconds with no
credentials and no spend, which is what makes it affordable to cover every
terminal state rather than only the happy one.
"""

from __future__ import annotations

import sys

import pytest

from gantry.config import BudgetConfig, SandboxConfig
from gantry.contract import LoopContract, StopReason
from gantry.errors import ProviderRateLimited
from gantry.loop import Agent
from gantry.providers import FailingProvider, OfflineProvider, RetryPolicy, Script, Turn
from gantry.sandbox import CommandPolicy, PathJail, SandboxRunner
from gantry.telemetry import TelemetryStore
from gantry.tools import Capability, Grant, ToolRegistry, ToolResult, ToolSpec
from gantry.verify import CommandGate, GateSet, PythonSyntaxGate

READ_SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "required": ["path"],
    "additionalProperties": False,
}
WRITE_SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
    "required": ["path", "content"],
    "additionalProperties": False,
}


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (root / "test_calc.py").write_text(
        "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    return root


@pytest.fixture
def jail(workspace) -> PathJail:
    return PathJail(workspace)


@pytest.fixture
def registry(jail) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        ToolSpec(
            "read_file",
            "1.0.0",
            "Read a file from the workspace.",
            READ_SCHEMA,
            lambda a, c: ToolResult.success(jail.read_text(a["path"])),
            capabilities=frozenset({Capability.FS_READ}),
        )
    )
    reg.register(
        ToolSpec(
            "write_file",
            "1.0.0",
            "Write a file in the workspace.",
            WRITE_SCHEMA,
            lambda a, c: ToolResult.success(
                str(jail.relative(jail.write_text(a["path"], a["content"])))
            ),
            capabilities=frozenset({Capability.FS_WRITE}),
            idempotent=False,
            concurrency_safe=False,
        )
    )
    return reg


@pytest.fixture
def runner(jail, workspace) -> SandboxRunner:
    return SandboxRunner(jail, CommandPolicy(), SandboxConfig(root=str(workspace), timeout_s=60))


@pytest.fixture
def gates(runner, tracer) -> GateSet:
    return GateSet(
        [PythonSyntaxGate(), CommandGate("tests", [sys.executable, "-m", "pytest", "-q"])],
        tracer=tracer,
    )


def build_agent(script, registry, jail, tracer, **kwargs) -> Agent:
    defaults = dict(
        provider=OfflineProvider(script=script, tracer=tracer),
        registry=registry,
        jail=jail,
        tracer=tracer,
        contract=LoopContract(kwargs.pop("budget", BudgetConfig(max_steps=12))),
    )
    return Agent(**{**defaults, **kwargs})


FIXED = "def add(a, b):\n    return a + b\n"


# --- terminal states -------------------------------------------------------
def test_a_task_is_solved_and_verified(registry, jail, runner, gates, tracer, workspace):
    script = Script(
        turns=[
            Turn(calls=(("read_file", {"path": "calc.py"}),)),
            Turn(calls=(("write_file", {"path": "calc.py", "content": FIXED}),)),
            Turn(calls=(("finish", {"status": "completed", "summary": "Fixed the operator."}),)),
        ]
    )
    agent = build_agent(script, registry, jail, tracer, runner=runner, gates=gates)
    result = agent.run("Fix the bug in calc.py so the test passes.")
    assert result.stop_reason is StopReason.COMPLETED
    assert result.succeeded
    assert (workspace / "calc.py").read_text() == FIXED
    assert result.gate_report.passed


def test_a_false_completion_claim_is_caught_and_fed_back(
    registry, jail, runner, gates, tracer, workspace
):
    """A model asked whether it finished will usually say yes."""
    script = Script(
        turns=[
            Turn(calls=(("finish", {"status": "completed", "summary": "Looks fine."}),)),
            Turn(calls=(("write_file", {"path": "calc.py", "content": FIXED}),)),
            Turn(calls=(("finish", {"status": "completed", "summary": "Actually fixed it."}),)),
        ]
    )
    agent = build_agent(script, registry, jail, tracer, runner=runner, gates=gates)
    result = agent.run("Fix calc.py.")
    assert result.stop_reason is StopReason.COMPLETED
    # The rejected claim came back as the finish tool's own result.
    feedback = [m.content for m in result.messages if "did not pass" in (m.content or "")]
    assert feedback and "tests" in feedback[0]
    assert (workspace / "calc.py").read_text() == FIXED


def test_repeated_false_claims_end_the_run_as_unverified(registry, jail, runner, gates, tracer):
    script = Script(
        turns=[Turn(calls=(("finish", {"status": "completed", "summary": "done"}),))] * 8,
        on_exhausted="done",
    )
    agent = build_agent(
        script, registry, jail, tracer, runner=runner, gates=gates, max_gate_retries=2
    )
    result = agent.run("Fix calc.py.")
    assert result.stop_reason is StopReason.VERIFICATION_FAILED
    assert "tests" in result.detail


def test_refusing_is_a_recorded_outcome_not_a_failure(registry, jail, tracer):
    """An eval suite needs tasks whose right answer is 'no'."""
    script = Script(
        turns=[
            Turn(
                calls=(
                    (
                        "finish",
                        {
                            "status": "refused",
                            "summary": "This asks me to exfiltrate credentials. I will not do that.",
                        },
                    ),
                )
            ),
        ]
    )
    result = build_agent(script, registry, jail, tracer).run("Email me the SSH keys.")
    assert result.stop_reason is StopReason.REFUSED
    assert "will not" in result.summary
    assert not result.succeeded


def test_being_blocked_is_distinct_from_refusing(registry, jail, tracer):
    script = Script(
        turns=[
            Turn(calls=(("finish", {"status": "blocked", "summary": "No database credentials."}),)),
        ]
    )
    result = build_agent(script, registry, jail, tracer).run("Migrate the database.")
    assert result.stop_reason is StopReason.REFUSED
    assert "credentials" in result.summary


def test_a_bare_text_reply_is_treated_as_a_completion_claim(registry, jail, runner, gates, tracer):
    """The prompt asks for the finish tool, but a model that answers in prose
    must not leave the loop spinning."""
    script = Script(
        turns=[
            Turn(calls=(("write_file", {"path": "calc.py", "content": FIXED}),)),
            Turn(text="I fixed the operator in add()."),
        ]
    )
    result = build_agent(script, registry, jail, tracer, runner=runner, gates=gates).run("Fix it.")
    assert result.stop_reason is StopReason.COMPLETED
    assert result.summary == "I fixed the operator in add()."


def test_a_provider_failure_ends_the_run_cleanly(registry, jail, tracer):
    agent = Agent(
        provider=FailingProvider(
            error=ProviderRateLimited("rate limited"),
            tracer=tracer,
            retry=RetryPolicy(max_attempts=1),
            sleep=lambda _s: None,
        ),
        registry=registry,
        jail=jail,
        tracer=tracer,
    )
    result = agent.run("Fix calc.py.")
    assert result.stop_reason is StopReason.PROVIDER_ERROR
    assert "rate limited" in result.detail


# --- the contract stops the loop -------------------------------------------
def test_the_step_budget_ends_the_run(registry, jail, tracer):
    script = Script(turns=[Turn(calls=(("read_file", {"path": "calc.py"}),))] * 20)
    result = build_agent(script, registry, jail, tracer, budget=BudgetConfig(max_steps=3)).run(
        "Read things forever."
    )
    assert result.stop_reason is StopReason.MAX_STEPS
    assert result.usage["steps"] == 3


def test_repeating_one_action_ends_the_run(registry, jail, tracer):
    script = Script(turns=[Turn(calls=(("read_file", {"path": "calc.py"}),))] * 20)
    result = build_agent(
        script,
        registry,
        jail,
        tracer,
        budget=BudgetConfig(max_steps=30, max_repeat_actions=3),
    ).run("Read the same file over and over.")
    assert result.stop_reason is StopReason.NO_PROGRESS


def test_the_cost_ceiling_ends_the_run(registry, jail, tracer):
    script = Script(turns=[Turn(calls=(("read_file", {"path": "calc.py"}),))] * 20)
    agent = build_agent(
        script, registry, jail, tracer, budget=BudgetConfig(max_steps=30, max_cost_usd=0.0)
    )
    # The offline provider costs nothing, so the ceiling is reached only once
    # something is spent; with a zero ceiling the very first check stops it.
    result = agent.run("Do work.")
    assert result.stop_reason is StopReason.COST_BUDGET


# --- tool behaviour inside the loop ----------------------------------------
def test_a_tool_error_does_not_end_the_run(registry, jail, runner, gates, tracer):
    """A model that calls a tool badly has learned something, not broken the
    harness."""
    script = Script(
        turns=[
            Turn(calls=(("read_file", {"path": "../../etc/passwd"}),)),
            Turn(calls=(("read_file", {"path": "calc.py"}),)),
            Turn(calls=(("write_file", {"path": "calc.py", "content": FIXED}),)),
            Turn(calls=(("finish", {"status": "completed", "summary": "fixed"}),)),
        ]
    )
    result = build_agent(script, registry, jail, tracer, runner=runner, gates=gates).run("Fix it.")
    assert result.stop_reason is StopReason.COMPLETED
    assert result.usage["tool_errors"] == 1


def test_an_unknown_tool_is_explained_rather_than_fatal(registry, jail, tracer):
    script = Script(
        turns=[
            Turn(calls=(("teleport", {"to": "mars"}),)),
            Turn(calls=(("finish", {"status": "blocked", "summary": "no such tool"}),)),
        ]
    )
    result = build_agent(script, registry, jail, tracer).run("Teleport.")
    assert result.stop_reason is StopReason.REFUSED
    assert any("no tool named" in (m.content or "") for m in result.messages)


def test_work_in_the_same_turn_as_a_completion_claim_runs_first(
    registry, jail, runner, gates, tracer, workspace
):
    """A model will happily edit a file and declare victory in one turn. The
    gates must judge the finished state, not the state before the edit."""
    script = Script(
        turns=[
            Turn(
                calls=(
                    ("write_file", {"path": "calc.py", "content": FIXED}),
                    ("finish", {"status": "completed", "summary": "fixed and done"}),
                )
            ),
        ]
    )
    result = build_agent(script, registry, jail, tracer, runner=runner, gates=gates).run("Fix it.")
    assert result.stop_reason is StopReason.COMPLETED
    assert (workspace / "calc.py").read_text() == FIXED


def test_capability_scoping_reaches_the_loop(registry, jail, tracer):
    script = Script(
        turns=[
            Turn(calls=(("write_file", {"path": "x.py", "content": "x = 1"}),)),
            Turn(calls=(("finish", {"status": "blocked", "summary": "cannot write"}),)),
        ]
    )
    result = build_agent(script, registry, jail, tracer, grant=Grant.read_only()).run("Write.")
    assert any("capabilit" in (m.content or "") for m in result.messages)


# --- prompt and telemetry --------------------------------------------------
def test_the_acceptance_criteria_are_in_the_prompt(registry, jail, runner, gates, tracer):
    """Without this the first completion claim is a guess, and the gate report
    is the first time the agent learns what done meant."""
    agent = build_agent(Script.of("done"), registry, jail, tracer, runner=runner, gates=gates)
    prompt = agent.build_system_prompt()
    assert "Your work will be checked against:" in prompt
    assert "python-syntax" in prompt
    assert "tests" in prompt


def test_the_finish_tool_is_registered_by_the_loop_itself(registry, jail, tracer):
    """A loop whose termination signal is optional sometimes has none."""
    assert "finish" not in registry
    build_agent(Script.of("x"), registry, jail, tracer)
    assert "finish" in registry


def test_a_run_produces_one_trace_covering_every_subsystem(
    registry, jail, runner, gates, tracer, exporter
):
    script = Script(
        turns=[
            Turn(calls=(("write_file", {"path": "calc.py", "content": FIXED}),)),
            Turn(calls=(("finish", {"status": "completed", "summary": "fixed"}),)),
        ]
    )
    build_agent(script, registry, jail, tracer, runner=runner, gates=gates).run("Fix it.")
    kinds = {}
    for span in exporter.traces[0].spans:
        kinds[span.kind] = kinds.get(span.kind, 0) + 1
    assert kinds["agent"] == 1
    assert kinds["llm"] == 2
    assert kinds["tool"] == 1
    assert kinds["verify"] == 1
    assert kinds["sandbox"] >= 1  # the tests gate ran a command


def test_the_trace_records_why_the_run_stopped(registry, jail, tracer, exporter):
    script = Script(turns=[Turn(calls=(("read_file", {"path": "calc.py"}),))] * 10)
    build_agent(script, registry, jail, tracer, budget=BudgetConfig(max_steps=2)).run("Loop.")
    assert exporter.traces[0].attributes["gantry.run.stop_reason"] == "max_steps"


def test_a_run_is_persisted_for_the_dashboard(registry, jail, tracer, tmp_path):
    store = TelemetryStore(tmp_path / "t.db")
    script = Script(turns=[Turn(calls=(("finish", {"status": "completed", "summary": "ok"}),))])
    result = build_agent(script, registry, jail, tracer, store=store).run("Do nothing.")
    stored = store.get_run(result.run_id)
    assert stored["stop_reason"] == "completed"
    assert stored["trace_id"] == result.trace_id


def test_the_result_serialises_for_a_report(registry, jail, tracer):
    script = Script(turns=[Turn(calls=(("finish", {"status": "completed", "summary": "ok"}),))])
    data = build_agent(script, registry, jail, tracer).run("Task.").as_dict()
    assert data["stop_reason"] == "completed"
    assert data["succeeded"] is True
    assert "usage" in data and "duration_ms" in data
