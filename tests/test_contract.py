"""The loop contract is the only thing that can stop a run, so every one of its
termination paths is covered here. These tests use an injected clock and no
provider, which is the whole point: agent termination is ordinary control flow
and should be testable in microseconds."""

from __future__ import annotations

import pytest

from gantry.config import BudgetConfig
from gantry.contract import (
    Cancellation,
    LoopContract,
    Phase,
    RunState,
    StopReason,
    action_fingerprint,
)


@pytest.fixture
def contract(clock):
    return LoopContract(BudgetConfig(), clock=clock)


def test_a_fresh_run_may_proceed(contract):
    assert contract.before_step(contract.start_run("task")).allowed


def test_step_limit_stops_the_run(clock):
    contract = LoopContract(BudgetConfig(max_steps=3), clock=clock)
    state = contract.start_run()
    for _ in range(3):
        assert contract.before_step(state).allowed
        contract.record_step(state)
    decision = contract.before_step(state)
    assert decision.stop_reason is StopReason.MAX_STEPS
    assert "3" in decision.detail


def test_deadline_uses_the_contract_clock(clock):
    """Regression: a RunState built directly anchors to the wall clock, so the
    deadline silently never fires. Runs must be created via start_run."""
    contract = LoopContract(BudgetConfig(wall_clock_s=10.0), clock=clock)
    state = contract.start_run()
    clock.advance(9.99)
    assert contract.before_step(state).allowed
    clock.advance(0.01)
    assert contract.before_step(state).stop_reason is StopReason.DEADLINE


def test_cost_ceiling_is_checked_before_spending(clock):
    contract = LoopContract(BudgetConfig(max_cost_usd=1.0), clock=clock)
    state = contract.start_run()
    contract.record_provider_call(state, cost_usd=0.90)
    assert contract.can_afford(state, 0.05).allowed
    decision = contract.can_afford(state, 0.20)
    assert decision.stop_reason is StopReason.COST_BUDGET
    # The projected spend was refused, so nothing was actually charged.
    assert state.usage.cost_usd == pytest.approx(0.90)


def test_token_budget_counts_both_directions(clock):
    contract = LoopContract(BudgetConfig(max_tokens=1000), clock=clock)
    state = contract.start_run()
    contract.record_provider_call(state, input_tokens=600, output_tokens=400)
    assert contract.before_step(state).stop_reason is StopReason.TOKEN_BUDGET


def test_identical_repeated_actions_stop_the_run(clock):
    contract = LoopContract(BudgetConfig(max_repeat_actions=3), clock=clock)
    state = contract.start_run()
    for _ in range(3):
        assert contract.before_action(state, "bash", {"cmd": "pytest"}).allowed
        contract.record_action(state, "bash", {"cmd": "pytest"}, ok=False)
    assert contract.before_action(state, "bash", {"cmd": "pytest"}).stop_reason is (
        StopReason.NO_PROGRESS
    )
    # A genuinely different action is still allowed.
    assert contract.before_action(state, "bash", {"cmd": "pytest -x"}).allowed


def test_consecutive_tool_errors_stop_the_run_but_a_success_resets(clock):
    contract = LoopContract(BudgetConfig(max_consecutive_tool_errors=3), clock=clock)
    state = contract.start_run()
    for i in range(2):
        contract.record_action(state, "bash", {"i": i}, ok=False)
    contract.record_action(state, "bash", {"i": "ok"}, ok=True)
    assert state.usage.consecutive_tool_errors == 0
    assert contract.before_step(state).allowed
    for i in range(3):
        contract.record_action(state, "bash", {"j": i}, ok=False)
    assert contract.before_step(state).stop_reason is StopReason.TOOL_ERROR_BUDGET
    # Total errors are still tracked separately from the consecutive streak.
    assert state.usage.tool_errors == 5


def test_cancellation_is_observed_at_both_checkpoints(clock):
    cancel = Cancellation()
    contract = LoopContract(BudgetConfig(), clock=clock, cancellation=cancel)
    state = contract.start_run()
    assert contract.before_step(state).allowed
    cancel.cancel("user pressed ctrl-c")
    assert contract.before_step(state).stop_reason is StopReason.CANCELLED
    decision = contract.before_action(state, "bash", {})
    assert decision.stop_reason is StopReason.CANCELLED
    assert decision.detail == "user pressed ctrl-c"


def test_fingerprint_ignores_key_order_but_not_values():
    assert action_fingerprint("t", {"a": 1, "b": 2}) == action_fingerprint("t", {"b": 2, "a": 1})
    assert action_fingerprint("t", {"a": 1}) != action_fingerprint("t", {"a": 2})
    assert action_fingerprint("t", {"a": 1}) != action_fingerprint("u", {"a": 1})


def test_finish_records_the_terminal_state(contract):
    state = contract.start_run("task")
    contract.finish(state, StopReason.COMPLETED, "all checks passed")
    assert state.phase is Phase.DONE
    assert state.stop_reason.is_success
    assert state.stop_detail == "all checks passed"


def test_remaining_never_goes_negative(clock):
    contract = LoopContract(BudgetConfig(max_steps=2, max_cost_usd=1.0), clock=clock)
    state = contract.start_run()
    for _ in range(5):
        contract.record_step(state)
    contract.record_provider_call(state, cost_usd=3.0)
    remaining = contract.remaining(state)
    assert remaining["steps"] == 0
    assert remaining["cost_usd"] == 0.0


def test_stop_reasons_classify_themselves():
    assert StopReason.COMPLETED.is_success
    assert not StopReason.NO_PROGRESS.is_success
    assert StopReason.DEADLINE.is_budget
    assert not StopReason.REFUSED.is_budget


def test_contract_cannot_read_the_transcript():
    """The contract judges a run on what it consumed, never on what it said."""
    assert not hasattr(RunState(), "messages")
