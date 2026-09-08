"""Dispatcher behaviour.

The governing rule is that a tool failure is data, never an exception. These
tests assert that every way a call can go wrong - unknown tool, denied
capability, malformed JSON, schema violation, a handler that raises, one that
hangs, one that returns the wrong type - comes back as a result the model can
read, in the right order, with the run still alive.
"""

from __future__ import annotations

import threading
import time

import pytest

from conftest import make_spec
from gantry.config import BudgetConfig
from gantry.contract import LoopContract, StopReason
from gantry.dispatch import Dispatcher, ToolCall
from gantry.errors import ToolExecutionError
from gantry.tools import Capability, Grant, ToolContext, ToolRegistry, ToolResult


@pytest.fixture
def ctx() -> ToolContext:
    return ToolContext(run_id="r1")


def build(registry: ToolRegistry, grant: Grant | None = None, budget=None, **kw) -> Dispatcher:
    return Dispatcher(
        registry=registry,
        grant=grant or Grant.unrestricted(),
        contract=LoopContract(budget or BudgetConfig()),
        sleep=lambda _s: None,  # retries must not make the suite slow
        **kw,
    )


# --- happy path ------------------------------------------------------------
def test_a_successful_call_returns_a_provider_message(registry, ctx, tracer):
    registry.register(make_spec("echo"))
    dispatcher = build(registry, tracer=tracer)
    state = dispatcher.contract.start_run()
    outcome = dispatcher.dispatch([ToolCall("c1", "echo", '{"text":"hi"}')], state, ctx)
    assert outcome.results[0].ok
    assert outcome.provider_messages() == [{"role": "tool", "tool_call_id": "c1", "content": "hi"}]
    assert state.usage.tool_calls == 1


def test_results_come_back_in_call_order(registry, ctx):
    registry.register(
        make_spec("slow", handler=lambda a, c: (time.sleep(0.02), ToolResult.success(a["text"]))[1])
    )
    dispatcher = build(registry)
    state = dispatcher.contract.start_run()
    calls = [ToolCall(f"c{i}", "slow", f'{{"text":"{i}"}}') for i in range(4)]
    outcome = dispatcher.dispatch(calls, state, ctx)
    # Providers reject a turn whose tool messages are out of order.
    assert [r.call.id for r in outcome.results] == ["c0", "c1", "c2", "c3"]
    assert [r.result.content for r in outcome.results] == ["0", "1", "2", "3"]


def test_dict_arguments_are_accepted_as_well_as_json_strings(registry, ctx):
    registry.register(make_spec("echo"))
    dispatcher = build(registry)
    state = dispatcher.contract.start_run()
    outcome = dispatcher.dispatch([ToolCall("c1", "echo", {"text": "hi"})], state, ctx)
    assert outcome.results[0].result.content == "hi"


# --- failures are data -----------------------------------------------------
def test_an_unknown_tool_is_a_result_not_an_exception(registry, ctx):
    dispatcher = build(registry)
    state = dispatcher.contract.start_run()
    result = dispatcher.dispatch([ToolCall("c1", "nope", "{}")], state, ctx).results[0]
    assert not result.ok
    assert result.result.error_code == "tool.not_found"


def test_a_denied_capability_is_a_result(registry, ctx):
    registry.register(make_spec("write_file", capabilities=frozenset({Capability.FS_WRITE})))
    dispatcher = build(registry, grant=Grant.read_only())
    state = dispatcher.contract.start_run()
    result = dispatcher.dispatch([ToolCall("c1", "write_file", "{}")], state, ctx).results[0]
    assert result.result.error_code == "tool.capability_denied"


def test_malformed_json_arguments_are_explained_to_the_model(registry, ctx):
    registry.register(make_spec("echo"))
    dispatcher = build(registry)
    state = dispatcher.contract.start_run()
    result = dispatcher.dispatch([ToolCall("c1", "echo", '{"text": ')], state, ctx).results[0]
    assert result.result.error_code == "tool.invalid_arguments"
    assert "not valid JSON" in result.result.content


def test_schema_violations_are_explained_to_the_model(registry, ctx):
    registry.register(make_spec("echo"))
    dispatcher = build(registry)
    state = dispatcher.contract.start_run()
    result = dispatcher.dispatch([ToolCall("c1", "echo", '{"wrong":"x"}')], state, ctx).results[0]
    assert result.result.error_code == "tool.invalid_arguments"
    assert "text" in result.result.content


def test_a_raising_handler_becomes_a_failed_result(registry, ctx):
    def explode(args, c):
        raise RuntimeError("kaboom")

    registry.register(make_spec("bad", handler=explode))
    dispatcher = build(registry)
    state = dispatcher.contract.start_run()
    result = dispatcher.dispatch([ToolCall("c1", "bad", '{"text":"x"}')], state, ctx).results[0]
    assert not result.ok
    assert "kaboom" in result.result.content


def test_a_harness_error_keeps_its_code(registry, ctx):
    def explode(args, c):
        raise ToolExecutionError("git is not installed", binary="git")

    registry.register(make_spec("bad", handler=explode))
    dispatcher = build(registry)
    state = dispatcher.contract.start_run()
    result = dispatcher.dispatch([ToolCall("c1", "bad", '{"text":"x"}')], state, ctx).results[0]
    assert result.result.error_code == "tool.execution_failed"
    assert result.result.data["binary"] == "git"


def test_a_handler_returning_the_wrong_type_is_caught(registry, ctx):
    registry.register(make_spec("bad", handler=lambda a, c: "a bare string"))
    dispatcher = build(registry)
    state = dispatcher.contract.start_run()
    result = dispatcher.dispatch([ToolCall("c1", "bad", '{"text":"x"}')], state, ctx).results[0]
    assert result.result.error_code == "tool.contract_violation"


def test_a_hanging_handler_hits_its_timeout(registry, ctx):
    registry.register(
        make_spec("hang", handler=lambda a, c: threading.Event().wait(5), timeout_s=0.05)
    )
    dispatcher = build(registry)
    state = dispatcher.contract.start_run()
    started = time.monotonic()
    result = dispatcher.dispatch([ToolCall("c1", "hang", '{"text":"x"}')], state, ctx).results[0]
    # Bounds how long the agent waits, not how long the work takes: a Python
    # thread cannot be killed, which is why the result records `abandoned`.
    assert time.monotonic() - started < 2
    assert result.result.error_code == "tool.timeout"
    assert result.result.data["abandoned"] is True


def test_output_is_truncated_to_the_tool_budget(registry, ctx):
    registry.register(
        make_spec("big", handler=lambda a, c: ToolResult.success("x" * 5000), max_output_chars=100)
    )
    dispatcher = build(registry)
    state = dispatcher.contract.start_run()
    result = dispatcher.dispatch([ToolCall("c1", "big", '{"text":"x"}')], state, ctx).results[0]
    assert result.result.truncated
    assert len(result.result.content) < 200


# --- concurrency -----------------------------------------------------------
def test_concurrency_safe_calls_run_in_parallel(registry, ctx):
    barrier = threading.Barrier(3, timeout=3)

    def wait_for_others(args, c):
        barrier.wait()  # only returns if three calls are in flight at once
        return ToolResult.success("ok")

    registry.register(make_spec("par", handler=wait_for_others, concurrency_safe=True))
    dispatcher = build(registry, max_workers=3)
    state = dispatcher.contract.start_run()
    calls = [ToolCall(f"c{i}", "par", f'{{"text":"{i}"}}') for i in range(3)]
    outcome = dispatcher.dispatch(calls, state, ctx)
    assert all(r.ok for r in outcome.results)


def test_a_non_concurrency_safe_tool_acts_as_a_barrier(registry, ctx):
    """Two writes to the same file must not race just because the model emitted
    them in one turn."""
    active, peak = [], []
    lock = threading.Lock()

    def track(args, c):
        with lock:
            active.append(1)
            peak.append(len(active))
        time.sleep(0.01)
        with lock:
            active.pop()
        return ToolResult.success(args["text"])

    registry.register(make_spec("write", handler=track, concurrency_safe=False))
    dispatcher = build(registry, max_workers=4)
    state = dispatcher.contract.start_run()
    calls = [ToolCall(f"c{i}", "write", f'{{"text":"{i}"}}') for i in range(4)]
    outcome = dispatcher.dispatch(calls, state, ctx)
    assert max(peak) == 1
    assert [r.result.content for r in outcome.results] == ["0", "1", "2", "3"]


def test_parallel_tool_spans_stay_attached_to_their_run(registry, ctx, tracer, exporter):
    """Worker threads start with an empty contextvars context, so the spans
    would otherwise be orphaned from the run that made them."""
    registry.register(make_spec("par", concurrency_safe=True))
    dispatcher = build(registry, max_workers=4, tracer=tracer)
    state = dispatcher.contract.start_run()
    with tracer.trace("agent.run"):
        dispatcher.dispatch(
            [ToolCall(f"c{i}", "par", f'{{"text":"{i}"}}') for i in range(4)], state, ctx
        )
    trace = exporter.traces[0]
    tool_spans = [s for s in trace.spans if s.kind == "tool"]
    assert len(tool_spans) == 4
    assert all(s.parent_span_id is not None for s in tool_spans)


# --- retries ---------------------------------------------------------------
def test_a_retryable_failure_on_an_idempotent_tool_is_retried(registry, ctx):
    attempts = []

    def flaky(args, c):
        attempts.append(1)
        if len(attempts) < 3:
            return ToolResult.failure("temporarily unavailable", retryable=True)
        return ToolResult.success("ok")

    registry.register(make_spec("flaky", handler=flaky, idempotent=True))
    dispatcher = build(registry, max_retries=3)
    state = dispatcher.contract.start_run()
    result = dispatcher.dispatch([ToolCall("c1", "flaky", '{"text":"x"}')], state, ctx).results[0]
    assert result.ok
    assert result.attempts == 3
    # The whole retried call counts once against the budget, not three times.
    assert state.usage.tool_calls == 1


def test_a_non_idempotent_tool_is_never_retried(registry, ctx):
    """A timeout means the answer was lost, not that the work was."""
    attempts = []

    def flaky(args, c):
        attempts.append(1)
        return ToolResult.failure("temporarily unavailable", retryable=True)

    registry.register(make_spec("write", handler=flaky, idempotent=False))
    dispatcher = build(registry, max_retries=3)
    state = dispatcher.contract.start_run()
    dispatcher.dispatch([ToolCall("c1", "write", '{"text":"x"}')], state, ctx)
    assert len(attempts) == 1


def test_a_non_retryable_failure_is_not_retried(registry, ctx):
    attempts = []
    registry.register(
        make_spec(
            "bad",
            handler=lambda a, c: (attempts.append(1), ToolResult.failure("no such file"))[1],
        )
    )
    dispatcher = build(registry, max_retries=3)
    state = dispatcher.contract.start_run()
    dispatcher.dispatch([ToolCall("c1", "bad", '{"text":"x"}')], state, ctx)
    assert len(attempts) == 1


# --- contract integration --------------------------------------------------
def test_the_contract_stops_a_turn_and_the_rest_still_get_messages(registry, ctx):
    """A provider rejects a turn where any tool-call id went unanswered, so a
    clean budget stop must not become a protocol error."""
    registry.register(make_spec("write", concurrency_safe=False))
    dispatcher = build(registry, budget=BudgetConfig(max_tool_calls=2))
    state = dispatcher.contract.start_run()
    calls = [ToolCall(f"c{i}", "write", f'{{"text":"{i}"}}') for i in range(4)]
    outcome = dispatcher.dispatch(calls, state, ctx)
    assert outcome.should_stop
    assert outcome.stop.stop_reason is StopReason.MAX_TOOL_CALLS
    assert len(outcome.provider_messages()) == 4
    assert [r.executed for r in outcome.results] == [True, True, False, False]


def test_repeating_one_action_stops_the_run(registry, ctx):
    registry.register(make_spec("echo", concurrency_safe=False))
    dispatcher = build(registry, budget=BudgetConfig(max_repeat_actions=2))
    state = dispatcher.contract.start_run()
    calls = [ToolCall(f"c{i}", "echo", '{"text":"same"}') for i in range(4)]
    outcome = dispatcher.dispatch(calls, state, ctx)
    assert outcome.stop.stop_reason is StopReason.NO_PROGRESS


def test_rejected_calls_still_count_toward_the_error_budget(registry, ctx):
    dispatcher = build(registry)
    state = dispatcher.contract.start_run()
    dispatcher.dispatch([ToolCall("c1", "nope", "{}")], state, ctx)
    assert state.usage.consecutive_tool_errors == 1


def test_an_empty_turn_is_a_no_op(registry, ctx):
    dispatcher = build(registry)
    outcome = dispatcher.dispatch([], dispatcher.contract.start_run(), ctx)
    assert outcome.results == [] and not outcome.should_stop
