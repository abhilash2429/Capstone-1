"""The tool-call dispatcher.

One rule governs this module: **a tool failure is data, not an exception**. A
model that calls a tool with a bad path, a missing argument or a command that
exits non-zero has not broken the harness, it has learned something, and the
loop's job is to hand that back as a result it can act on. An exception escaping
here would kill a run that was one turn from succeeding.

Three behaviours are worth calling out because they are easy to get wrong:

* **Ordering.** Results come back in call order, always. Providers require one
  tool message per tool-call id, in sequence, and a reordered reply is a
  protocol error rather than a cosmetic issue.
* **Parallelism is opt-in per tool.** Calls run concurrently only while every
  call in the group declares itself concurrency-safe. A tool that is not acts
  as a barrier, so two writes to the same file never race just because the
  model emitted them in one turn.
* **Retries are opt-in per tool.** Only idempotent tools are retried, and only
  on errors marked retryable. Retrying a write after a timeout can apply it
  twice, because a timeout means the *answer* was lost, not the work.
"""

from __future__ import annotations

import contextvars
import random
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from gantry.contract import Decision, LoopContract, RunState, StopReason
from gantry.errors import (
    CapabilityDenied,
    GantryError,
    ToolNotFound,
    ToolTimeout,
    ToolValidationError,
)
from gantry.messages import ToolCall
from gantry.telemetry import metrics
from gantry.telemetry import semconv as sc
from gantry.telemetry.tracer import Tracer, get_tracer
from gantry.tools import Grant, ToolContext, ToolRegistry, ToolResult, ToolSpec


@dataclass
class DispatchResult:
    """What happened to one tool call."""

    call: ToolCall
    result: ToolResult
    spec: ToolSpec | None = None
    fingerprint: str = ""
    attempts: int = 1
    duration_ms: float = 0.0
    executed: bool = True

    @property
    def ok(self) -> bool:
        return self.result.ok

    def to_provider_message(self) -> dict[str, Any]:
        """The OpenAI tool-result message for this call."""
        return {"role": "tool", "tool_call_id": self.call.id, "content": self.result.content}


@dataclass
class DispatchOutcome:
    """The results of one turn's tool calls, plus any decision to stop."""

    results: list[DispatchResult] = field(default_factory=list)
    stop: Decision | None = None

    @property
    def should_stop(self) -> bool:
        return self.stop is not None and not self.stop.allowed

    def provider_messages(self) -> list[dict[str, Any]]:
        """One message per call, in call order.

        Calls the contract cut short still get a message. A provider rejects a
        turn where any tool-call id went unanswered, so skipping them would
        turn a clean budget stop into a protocol error.
        """
        return [r.to_provider_message() for r in self.results]


def _not_executed(call: ToolCall, reason: str) -> DispatchResult:
    return DispatchResult(
        call=call,
        result=ToolResult.failure(f"Not executed: {reason}", error_code="loop.not_executed"),
        executed=False,
    )


class Dispatcher:
    """Routes model tool calls to handlers, under the loop contract."""

    def __init__(
        self,
        registry: ToolRegistry,
        grant: Grant,
        contract: LoopContract,
        tracer: Tracer | None = None,
        max_workers: int = 4,
        max_retries: int = 2,
        retry_base_delay_s: float = 0.2,
        sleep=time.sleep,
    ) -> None:
        self.registry = registry
        self.grant = grant
        self.contract = contract
        self.tracer = tracer or get_tracer()
        self.max_workers = max(1, max_workers)
        self.max_retries = max(0, max_retries)
        self.retry_base_delay_s = retry_base_delay_s
        self._sleep = sleep

    # -- public API ------------------------------------------------------
    def dispatch(self, calls: list[ToolCall], state: RunState, ctx: ToolContext) -> DispatchOutcome:
        """Execute a turn's tool calls and return one result per call."""
        outcome = DispatchOutcome()
        if not calls:
            return outcome

        pending = list(calls)
        while pending:
            group, pending = self._next_group(pending)
            if outcome.should_stop:
                outcome.results.extend(
                    _not_executed(c, outcome.stop.detail or "run stopped") for c in group
                )
                continue
            outcome.results.extend(self._run_group(group, state, ctx, outcome))

        # Order must match the provider's tool-call order exactly.
        by_id = {r.call.id: r for r in outcome.results}
        outcome.results = [by_id[c.id] for c in calls]
        return outcome

    # -- grouping --------------------------------------------------------
    def _next_group(self, calls: list[ToolCall]) -> tuple[list[ToolCall], list[ToolCall]]:
        """Take the longest leading run of calls that may execute together.

        A call whose tool is unknown, denied or not concurrency-safe ends the
        group. Unknown tools are kept serial deliberately: resolving them
        happens inside execution, and a group is only safe if every member of
        it is known to be safe.
        """
        group: list[ToolCall] = []
        for index, call in enumerate(calls):
            if not self._is_concurrency_safe(call):
                # A barrier call runs alone; if a group is already forming, it
                # waits for the next round.
                return (group or [call]), calls[index + 1 :] if not group else calls[index:]
            group.append(call)
            if len(group) >= self.max_workers:
                return group, calls[index + 1 :]
        return group, []

    def _is_concurrency_safe(self, call: ToolCall) -> bool:
        try:
            spec = self.registry.get(call.name)
        except ToolNotFound:
            return False
        return spec.concurrency_safe

    # -- execution -------------------------------------------------------
    def _run_group(
        self,
        group: list[ToolCall],
        state: RunState,
        ctx: ToolContext,
        outcome: DispatchOutcome,
    ) -> list[DispatchResult]:
        if len(group) == 1:
            return [self._execute(group[0], state, ctx, outcome)]

        # copy_context() per task, so each worker thread inherits the current
        # trace and span. Threads otherwise start with an empty context and the
        # parallel tool spans would be orphaned from the run that made them.
        futures: list[Future[DispatchResult]] = []
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(group))) as pool:
            for call in group:
                context = contextvars.copy_context()
                futures.append(pool.submit(context.run, self._execute, call, state, ctx, outcome))
            return [f.result() for f in futures]

    def _execute(
        self,
        call: ToolCall,
        state: RunState,
        ctx: ToolContext,
        outcome: DispatchOutcome,
    ) -> DispatchResult:
        started = time.monotonic()
        with self.tracer.tool_span(call.name, **{sc.GEN_AI_TOOL_CALL_ID: call.id}) as span:
            try:
                spec = self.registry.authorize(call.name, self.grant)
                arguments = call.parse_arguments()
                self.registry.validate_arguments(call.name, arguments)
            except (ToolNotFound, CapabilityDenied, ToolValidationError) as exc:
                # Recoverable by the model: it is told exactly what was wrong so
                # the next turn can correct itself.
                span.set_status("error", exc.code)
                span.set_attribute(sc.TOOL_OUTCOME, "rejected")
                metrics.observe_tool_call(call.name, time.monotonic() - started, ok=False)
                self.contract.record_action(state, call.name, call.arguments, ok=False)
                return DispatchResult(
                    call=call,
                    result=ToolResult.failure(exc.message, error_code=exc.code),
                    duration_ms=(time.monotonic() - started) * 1000.0,
                )

            decision = self.contract.before_action(state, call.name, arguments)
            if not decision.allowed:
                outcome.stop = decision
                span.set_attribute(sc.TOOL_OUTCOME, "blocked")
                span.set_attribute(sc.RUN_STOP_REASON, str(decision.stop_reason))
                return _not_executed(call, decision.detail or str(decision.stop_reason))

            span.set_attributes(
                {
                    sc.TOOL_VERSION: spec.version,
                    sc.TOOL_CAPABILITIES: sorted(spec.capabilities),
                    sc.GEN_AI_TOOL_DESCRIPTION: spec.description[:200],
                }
            )
            result, attempts = self._invoke_with_retries(spec, arguments, ctx, span)
            result.truncate(spec.max_output_chars)

            duration_s = time.monotonic() - started
            fingerprint = self.contract.record_action(state, call.name, arguments, ok=result.ok)
            span.set_attributes(
                {
                    sc.TOOL_OUTCOME: "ok" if result.ok else "error",
                    sc.TOOL_TRUNCATED: result.truncated,
                    sc.TOOL_ACTION_FINGERPRINT: fingerprint,
                    sc.RETRY_COUNT: attempts - 1,
                }
            )
            if not result.ok:
                span.set_status("error", result.error_code or "tool.execution_failed")
            metrics.observe_tool_call(call.name, duration_s, ok=result.ok)
            return DispatchResult(
                call=call,
                result=result,
                spec=spec,
                fingerprint=fingerprint,
                attempts=attempts,
                duration_ms=duration_s * 1000.0,
            )

    def _invoke_with_retries(
        self, spec: ToolSpec, arguments: dict[str, Any], ctx: ToolContext, span: Any
    ) -> tuple[ToolResult, int]:
        attempts = 0
        last: ToolResult | None = None
        while attempts <= self.max_retries:
            attempts += 1
            result = self._invoke_once(spec, arguments, ctx)
            if result.ok:
                return result, attempts
            last = result
            retryable = result.data.get("retryable", False)
            if not (retryable and spec.is_retryable) or attempts > self.max_retries:
                break
            # Full jitter: a fixed backoff synchronises retries across parallel
            # calls into a thundering herd against the same failing resource.
            delay = random.uniform(0, self.retry_base_delay_s * (2 ** (attempts - 1)))  # noqa: S311
            span.add_event(
                "retry", attempt=attempts, delay_s=round(delay, 3), reason=result.error_code
            )
            self._sleep(delay)
        return last or ToolResult.failure("tool produced no result"), attempts

    def _invoke_once(
        self, spec: ToolSpec, arguments: dict[str, Any], ctx: ToolContext
    ) -> ToolResult:
        """Run one handler under its timeout, converting any failure to a result.

        A daemon thread rather than a pooled worker, deliberately.
        ``ThreadPoolExecutor`` shuts down with ``wait=True`` when it leaves a
        ``with`` block, so a pooled call would block on the very handler the
        timeout exists to escape - the timeout would look correct in review and
        do nothing at runtime. A daemon thread can simply be abandoned, and
        cannot hold up interpreter exit.

        What this bounds is how long the *agent* waits, not how long the work
        takes: Python cannot kill a thread, so the abandoned handler runs on.
        A hard kill needs process isolation, which is what the sandbox provides
        for the tools that can genuinely hang.
        """
        started = time.monotonic()
        outcome: dict[str, Any] = {}
        context = contextvars.copy_context()

        def run() -> None:
            try:
                outcome["result"] = context.run(spec.handler, arguments, ctx)
            except BaseException as exc:  # noqa: BLE001 - reported below as a result
                outcome["error"] = exc

        worker = threading.Thread(target=run, name=f"gantry-tool-{spec.name}", daemon=True)
        worker.start()
        worker.join(timeout=spec.timeout_s)

        if worker.is_alive():
            return ToolResult.failure(
                f"Tool {spec.name!r} exceeded its {spec.timeout_s:g}s timeout.",
                error_code=ToolTimeout.code,
                retryable=spec.is_retryable,
                abandoned=True,
            )

        if "error" in outcome:
            exc = outcome["error"]
            if isinstance(exc, GantryError):
                return ToolResult.failure(
                    exc.message, error_code=exc.code, retryable=exc.retryable, **exc.details
                )
            return ToolResult.failure(
                f"{type(exc).__name__}: {exc}", error_code="tool.execution_failed"
            )

        result = outcome.get("result")
        if not isinstance(result, ToolResult):
            return ToolResult.failure(
                f"Tool {spec.name!r} returned {type(result).__name__}, expected ToolResult",
                error_code="tool.contract_violation",
            )
        result.duration_ms = (time.monotonic() - started) * 1000.0
        return result


def stop_reason_from(outcome: DispatchOutcome) -> StopReason | None:
    return outcome.stop.stop_reason if outcome.should_stop else None


#: Re-exported so callers can import the call and its dispatcher together.
__all__ = ["DispatchOutcome", "DispatchResult", "Dispatcher", "ToolCall", "stop_reason_from"]
