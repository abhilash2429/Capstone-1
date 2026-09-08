"""The loop contract: the only thing allowed to stop an agent run.

Most agent bugs are not reasoning bugs. They are loop bugs - a run that spends
forty dollars retrying the same failing command, one that declares success
without checking, one that quietly runs until the process is killed. Those are
control-flow problems, and control-flow problems are fixable in ordinary code.

So termination is pulled out of the agent entirely and made a pure, testable
object. :class:`LoopContract` observes what a run consumes and answers one
question - *may this run continue?* - with a typed :class:`StopReason` when the
answer is no. It has no dependency on a model, a provider or a network, which
means every termination path in this project is covered by a unit test that
runs in microseconds.

The invariants it enforces:

* A run terminates. Every path out of the loop is a named ``StopReason``; there
  is no implicit fall-through and no unbounded ``while True``.
* A run terminates *before* exceeding its budget, not after. Costs are checked
  against the projected next step, so the ceiling is a ceiling.
* Repeating an identical action is progress-free by definition, and a run that
  makes no progress is stopped rather than left to burn its budget.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from gantry.config import BudgetConfig


class Phase(StrEnum):
    """Where a run is in the plan-act-observe-verify cycle."""

    PLAN = "plan"
    ACT = "act"
    OBSERVE = "observe"
    VERIFY = "verify"
    DONE = "done"


class StopReason(StrEnum):
    """Every way a run can end. Exhaustive by design."""

    COMPLETED = "completed"
    """The agent declared completion and verification agreed."""

    MAX_STEPS = "max_steps"
    MAX_TOOL_CALLS = "max_tool_calls"
    TOKEN_BUDGET = "token_budget"
    COST_BUDGET = "cost_budget"
    DEADLINE = "deadline"
    TOOL_ERROR_BUDGET = "tool_error_budget"
    NO_PROGRESS = "no_progress"
    VERIFICATION_FAILED = "verification_failed"
    REFUSED = "refused"
    """The agent declined the task. A legitimate outcome, not a failure."""

    PROVIDER_ERROR = "provider_error"
    CANCELLED = "cancelled"
    INTERNAL_ERROR = "internal_error"

    @property
    def is_success(self) -> bool:
        return self is StopReason.COMPLETED

    @property
    def is_budget(self) -> bool:
        return self in {
            StopReason.MAX_STEPS,
            StopReason.MAX_TOOL_CALLS,
            StopReason.TOKEN_BUDGET,
            StopReason.COST_BUDGET,
            StopReason.DEADLINE,
        }


def action_fingerprint(tool: str, arguments: Any) -> str:
    """A stable hash of an intended action.

    Arguments are canonicalised (sorted keys, no insignificant whitespace) so
    that two calls differing only in key order count as the same action - which
    they are. Twelve hex characters is ample for in-run comparison and keeps
    traces readable.
    """
    payload = json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(f"{tool}\x00{payload}".encode()).hexdigest()[:12]


@dataclass
class Usage:
    """Running totals for one agent run."""

    steps: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    tool_errors: int = 0
    consecutive_tool_errors: int = 0
    provider_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "tool_calls": self.tool_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "tool_errors": self.tool_errors,
            "provider_calls": self.provider_calls,
        }


@dataclass
class RunState:
    """Everything the contract needs to judge a run, and nothing else.

    Kept separate from the agent's conversation state on purpose: the contract
    should not be able to read the transcript, so it cannot be talked out of
    stopping.
    """

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    task: str = ""
    phase: Phase = Phase.PLAN
    usage: Usage = field(default_factory=Usage)
    #: Monotonic timestamp from the owning contract's clock. Use
    #: :meth:`LoopContract.start_run` rather than setting this by hand.
    started_at: float = field(default_factory=time.monotonic)
    stop_reason: StopReason | None = None
    stop_detail: str = ""
    action_counts: Counter[str] = field(default_factory=Counter)
    trace_id: str | None = None

    def elapsed_s(self, now: float | None = None) -> float:
        return (now if now is not None else time.monotonic()) - self.started_at


class Cancellation:
    """A cancellation token the transport, CLI or a timeout can trip."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._reason = ""

    def cancel(self, reason: str = "cancelled by caller") -> None:
        self._reason = reason
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        return self._reason

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)


@dataclass(frozen=True)
class Decision:
    """The contract's verdict at a checkpoint."""

    allowed: bool
    stop_reason: StopReason | None = None
    detail: str = ""

    @classmethod
    def go(cls) -> Decision:
        return cls(allowed=True)

    @classmethod
    def stop(cls, reason: StopReason, detail: str = "") -> Decision:
        return cls(allowed=False, stop_reason=reason, detail=detail)


class LoopContract:
    """Enforces the budget and progress invariants of a single run.

    The clock is injectable so that deadline behaviour is testable without
    sleeping, which is the difference between a timeout path that is covered and
    one that is merely hoped for.
    """

    def __init__(
        self,
        budget: BudgetConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
        cancellation: Cancellation | None = None,
    ) -> None:
        self.budget = budget or BudgetConfig()
        self.clock = clock
        self.cancellation = cancellation or Cancellation()

    # -- run lifecycle --------------------------------------------------
    def start_run(self, task: str = "", run_id: str | None = None) -> RunState:
        """Create a run anchored to *this contract's* clock.

        Always construct runs through here. A :class:`RunState` built directly
        anchors ``started_at`` to the wall clock, and if the contract is using
        an injected clock the two disagree and the deadline silently never
        fires - a bug this project found by testing the timeout path rather
        than assuming it.
        """
        state = RunState(task=task, started_at=self.clock())
        if run_id:
            state.run_id = run_id
        return state

    # -- checkpoints ----------------------------------------------------
    def before_step(self, state: RunState) -> Decision:
        """Called once before each planning turn."""
        if self.cancellation.cancelled:
            return Decision.stop(StopReason.CANCELLED, self.cancellation.reason)

        if state.usage.steps >= self.budget.max_steps:
            return Decision.stop(
                StopReason.MAX_STEPS,
                f"reached step limit of {self.budget.max_steps}",
            )
        if state.usage.tool_calls >= self.budget.max_tool_calls:
            return Decision.stop(
                StopReason.MAX_TOOL_CALLS,
                f"reached tool-call limit of {self.budget.max_tool_calls}",
            )
        if state.usage.total_tokens >= self.budget.max_tokens:
            return Decision.stop(
                StopReason.TOKEN_BUDGET,
                f"consumed {state.usage.total_tokens} of {self.budget.max_tokens} tokens",
            )
        if state.usage.cost_usd >= self.budget.max_cost_usd:
            return Decision.stop(
                StopReason.COST_BUDGET,
                f"spent ${state.usage.cost_usd:.4f} of ${self.budget.max_cost_usd:.2f}",
            )
        elapsed = state.elapsed_s(self.clock())
        if elapsed >= self.budget.wall_clock_s:
            return Decision.stop(
                StopReason.DEADLINE,
                f"ran for {elapsed:.1f}s of {self.budget.wall_clock_s:.0f}s allowed",
            )
        if state.usage.consecutive_tool_errors >= self.budget.max_consecutive_tool_errors:
            return Decision.stop(
                StopReason.TOOL_ERROR_BUDGET,
                f"{state.usage.consecutive_tool_errors} consecutive tool failures",
            )
        return Decision.go()

    def before_action(self, state: RunState, tool: str, arguments: Any) -> Decision:
        """Called before dispatching a tool call.

        This is where no-progress is caught. An agent that has issued the same
        call with the same arguments ``max_repeat_actions`` times is not going
        to succeed on the next attempt, and letting it try is how a run burns
        its whole budget on one broken command.
        """
        if self.cancellation.cancelled:
            return Decision.stop(StopReason.CANCELLED, self.cancellation.reason)
        if state.usage.tool_calls >= self.budget.max_tool_calls:
            return Decision.stop(
                StopReason.MAX_TOOL_CALLS,
                f"reached tool-call limit of {self.budget.max_tool_calls}",
            )

        fingerprint = action_fingerprint(tool, arguments)
        if state.action_counts[fingerprint] >= self.budget.max_repeat_actions:
            return Decision.stop(
                StopReason.NO_PROGRESS,
                f"action {tool!r} repeated {state.action_counts[fingerprint]} times "
                "with identical arguments",
            )
        return Decision.go()

    def can_afford(self, state: RunState, projected_cost_usd: float) -> Decision:
        """Check a *projected* spend before committing to it.

        Checking after the fact turns a budget into a suggestion. A caller that
        can estimate the cost of the next provider call asks here first.
        """
        if state.usage.cost_usd + projected_cost_usd > self.budget.max_cost_usd:
            return Decision.stop(
                StopReason.COST_BUDGET,
                f"next call would reach ${state.usage.cost_usd + projected_cost_usd:.4f}, "
                f"over the ${self.budget.max_cost_usd:.2f} ceiling",
            )
        return Decision.go()

    # -- accounting -----------------------------------------------------
    def record_step(self, state: RunState) -> None:
        state.usage.steps += 1

    def record_provider_call(
        self,
        state: RunState,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        state.usage.provider_calls += 1
        state.usage.input_tokens += input_tokens
        state.usage.output_tokens += output_tokens
        state.usage.cost_usd += cost_usd

    def record_action(self, state: RunState, tool: str, arguments: Any, ok: bool) -> str:
        """Record a completed tool call. Returns the action fingerprint."""
        fingerprint = action_fingerprint(tool, arguments)
        state.action_counts[fingerprint] += 1
        state.usage.tool_calls += 1
        if ok:
            state.usage.consecutive_tool_errors = 0
        else:
            state.usage.tool_errors += 1
            state.usage.consecutive_tool_errors += 1
        return fingerprint

    def finish(self, state: RunState, reason: StopReason, detail: str = "") -> RunState:
        state.stop_reason = reason
        state.stop_detail = detail
        state.phase = Phase.DONE
        return state

    # -- introspection --------------------------------------------------
    def remaining(self, state: RunState) -> dict[str, Any]:
        """What is left of each budget dimension. Surfaced in the dashboard."""
        return {
            "steps": max(0, self.budget.max_steps - state.usage.steps),
            "tool_calls": max(0, self.budget.max_tool_calls - state.usage.tool_calls),
            "tokens": max(0, self.budget.max_tokens - state.usage.total_tokens),
            "cost_usd": round(max(0.0, self.budget.max_cost_usd - state.usage.cost_usd), 6),
            "seconds": round(max(0.0, self.budget.wall_clock_s - state.elapsed_s(self.clock())), 2),
        }
