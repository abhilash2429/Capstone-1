"""The observation budget: keeping a long loop from drowning in its own output.

An agent's transcript grows monotonically. Every tool result is appended and
resent on every subsequent turn, so a run that reads six files has paid for the
first one six times. Left alone this ends one of two ways: a context-length
error partway through a task, or a bill dominated by re-reading stale output.

The fix is to shrink old observations rather than drop turns. Three rules
decide what survives:

* **The task never goes.** The system prompt and the original instruction stay
  whole. An agent that forgets what it was asked will confidently finish the
  wrong job.
* **Recent turns never shrink.** The last few exchanges are what the next
  decision is actually based on.
* **Old tool output shrinks to a receipt.** The body is replaced with a note
  saying what ran, how much was elided and where the full text lives in the
  trace, so nothing is lost - it is just no longer being paid for every turn.

The invariant that makes this subtle: the message list has to stay valid for
the provider. Every assistant tool call must still be answered by a tool
message with a matching id. Dropping a message to save tokens produces a 400,
so elision only ever rewrites content, never removes a turn.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from gantry.messages import Message, Role

#: Characters per token. Wrong in detail, right in magnitude, and identical on
#: every machine - which matters more here than accuracy, because a budget that
#: depends on a tokenizer download is a budget that fails in CI.
CHARS_PER_TOKEN = 4

#: Rough per-message envelope the provider adds (role, delimiters, ids).
MESSAGE_OVERHEAD_TOKENS = 4

#: Nothing is shrunk below this: a message trimmed to nothing is worse than
#: absent, because it still costs its envelope and carries no information.
MIN_SALVAGE_TOKENS = 12


def estimate_tokens(text: str) -> int:
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def message_tokens(message: Message) -> int:
    total = estimate_tokens(message.content or "") + MESSAGE_OVERHEAD_TOKENS
    for call in message.tool_calls:
        total += estimate_tokens(call.name) + estimate_tokens(call.arguments_json())
    return total


def conversation_tokens(messages: Sequence[Message]) -> int:
    return sum(message_tokens(m) for m in messages)


@dataclass
class Elision:
    """What the budget had to give up, and how much it saved."""

    elided_messages: int = 0
    tokens_saved: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    truncated_messages: int = 0
    #: False when even the last-resort pass could not get under the limit. The
    #: caller has to know: sending the request anyway earns a context-length
    #: 400, and reporting success while returning something oversized is how
    #: that becomes a mystery.
    fits: bool = True

    @property
    def changed(self) -> bool:
        return bool(self.elided_messages or self.truncated_messages)

    def as_attributes(self) -> dict[str, object]:
        return {
            "gantry.budget.fits": self.fits,
            "gantry.budget.tokens_before": self.tokens_before,
            "gantry.budget.tokens_after": self.tokens_after,
            "gantry.budget.elided_messages": self.elided_messages,
            "gantry.budget.truncated_messages": self.truncated_messages,
            "gantry.budget.tokens_saved": self.tokens_saved,
        }


@dataclass
class ObservationBudget:
    """Fits a growing transcript into a fixed context window."""

    #: The model's context window, in tokens.
    max_context_tokens: int = 96_000
    #: Held back for the reply, so a full prompt still leaves room to answer.
    reserve_output_tokens: int = 8_000
    #: Ceiling on a single fresh observation. A 400KB test log is not more
    #: informative than its first few hundred lines, it is only more expensive.
    max_observation_tokens: int = 4_000
    #: Turns at the end of the transcript that are never shrunk.
    keep_recent_turns: int = 6
    #: Messages at the start that are never shrunk: the system prompt and the
    #: original task.
    keep_leading_turns: int = 2
    #: Ceiling applied to the agent's own turns in the salvage pass, once every
    #: observation is already a receipt. Small on purpose.
    salvage_tokens: int = 128

    @property
    def available_tokens(self) -> int:
        return max(1, self.max_context_tokens - self.reserve_output_tokens)

    # -- single observations ---------------------------------------------
    def cap_observation(self, content: str, max_tokens: int | None = None) -> tuple[str, bool]:
        """Trim one message to a token ceiling.

        Keeps the head and the tail. The head says what ran and the tail
        carries the failure, and a naive head-only truncation reliably discards
        the stack trace that made the output worth reading.

        ``max_tokens`` overrides the per-observation ceiling. The salvage pass
        below needs a much tighter one: reusing the generous default there
        means nothing is ever short enough to trim, and the pass silently does
        nothing at all.
        """
        limit = max(1, (max_tokens or self.max_observation_tokens)) * CHARS_PER_TOKEN
        if len(content) <= limit:
            return content, False
        marker = f"[... {len(content) - limit} characters elided from the middle ...]"
        if limit <= len(marker):
            # The allowance is smaller than the note explaining the trim. Say
            # only that, rather than returning something longer than the limit
            # and quietly breaking the caller's arithmetic.
            return marker, True
        head = int(limit * 0.6)
        tail = limit - head
        return f"{content[:head]}\n{marker}\n{content[-tail:]}", True

    # -- whole conversations ---------------------------------------------
    def fit(
        self, messages: Sequence[Message], trace_id: str | None = None
    ) -> tuple[list[Message], Elision]:
        """Return a transcript that fits, plus a record of what was given up."""
        result = list(messages)
        report = Elision(tokens_before=conversation_tokens(result))
        report.tokens_after = report.tokens_before
        if report.tokens_before <= self.available_tokens:
            return result, report

        # Oldest first, so the least relevant output is the first to go.
        for index in self._elidable_indices(len(result)):
            if report.tokens_after <= self.available_tokens:
                break
            message = result[index]
            if message.role is not Role.TOOL or _is_receipt(message.content):
                continue
            before = message_tokens(message)
            result[index] = Message(
                role=Role.TOOL,
                content=_receipt(message.content, trace_id),
                tool_call_id=message.tool_call_id,
            )
            saved = before - message_tokens(result[index])
            report.elided_messages += 1
            report.tokens_saved += saved
            report.tokens_after -= saved

        if report.tokens_after > self.available_tokens:
            # Every observation is a receipt and it still does not fit, so the
            # remaining weight is the agent's own commentary. Shrink that next,
            # to a hard floor rather than the per-observation ceiling: the
            # ceiling exists to stop one enormous log, and is far too generous
            # to help when the problem is many medium-sized turns.
            for index in self._elidable_indices(len(result)):
                if report.tokens_after <= self.available_tokens:
                    break
                message = result[index]
                if message.role is not Role.ASSISTANT or not message.content:
                    continue
                capped, changed = self.cap_observation(
                    message.content, max_tokens=self.salvage_tokens
                )
                if not changed:
                    continue
                before = message_tokens(message)
                result[index] = Message(
                    role=Role.ASSISTANT, content=capped, tool_calls=message.tool_calls
                )
                saved = before - message_tokens(result[index])
                report.truncated_messages += 1
                report.tokens_saved += saved
                report.tokens_after -= saved

        if report.tokens_after > self.available_tokens:
            # Last resort: shrink the recent turns too. Keeping them whole is a
            # preference - it protects the context the next decision uses -
            # whereas keeping the instruction is the invariant. A degraded run
            # beats a run that dies on a context-length error.
            report = self._last_resort(result, report)

        report.fits = report.tokens_after <= self.available_tokens
        return result, report

    def _last_resort(self, result: list[Message], report: Elision) -> Elision:
        start = min(self.keep_leading_turns, len(result))
        shrinkable = [i for i in range(start, len(result)) if result[i].content]
        if not shrinkable:
            return report

        # Derive the per-message allowance from the actual deficit. A fixed
        # floor cannot converge: sixteen messages at a 128-token floor already
        # exceed a 1,500-token window, so the pass would shrink everything to
        # its floor and still not fit.
        protected = sum(message_tokens(result[i]) for i in range(start))
        overhead = MESSAGE_OVERHEAD_TOKENS * len(shrinkable)
        room = self.available_tokens - protected - overhead
        allowance = max(MIN_SALVAGE_TOKENS, room // max(1, len(shrinkable)))

        for index in shrinkable:
            if report.tokens_after <= self.available_tokens:
                break
            message = result[index]
            capped, changed = self.cap_observation(message.content, max_tokens=allowance)
            if not changed:
                continue
            before = message_tokens(message)
            result[index] = Message(
                role=message.role,
                content=capped,
                tool_calls=message.tool_calls,
                tool_call_id=message.tool_call_id,
            )
            saved = before - message_tokens(result[index])
            report.truncated_messages += 1
            report.tokens_saved += saved
            report.tokens_after -= saved
        return report

    def _elidable_indices(self, count: int) -> list[int]:
        """Indices that may be shrunk, oldest first.

        Excludes the leading messages (system prompt, original task) and the
        most recent turns, which is what keeps elision from removing the
        context the next decision depends on.
        """
        start = min(self.keep_leading_turns, count)
        end = max(start, count - self.keep_recent_turns)
        return list(range(start, end))


#: Marker that identifies an already-elided observation, so a second pass does
#: not wrap a receipt inside another receipt.
RECEIPT_PREFIX = "[observation elided"


def _is_receipt(content: str) -> bool:
    return content.startswith(RECEIPT_PREFIX)


def _receipt(content: str, trace_id: str | None) -> str:
    """Replace an old observation with a note about what it was."""
    first_line = content.strip().splitlines()[0][:120] if content.strip() else ""
    where = f", full text in trace {trace_id}" if trace_id else ""
    return (
        f"{RECEIPT_PREFIX} to save context: "
        f"{estimate_tokens(content)} tokens{where}]\n"
        f"First line was: {first_line}"
    )


@dataclass
class BudgetCounters:
    """Cumulative savings, for the dashboard."""

    elisions: int = 0
    tokens_saved: int = 0
    history: list[Elision] = field(default_factory=list)

    def record(self, report: Elision) -> None:
        if report.changed:
            self.elisions += 1
            self.tokens_saved += report.tokens_saved
            self.history.append(report)
