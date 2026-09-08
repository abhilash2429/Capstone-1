"""The observation budget.

The invariant that matters most is not the token count, it is that the
transcript stays valid for the provider afterwards. Every assistant tool call
must still be answered by a tool message with a matching id, so elision may
only rewrite content and never remove a turn.
"""

from __future__ import annotations

from gantry.budget import (
    RECEIPT_PREFIX,
    BudgetCounters,
    ObservationBudget,
    conversation_tokens,
    estimate_tokens,
)
from gantry.messages import Message, Role, ToolCall


def conversation(turns: int = 10, size: int = 1500) -> list[Message]:
    messages = [Message.system("You are an agent."), Message.user("Fix the failing test.")]
    for i in range(turns):
        messages.append(
            Message.assistant(tool_calls=(ToolCall(f"c{i}", "read_file", {"path": f"f{i}.py"}),))
        )
        messages.append(Message.tool(f"c{i}", f"file {i} contents\n" + "x" * size))
    return messages


def tight_budget(**kwargs) -> ObservationBudget:
    defaults = dict(
        max_context_tokens=2000,
        reserve_output_tokens=500,
        keep_recent_turns=2,
        keep_leading_turns=2,
    )
    return ObservationBudget(**{**defaults, **kwargs})


def test_a_transcript_that_fits_is_untouched():
    messages = conversation(turns=1, size=10)
    fitted, report = ObservationBudget().fit(messages)
    assert fitted == messages
    assert not report.changed


def test_an_oversized_transcript_is_brought_under_the_limit():
    budget = tight_budget()
    fitted, report = budget.fit(conversation())
    assert report.tokens_before > budget.available_tokens
    assert report.tokens_after <= budget.available_tokens
    assert conversation_tokens(fitted) <= budget.available_tokens


def test_elision_never_removes_a_turn():
    """Dropping a message to save tokens leaves an unanswered tool call, and
    the provider rejects the whole request with a 400."""
    messages = conversation()
    fitted, _ = tight_budget().fit(messages)
    assert len(fitted) == len(messages)
    calls = {c.id for m in fitted for c in m.tool_calls}
    answers = {m.tool_call_id for m in fitted if m.role is Role.TOOL}
    assert calls == answers


def test_the_system_prompt_and_the_task_are_never_shrunk():
    """An agent that forgets what it was asked finishes the wrong job."""
    messages = conversation()
    fitted, _ = tight_budget().fit(messages)
    assert fitted[0] == messages[0]
    assert fitted[1] == messages[1]


def test_recent_turns_are_never_shrunk():
    messages = conversation()
    fitted, _ = tight_budget(keep_recent_turns=4).fit(messages)
    assert fitted[-4:] == messages[-4:]


def test_the_oldest_observations_go_first():
    messages = conversation()
    fitted, _ = tight_budget().fit(messages)
    elided = [i for i, m in enumerate(fitted) if m.content.startswith(RECEIPT_PREFIX)]
    kept = [
        i
        for i, m in enumerate(fitted)
        if m.role is Role.TOOL and not m.content.startswith(RECEIPT_PREFIX)
    ]
    assert elided and kept
    assert max(elided) < min(kept)


def test_a_receipt_says_what_was_lost_and_where_to_find_it():
    messages = conversation(turns=6)
    fitted, _ = tight_budget().fit(messages, trace_id="abc123")
    receipt = next(m.content for m in fitted if m.content.startswith(RECEIPT_PREFIX))
    assert "abc123" in receipt
    assert "tokens" in receipt
    assert "file 0 contents" in receipt  # the first line survives as a hint


def test_a_receipt_is_not_wrapped_in_another_receipt():
    budget = tight_budget()
    once, _ = budget.fit(conversation())
    twice, second = budget.fit(once)
    assert twice == once
    assert not second.changed


def test_a_single_huge_observation_keeps_its_head_and_its_tail():
    """A naive head-only truncation discards the stack trace, which is the
    part that made the output worth reading."""
    budget = ObservationBudget(max_observation_tokens=100)
    content = "START\n" + "middle\n" * 5000 + "AssertionError: boom"
    capped, changed = budget.cap_observation(content)
    assert changed
    assert capped.startswith("START")
    assert capped.endswith("AssertionError: boom")
    assert "elided from the middle" in capped
    assert estimate_tokens(capped) < 200


def test_a_small_observation_is_left_alone():
    content, changed = ObservationBudget().cap_observation("short output")
    assert content == "short output"
    assert not changed


def test_assistant_turns_are_shrunk_only_after_every_observation_has_been():
    budget = tight_budget()
    messages = [Message.system("s"), Message.user("t")]
    for i in range(8):
        messages.append(Message.assistant("thinking out loud " * 200))
        messages.append(Message.tool(f"c{i}", "y" * 1500))
    fitted, report = budget.fit(messages)
    assert report.elided_messages > 0
    assert report.truncated_messages > 0
    assert conversation_tokens(fitted) <= budget.available_tokens
    assert report.fits


def test_the_last_resort_shrinks_recent_turns_rather_than_giving_up():
    """Keeping recent turns whole is a preference; keeping the instruction is
    the invariant. A degraded run beats one that dies on a context-length
    error."""
    budget = tight_budget(keep_recent_turns=6)
    messages = [Message.system("You are an agent."), Message.user("Fix the test.")]
    for i in range(8):
        messages.append(Message.assistant("verbose reasoning " * 300))
        messages.append(Message.tool(f"c{i}", "z" * 3000))
    fitted, report = budget.fit(messages)
    assert report.fits
    assert conversation_tokens(fitted) <= budget.available_tokens
    assert fitted[0].content == "You are an agent."
    assert fitted[1].content == "Fix the test."


def test_a_transcript_that_cannot_fit_says_so():
    """Reporting success while returning something oversized turns a
    context-length 400 into a mystery."""
    budget = ObservationBudget(max_context_tokens=40, reserve_output_tokens=20)
    huge = [Message.system("s" * 4000), Message.user("t" * 4000)]
    _, report = budget.fit(huge)
    assert not report.fits


def test_counters_accumulate_savings_for_the_dashboard():
    counters = BudgetCounters()
    budget = tight_budget()
    for _ in range(3):
        counters.record(budget.fit(conversation())[1])
    assert counters.elisions == 3
    assert counters.tokens_saved > 0


def test_the_estimate_is_identical_on_every_machine():
    """A budget that depends on a tokenizer download is a budget that fails
    in CI."""
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
