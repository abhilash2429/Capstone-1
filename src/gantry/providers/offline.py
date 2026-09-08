"""A deterministic provider that needs no credentials and costs nothing.

This is not a mock bolted on for the tests. It is the reason the whole
repository is runnable: the full agent loop, the eval suite and CI all execute
against it, so anyone can clone this project and watch an agent solve tasks
without an API key, and a continuous-integration run cannot quietly spend
money.

Two modes:

* **Scripted** - you supply the turns the model would have taken. Precise, and
  what the loop and eval tests use to drive specific paths, including the ugly
  ones (a malformed tool call, a refusal, an output cut off mid-call).
* **Policy** - you supply a function from request to turn. Useful for fixtures
  where the agent should genuinely react to what the tools returned rather than
  follow a fixed sequence.

Everything is derived from the request, so the same conversation always
produces the same completion. That is what makes an eval number reproducible.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gantry.errors import ProviderError
from gantry.messages import Completion, FinishReason, Message, ToolCall
from gantry.providers.base import CompletionRequest, Provider

#: Priced at zero in the bundled table, so offline runs report $0.00 as a fact
#: rather than as a missing price.
OFFLINE_MODEL = "offline-fixture"


@dataclass
class Turn:
    """One scripted assistant turn."""

    text: str = ""
    #: ``(tool_name, arguments)``. Arguments may be a dict, or a raw string when
    #: the point of the fixture is a malformed payload.
    calls: tuple[tuple[str, Any], ...] = ()
    finish_reason: FinishReason | None = None

    def to_completion(self, index: int) -> Completion:
        tool_calls = tuple(
            ToolCall(id=f"call_offline_{index}_{i}", name=name, arguments=arguments)
            for i, (name, arguments) in enumerate(self.calls)
        )
        reason = self.finish_reason or (
            FinishReason.TOOL_CALLS if tool_calls else FinishReason.STOP
        )
        return Completion(
            message=Message.assistant(self.text, tool_calls),
            finish_reason=reason,
            model=OFFLINE_MODEL,
            response_id=f"offline-{index}",
        )


@dataclass
class Script:
    """A fixed sequence of turns, replayed in order."""

    turns: list[Turn] = field(default_factory=list)
    #: What to say once the script runs out. A script that simply stops would
    #: leave the loop waiting for a reply that never comes, so exhaustion has a
    #: defined, terminating answer.
    on_exhausted: str = "Offline script exhausted; stopping."

    @classmethod
    def of(cls, *turns: Turn | str) -> Script:
        return cls(turns=[Turn(text=t) if isinstance(t, str) else t for t in turns])

    @classmethod
    def calling(cls, tool: str, arguments: Any, then: str = "Done.") -> Script:
        """A two-turn script: call one tool, then answer."""
        return cls(turns=[Turn(calls=((tool, arguments),)), Turn(text=then)])


Policy = Callable[[CompletionRequest, int], Turn | Completion | None]


class OfflineProvider(Provider):
    """Replays scripted or policy-driven turns with no network access."""

    name = "offline"

    def __init__(
        self,
        script: Script | Iterable[Turn] | None = None,
        policy: Policy | None = None,
        model: str = OFFLINE_MODEL,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if isinstance(script, Script) or script is None:
            self.script = script
        else:
            self.script = Script(turns=list(script))
        self.policy = policy
        self.model = model
        self.calls: list[CompletionRequest] = []

    def model_for(self, role: str) -> str:
        return self.model

    @property
    def turn_index(self) -> int:
        return len(self.calls)

    def _complete(self, request: CompletionRequest, model: str) -> Completion:
        index = self.turn_index
        self.calls.append(request)

        turn: Turn | Completion | None = None
        if self.policy is not None:
            turn = self.policy(request, index)
        if turn is None and self.script is not None:
            turn = (
                self.script.turns[index]
                if index < len(self.script.turns)
                else Turn(text=self.script.on_exhausted)
            )
        if turn is None:
            turn = Turn(text="No offline script or policy is configured for this request.")

        completion = turn.to_completion(index) if isinstance(turn, Turn) else turn
        completion.model = model
        completion.usage = self.estimate_usage(request, completion.text)
        return completion

    # -- recorded traffic -------------------------------------------------
    @classmethod
    def from_jsonl(cls, path: str | Path, **kwargs: Any) -> OfflineProvider:
        """Replay completions recorded from a real provider.

        Recording a real session once and replaying it forever is how an eval
        number stays reproducible after the model behind a deployment changes
        underneath you.
        """
        turns: list[Turn] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            message = Message.from_wire(payload.get("message", {}))
            turns.append(
                Turn(
                    text=message.content,
                    calls=tuple((c.name, c.arguments) for c in message.tool_calls),
                    finish_reason=FinishReason.parse(payload.get("finish_reason")),
                )
            )
        return cls(script=Script(turns=turns), **kwargs)


class FailingProvider(Provider):
    """Raises on every call. Exercises the loop's provider-error path.

    A harness that has never been run against a broken provider has an
    untested failure mode, and it is the one that fires during an incident.
    """

    name = "failing"

    def __init__(self, error: ProviderError | None = None, fail_after: int = 0, **kwargs: Any):
        super().__init__(**kwargs)
        self.error = error or ProviderError("provider is unavailable")
        self.fail_after = fail_after
        self.attempts = 0

    def model_for(self, role: str) -> str:
        return OFFLINE_MODEL

    def _complete(self, request: CompletionRequest, model: str) -> Completion:
        self.attempts += 1
        if self.attempts > self.fail_after:
            raise self.error
        return Completion(
            message=Message.assistant("ok"), finish_reason=FinishReason.STOP, model=model
        )
