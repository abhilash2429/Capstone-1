"""The conversation vocabulary shared by the provider and dispatch layers.

Typed rather than raw dicts, for one concrete reason: an agent transcript is
append-only and long-lived, and a dict typo in an assistant turn surfaces as a
provider 400 several turns later with no indication of which message was
malformed. The wire format is produced in exactly one place, here, so there is
one thing to get right.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from gantry.errors import ToolValidationError


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class ToolCall:
    """A tool call as a provider emitted it.

    ``arguments`` is a JSON *string* on the wire, and a model can emit a
    malformed one, so parsing is a normal recoverable step rather than an
    invariant. Kept here rather than in the dispatcher because it is part of
    the provider's output, and both layers need it.
    """

    id: str
    name: str
    arguments: str | dict[str, Any] = "{}"

    def arguments_json(self) -> str:
        if isinstance(self.arguments, dict):
            return json.dumps(self.arguments, separators=(",", ":"))
        return self.arguments or "{}"

    def parse_arguments(self) -> dict[str, Any]:
        """Decode the arguments, treating a malformed payload as recoverable.

        A model can and does emit invalid JSON here. That is a normal turn
        outcome to be explained back to it, not an exception for the harness.
        """
        if isinstance(self.arguments, dict):
            return self.arguments
        text = (self.arguments or "").strip() or "{}"
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ToolValidationError(
                f"tool {self.name!r}: arguments were not valid JSON "
                f"({exc.msg} at position {exc.pos})",
                tool=self.name,
            ) from exc
        if not isinstance(parsed, dict):
            raise ToolValidationError(
                f"tool {self.name!r}: arguments must be a JSON object, got {type(parsed).__name__}",
                tool=self.name,
            )
        return parsed

    def to_wire(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments_json()},
        }

    @classmethod
    def from_wire(cls, payload: Any) -> ToolCall:
        function = _get(payload, "function") or {}
        return cls(
            id=_get(payload, "id") or "",
            name=_get(function, "name") or "",
            arguments=_get(function, "arguments") or "{}",
        )


@dataclass(frozen=True)
class Message:
    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    #: Set on tool results, linking them back to the call they answer.
    tool_call_id: str | None = None

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str = "", tool_calls: tuple[ToolCall, ...] = ()) -> Message:
        return cls(role=Role.ASSISTANT, content=content, tool_calls=tuple(tool_calls))

    @classmethod
    def tool(cls, tool_call_id: str, content: str) -> Message:
        return cls(role=Role.TOOL, content=content, tool_call_id=tool_call_id)

    def to_wire(self) -> dict[str, Any]:
        if self.role is Role.TOOL:
            return {
                "role": "tool",
                "tool_call_id": self.tool_call_id,
                "content": self.content,
            }
        payload: dict[str, Any] = {"role": str(self.role)}
        if self.role is Role.ASSISTANT and self.tool_calls:
            # An assistant turn that only calls tools carries no text. The API
            # wants an explicit null here, not an empty string.
            payload["content"] = self.content or None
            payload["tool_calls"] = [c.to_wire() for c in self.tool_calls]
        else:
            payload["content"] = self.content
        return payload

    @classmethod
    def from_wire(cls, payload: Any) -> Message:
        raw_calls = _get(payload, "tool_calls") or ()
        return cls(
            role=Role(_get(payload, "role") or "assistant"),
            content=_get(payload, "content") or "",
            tool_calls=tuple(ToolCall.from_wire(c) for c in raw_calls),
            tool_call_id=_get(payload, "tool_call_id"),
        )


@dataclass(frozen=True)
class Usage:
    """Token accounting for one provider call."""

    input_tokens: int = 0
    output_tokens: int = 0
    #: The subset of input tokens served from the provider's prompt cache.
    cached_input_tokens: int = 0
    #: Reasoning tokens are billed as output but are not in the visible reply.
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }


class FinishReason(StrEnum):
    STOP = "stop"
    TOOL_CALLS = "tool_calls"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    UNKNOWN = "unknown"

    @classmethod
    def parse(cls, value: str | None) -> FinishReason:
        try:
            return cls(value or "unknown")
        except ValueError:
            # Providers add finish reasons over time; an unrecognised one must
            # not crash a run mid-flight.
            return cls.UNKNOWN


@dataclass
class Completion:
    """One provider response, normalised across providers."""

    message: Message
    finish_reason: FinishReason = FinishReason.UNKNOWN
    model: str = ""
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    #: True when served from the response cache rather than the provider.
    cached: bool = False
    response_id: str = ""
    raw: dict[str, Any] | None = None

    @property
    def text(self) -> str:
        return self.message.content

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        return self.message.tool_calls

    @property
    def wants_tools(self) -> bool:
        return bool(self.message.tool_calls)

    @property
    def truncated(self) -> bool:
        """The reply was cut off by the output cap rather than finishing.

        Worth surfacing: a truncated assistant turn often carries a half-written
        tool call, and treating it as a normal reply is how a loop gets stuck.
        """
        return self.finish_reason is FinishReason.LENGTH


def _get(obj: Any, key: str) -> Any:
    """Read a field from either a dict or an SDK model object.

    Provider SDKs return typed objects, caches and fixtures return dicts, and
    both flow through the same normalisation path.
    """
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def to_wire(messages: list[Message]) -> list[dict[str, Any]]:
    return [m.to_wire() for m in messages]
