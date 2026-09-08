"""Tool specifications, capabilities and results.

A tool is not just a function. It is a function plus a contract: what it
accepts, what it is allowed to touch, how long it may run, whether it can be
retried, and whether running it twice is safe. Recording all of that at
registration time is what lets the harness make decisions about a tool -
schedule it in parallel, refuse it under a restricted grant, retry it after a
timeout - without special-casing tools by name.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from gantry.contract import Cancellation


class Capability(StrEnum):
    """What a tool needs permission to do.

    Coarse on purpose. A capability set that is too fine-grained is one nobody
    reads, and an unread permission model is decoration.
    """

    FS_READ = "fs:read"
    FS_WRITE = "fs:write"
    PROC_EXEC = "proc:exec"
    NET = "net"
    ENV_READ = "env:read"
    SECRET_READ = "secret:read"

    @property
    def is_dangerous(self) -> bool:
        return self in {Capability.PROC_EXEC, Capability.NET, Capability.SECRET_READ}


@dataclass(frozen=True)
class Grant:
    """The capability envelope a run executes inside.

    Deny-by-default: a grant lists what is permitted, never what is forbidden.
    An allowlist fails closed when someone adds a new tool and forgets to think
    about permissions, and a denylist fails open, which is the wrong direction
    for the failure to point.
    """

    capabilities: frozenset[Capability] = frozenset()
    #: ``None`` means "any tool whose capabilities are covered". A set narrows
    #: it further to specific tool names.
    tools: frozenset[str] | None = None

    @classmethod
    def read_only(cls) -> Grant:
        return cls(capabilities=frozenset({Capability.FS_READ}))

    @classmethod
    def developer(cls) -> Grant:
        """Read, write and execute inside the workspace. No network, no secrets."""
        return cls(
            capabilities=frozenset({Capability.FS_READ, Capability.FS_WRITE, Capability.PROC_EXEC})
        )

    @classmethod
    def unrestricted(cls) -> Grant:
        return cls(capabilities=frozenset(Capability))

    def permits(self, spec: ToolSpec) -> bool:
        if self.tools is not None and spec.name not in self.tools:
            return False
        return spec.capabilities <= self.capabilities

    def missing_for(self, spec: ToolSpec) -> frozenset[Capability]:
        return frozenset(spec.capabilities - self.capabilities)


@dataclass
class ToolContext:
    """Everything a handler is given besides its arguments.

    Handlers receive a context rather than reaching for globals, so a tool can
    be unit-tested by constructing one, and so a tool physically cannot escape
    the workspace it was handed.
    """

    run_id: str = ""
    trace_id: str | None = None
    workspace: Path = field(default_factory=lambda: Path.cwd())
    cancellation: Cancellation = field(default_factory=Cancellation)
    deadline_s: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def remaining_s(self, now: float | None = None) -> float | None:
        if self.deadline_s is None:
            return None
        return max(0.0, self.deadline_s - (now if now is not None else time.monotonic()))


@dataclass
class ToolResult:
    """What a tool hands back.

    ``content`` is what the model sees; ``data`` is what the harness sees. They
    are separate because the useful representation for a language model (a
    readable string, truncated to a budget) and for a program (structured,
    complete) are rarely the same object, and conflating them is how eval
    scoring ends up parsing prose.
    """

    ok: bool = True
    content: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    error_code: str = ""
    duration_ms: float = 0.0
    truncated: bool = False
    original_bytes: int = 0

    @classmethod
    def success(cls, content: str, **data: Any) -> ToolResult:
        return cls(ok=True, content=content, data=data)

    @classmethod
    def failure(
        cls, content: str, error_code: str = "tool.execution_failed", **data: Any
    ) -> ToolResult:
        return cls(ok=False, content=content, error_code=error_code, data=data)

    def truncate(
        self, max_chars: int, marker: str = "\n... [truncated {n} more characters]"
    ) -> ToolResult:
        """Bound the model-visible payload.

        Truncation is recorded rather than silent. An agent that cannot see it
        was given a partial file will confidently reason about the half it got,
        so the marker is part of the contract, not a nicety.
        """
        if len(self.content) <= max_chars:
            return self
        self.original_bytes = self.original_bytes or len(self.content)
        dropped = len(self.content) - max_chars
        self.content = self.content[:max_chars] + marker.format(n=dropped)
        self.truncated = True
        return self


#: Handlers take validated arguments plus a context and return a result.
ToolHandler = Callable[[dict[str, Any], ToolContext], ToolResult]


@dataclass(frozen=True)
class ToolSpec:
    """The registered description of one tool."""

    name: str
    version: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler
    capabilities: frozenset[Capability] = frozenset()
    #: Per-call wall-clock ceiling. The dispatcher enforces it.
    timeout_s: float = 30.0
    #: Running it twice with the same arguments has the same effect as once.
    idempotent: bool = True
    #: Mutates state that cannot be trivially reversed. Gated separately from
    #: capabilities so a run can permit writes but still require confirmation
    #: for destructive ones.
    destructive: bool = False
    #: Safe to execute concurrently with other tool calls in the same turn.
    concurrency_safe: bool = True
    #: Emit OpenAI structured-output strict mode for this tool's parameters.
    strict: bool = True
    #: Cap on model-visible output, applied by the dispatcher.
    max_output_chars: int = 16_000
    output_schema: dict[str, Any] | None = None

    @property
    def qualified_name(self) -> str:
        return f"{self.name}@{self.version}"

    @property
    def is_retryable(self) -> bool:
        """Only idempotent tools may be retried automatically.

        Retrying a non-idempotent tool after a timeout risks applying the same
        write twice, because a timeout says the answer was lost, not that the
        work was.
        """
        return self.idempotent

    def to_openai_tool(self) -> dict[str, Any]:
        """Render as an OpenAI Chat Completions function tool."""
        function: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
        }
        if self.strict:
            function["strict"] = True
        return {"type": "function", "function": function}
