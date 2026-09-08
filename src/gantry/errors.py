"""Typed error hierarchy.

Two properties matter more than the class names. Every error carries a stable
string ``code``, so failures can be counted and alerted on without parsing
messages, and every error declares whether it is ``retryable``, so retry logic
lives in one place instead of being re-litigated at each call site.

The codes double as JSON-RPC error data, which is why they are strings rather
than an enum of integers - a harness that grows new failure modes should not
have to renumber anything.
"""

from __future__ import annotations

from typing import Any


class GantryError(Exception):
    """Base class for every error the harness raises deliberately."""

    code: str = "gantry.error"
    retryable: bool = False
    #: JSON-RPC 2.0 error code used when this error crosses the transport.
    rpc_code: int = -32000

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "details": self.details,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}({self.code!r}, {self.message!r})"


# --- configuration ---------------------------------------------------------
class ConfigError(GantryError):
    code = "config.invalid"
    rpc_code = -32602


# --- tools -----------------------------------------------------------------
class ToolError(GantryError):
    code = "tool.error"


class ToolNotFound(ToolError):
    code = "tool.not_found"
    rpc_code = -32601


class ToolValidationError(ToolError):
    """Arguments failed JSON Schema validation.

    Not retryable by the harness, but *is* recoverable by the model: the
    validation detail is fed back so the next turn can correct itself.
    """

    code = "tool.invalid_arguments"
    rpc_code = -32602


class ToolExecutionError(ToolError):
    code = "tool.execution_failed"


class ToolTimeout(ToolError):
    code = "tool.timeout"
    retryable = True


class CapabilityDenied(ToolError):
    """The tool exists but the current grant does not permit it."""

    code = "tool.capability_denied"
    rpc_code = -32003


class ToolRegistrationError(ToolError):
    code = "tool.registration_invalid"


# --- sandbox ---------------------------------------------------------------
class SandboxError(GantryError):
    code = "sandbox.error"


class PathEscape(SandboxError):
    """A path resolved outside the jail root."""

    code = "sandbox.path_escape"


class CommandDenied(SandboxError):
    code = "sandbox.command_denied"


class ResourceLimitExceeded(SandboxError):
    code = "sandbox.resource_limit"


# --- transport -------------------------------------------------------------
class TransportError(GantryError):
    code = "transport.error"


class ProtocolError(TransportError):
    code = "transport.protocol"
    rpc_code = -32600


class ParseError(TransportError):
    code = "transport.parse"
    rpc_code = -32700


# --- providers -------------------------------------------------------------
class ProviderError(GantryError):
    code = "provider.error"


class ProviderAuthError(ProviderError):
    code = "provider.auth"


class ProviderBadRequest(ProviderError):
    code = "provider.bad_request"


class ProviderRateLimited(ProviderError):
    code = "provider.rate_limited"
    retryable = True


class ProviderUnavailable(ProviderError):
    code = "provider.unavailable"
    retryable = True


class ProviderTimeout(ProviderError):
    code = "provider.timeout"
    retryable = True


# --- loop ------------------------------------------------------------------
class BudgetExceeded(GantryError):
    code = "loop.budget_exceeded"


class VerificationFailed(GantryError):
    code = "loop.verification_failed"


class Cancelled(GantryError):
    code = "loop.cancelled"
