"""JSON-RPC 2.0 message types, parsing and serialisation.

Implemented against the specification rather than from memory, because the
parts people skip are exactly the parts that make a transport interoperable:
a notification is a request *without* an ``id`` member (not one with ``id:
null``), a parse error is answered with ``id: null``, an empty batch is itself
an invalid request, and a batch of only notifications gets no reply at all.

Keeping this module free of any I/O means the whole protocol layer is testable
against strings, with no pipes, no subprocesses and no sleeping.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

JSONRPC_VERSION = "2.0"

# --- specification error codes --------------------------------------------
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
#: -32000 to -32099 is reserved for implementation-defined server errors.
SERVER_ERROR = -32000

#: Gantry's own codes, inside the reserved server-error range.
REQUEST_CANCELLED = -32001
REQUEST_TIMEOUT = -32002
CAPABILITY_DENIED = -32003

_ERROR_MESSAGES = {
    PARSE_ERROR: "Parse error",
    INVALID_REQUEST: "Invalid Request",
    METHOD_NOT_FOUND: "Method not found",
    INVALID_PARAMS: "Invalid params",
    INTERNAL_ERROR: "Internal error",
    SERVER_ERROR: "Server error",
    REQUEST_CANCELLED: "Request cancelled",
    REQUEST_TIMEOUT: "Request timed out",
    CAPABILITY_DENIED: "Capability denied",
}


class JsonRpcError(Exception):
    """An error that should become a JSON-RPC error response."""

    def __init__(self, code: int, message: str | None = None, data: Any = None) -> None:
        self.code = code
        self.message = message or _ERROR_MESSAGES.get(code, "Server error")
        self.data = data
        super().__init__(f"[{code}] {self.message}")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            payload["data"] = self.data
        return payload


@dataclass(frozen=True)
class Request:
    """A call expecting a response.

    ``id`` may be null. The specification permits it ("a String, Number, or
    NULL value if included") while discouraging its use, so it is accepted and
    answered rather than rejected. Cancellation cannot target a null-id
    request, since several of them are indistinguishable.
    """

    method: str
    id: str | int | None
    params: dict[str, Any] | list[Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"jsonrpc": JSONRPC_VERSION, "method": self.method, "id": self.id}
        if self.params is not None:
            payload["params"] = self.params
        return payload


@dataclass(frozen=True)
class Notification:
    """A call expecting no response.

    Distinguished from a request by the *absence* of an ``id`` member. A
    message carrying ``"id": null`` is a request with a null id, which the
    specification discourages but does not forbid, and answering one of those
    with silence is a real interoperability bug.
    """

    method: str
    params: dict[str, Any] | list[Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"jsonrpc": JSONRPC_VERSION, "method": self.method}
        if self.params is not None:
            payload["params"] = self.params
        return payload


@dataclass(frozen=True)
class Response:
    """A successful or failed reply, carrying exactly one of result or error."""

    id: str | int | None
    result: Any = None
    error: dict[str, Any] | None = None

    @classmethod
    def ok(cls, id: str | int | None, result: Any) -> Response:
        return cls(id=id, result=result)

    @classmethod
    def failure(
        cls, id: str | int | None, code: int, message: str | None = None, data: Any = None
    ) -> Response:
        return cls(id=id, error=JsonRpcError(code, message, data).to_dict())

    @property
    def is_error(self) -> bool:
        return self.error is not None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"jsonrpc": JSONRPC_VERSION, "id": self.id}
        if self.error is not None:
            payload["error"] = self.error
        else:
            # `result` must be present on success even when it is null, so the
            # peer can tell success from a malformed message.
            payload["result"] = self.result
        return payload


Message = Request | Notification | Response


@dataclass
class Batch:
    """A JSON-RPC batch: several messages carried in one array."""

    messages: list[Request | Notification] = field(default_factory=list)


def parse_message(payload: Any) -> Request | Notification:
    """Turn one decoded JSON value into a request or notification.

    Raises :class:`JsonRpcError` with :data:`INVALID_REQUEST` for anything that
    does not satisfy the specification's shape rules.
    """
    if not isinstance(payload, dict):
        raise JsonRpcError(INVALID_REQUEST, data="a request must be a JSON object")
    if payload.get("jsonrpc") != JSONRPC_VERSION:
        raise JsonRpcError(INVALID_REQUEST, data=f"jsonrpc must be exactly {JSONRPC_VERSION!r}")
    method = payload.get("method")
    if not isinstance(method, str) or not method:
        raise JsonRpcError(INVALID_REQUEST, data="method must be a non-empty string")

    params = payload.get("params")
    if params is not None and not isinstance(params, (dict, list)):
        raise JsonRpcError(INVALID_PARAMS, data="params must be an object or an array")

    if "id" not in payload:
        return Notification(method=method, params=params)

    identifier = payload["id"]
    # Booleans are ints in Python but are not valid JSON-RPC ids. Null is
    # valid, and an explicit null id makes this a request, not a notification.
    if isinstance(identifier, bool) or not isinstance(identifier, (str, int, type(None))):
        raise JsonRpcError(INVALID_REQUEST, data="id must be a string, a number or null")
    return Request(method=method, id=identifier, params=params)


def parse_response(payload: Any) -> Response:
    """Turn one decoded JSON value into a response, for the client side."""
    if not isinstance(payload, dict):
        raise JsonRpcError(INVALID_REQUEST, data="a response must be a JSON object")
    if payload.get("jsonrpc") != JSONRPC_VERSION:
        raise JsonRpcError(INVALID_REQUEST, data=f"jsonrpc must be exactly {JSONRPC_VERSION!r}")
    has_result = "result" in payload
    has_error = "error" in payload
    if has_result == has_error:
        raise JsonRpcError(
            INVALID_REQUEST, data="a response must carry exactly one of result or error"
        )
    return Response(id=payload.get("id"), result=payload.get("result"), error=payload.get("error"))


def encode(message: Any) -> str:
    """Serialise to a single line of JSON.

    ``ensure_ascii`` keeps the payload to one line even for text containing
    line separators, which matters when the framing *is* the newline.
    """
    payload = message.to_dict() if hasattr(message, "to_dict") else message
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def decode(line: str) -> Any:
    """Parse one line of JSON, raising a JSON-RPC parse error on failure."""
    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        raise JsonRpcError(PARSE_ERROR, data=str(exc)) from exc


def error_response(id: str | int | None, exc: JsonRpcError) -> dict[str, Any]:
    return {"jsonrpc": JSONRPC_VERSION, "id": id, "error": exc.to_dict()}


MessageKind = Literal["request", "notification", "response", "batch"]
