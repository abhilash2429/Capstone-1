"""A JSON-RPC 2.0 server over a line transport.

The design decision that matters here is that :meth:`RpcServer.handle_line`
is a pure function from one line of text to zero or one lines of text. All the
specification's awkward cases - batches, notifications, parse errors, null ids -
are then testable as strings, with no pipes, no threads and no sleeping. Only
:meth:`serve_forever` touches I/O, and it is four lines long.

Handler exceptions never escape. An unhandled error becomes an ``Internal
error`` response, because a transport that dies on a bad request takes the
whole agent session with it.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from gantry.contract import Cancellation
from gantry.errors import CapabilityDenied, GantryError, ToolValidationError
from gantry.rpc import protocol as p
from gantry.rpc.transport import Transport

log = logging.getLogger("gantry.rpc")

#: LSP's cancellation notification, reused here because stdio agent tooling
#: already speaks it.
CANCEL_METHOD = "$/cancelRequest"


@dataclass
class RpcContext:
    """Per-request state handed to a method handler."""

    request_id: str | int | None = None
    method: str = ""
    cancellation: Cancellation = field(default_factory=Cancellation)
    server: RpcServer | None = None

    def notify(self, method: str, params: Any = None) -> None:
        """Send an out-of-band notification to the peer, e.g. progress."""
        if self.server is not None:
            self.server.notify(method, params)


Handler = Callable[[Any, RpcContext], Any]


def _map_exception(exc: Exception) -> p.JsonRpcError:
    """Translate a harness error into the closest JSON-RPC error.

    Every :class:`GantryError` already carries an ``rpc_code``, so the mapping
    is data rather than a chain of isinstance checks, and a new error type gets
    a sensible wire representation for free.
    """
    if isinstance(exc, p.JsonRpcError):
        return exc
    if isinstance(exc, ToolValidationError):
        return p.JsonRpcError(p.INVALID_PARAMS, exc.message, exc.details)
    if isinstance(exc, CapabilityDenied):
        return p.JsonRpcError(p.CAPABILITY_DENIED, exc.message, exc.details)
    if isinstance(exc, GantryError):
        return p.JsonRpcError(exc.rpc_code, exc.message, exc.to_dict())
    return p.JsonRpcError(p.INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")


class RpcServer:
    """Routes JSON-RPC methods to handlers."""

    def __init__(self, transport: Transport | None = None, name: str = "gantry") -> None:
        self.transport = transport
        self.name = name
        self._handlers: dict[str, Handler] = {}
        self._inflight: dict[str | int, Cancellation] = {}
        self._lock = threading.Lock()
        self.register(CANCEL_METHOD, self._handle_cancel)

    # -- registration ---------------------------------------------------
    def register(self, method: str, handler: Handler) -> None:
        if method in self._handlers:
            raise ValueError(f"method {method!r} is already registered")
        self._handlers[method] = handler

    def method(self, name: str) -> Callable[[Handler], Handler]:
        def decorate(handler: Handler) -> Handler:
            self.register(name, handler)
            return handler

        return decorate

    @property
    def methods(self) -> list[str]:
        return sorted(self._handlers)

    # -- cancellation ---------------------------------------------------
    def _handle_cancel(self, params: Any, ctx: RpcContext) -> None:
        target = (params or {}).get("id") if isinstance(params, dict) else None
        if target is None:
            return None
        with self._lock:
            token = self._inflight.get(target)
        if token is not None:
            token.cancel(f"cancelled by peer via {CANCEL_METHOD}")
        return None

    # -- message handling -----------------------------------------------
    def handle_line(self, line: str) -> str | None:
        """Process one framed message. Returns the reply line, or ``None``.

        ``None`` is returned when the specification says to stay silent: for a
        notification, and for a batch containing only notifications.
        """
        if not line.strip():
            return None
        try:
            payload = p.decode(line)
        except p.JsonRpcError as exc:
            # A message that could not be parsed has no recoverable id, so the
            # specification requires answering with a null id.
            return p.encode(p.error_response(None, exc))

        if isinstance(payload, list):
            return self._handle_batch(payload)
        response = self._handle_one(payload)
        return p.encode(response) if response is not None else None

    def _handle_batch(self, payloads: list[Any]) -> str | None:
        if not payloads:
            # An empty array is itself an invalid request, not an empty batch.
            return p.encode(
                p.error_response(None, p.JsonRpcError(p.INVALID_REQUEST, data="empty batch"))
            )
        responses = [r for r in (self._handle_one(item) for item in payloads) if r is not None]
        if not responses:
            # A batch of only notifications gets no reply at all.
            return None
        return p.encode([r.to_dict() for r in responses])

    def _handle_one(self, payload: Any) -> p.Response | None:
        try:
            message = p.parse_message(payload)
        except p.JsonRpcError as exc:
            # Recover the id if the payload had a usable one, so the peer can
            # correlate the failure with the call it made.
            identifier = payload.get("id") if isinstance(payload, dict) else None
            if isinstance(identifier, bool) or not isinstance(identifier, (str, int)):
                identifier = None
            return p.Response(id=identifier, error=exc.to_dict())

        handler = self._handlers.get(message.method)
        is_request = isinstance(message, p.Request)
        request_id = message.id if is_request else None

        if handler is None:
            if not is_request:
                return None  # never answer a notification, even a bad one
            return p.Response.failure(
                request_id,
                p.METHOD_NOT_FOUND,
                data={"method": message.method, "available": self.methods},
            )

        ctx = RpcContext(request_id=request_id, method=message.method, server=self)
        # A null-id request is answerable but not cancellable: two of them are
        # indistinguishable, so tracking them under one key would let a cancel
        # for either abort both.
        trackable = is_request and request_id is not None
        if trackable:
            with self._lock:
                self._inflight[request_id] = ctx.cancellation
        try:
            result = handler(message.params, ctx)
        except Exception as exc:  # the transport must survive a handler that raises
            log.debug("handler for %s failed", message.method, exc_info=True)
            if not is_request:
                return None
            return p.Response(id=request_id, error=_map_exception(exc).to_dict())
        finally:
            if trackable:
                with self._lock:
                    self._inflight.pop(request_id, None)

        if not is_request:
            return None
        if ctx.cancellation.cancelled:
            return p.Response.failure(request_id, p.REQUEST_CANCELLED, data=ctx.cancellation.reason)
        return p.Response.ok(request_id, result)

    # -- outbound -------------------------------------------------------
    def notify(self, method: str, params: Any = None) -> None:
        if self.transport is None:
            return
        self.transport.write_line(p.encode(p.Notification(method=method, params=params)))

    # -- I/O ------------------------------------------------------------
    def serve_forever(self) -> None:
        """Read messages until EOF. The only part of this module that blocks."""
        if self.transport is None:
            raise RuntimeError("serve_forever needs a transport")
        while (line := self.transport.read_line()) is not None:
            reply = self.handle_line(line)
            if reply is not None:
                self.transport.write_line(reply)
