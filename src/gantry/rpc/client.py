"""A JSON-RPC 2.0 client over a line transport.

Requests are correlated by id against a reader thread, which is the only
workable shape for a full-duplex stdio peer: the server may interleave progress
notifications and out-of-order responses, and a naive read-after-write client
deadlocks the first time it does.
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Callable
from typing import Any

from gantry.errors import TransportError
from gantry.rpc import protocol as p
from gantry.rpc.transport import Transport


class RpcClient:
    """Sends requests and notifications, and correlates replies."""

    def __init__(
        self,
        transport: Transport,
        on_notification: Callable[[str, Any], None] | None = None,
        default_timeout_s: float = 60.0,
    ) -> None:
        self.transport = transport
        self.default_timeout_s = default_timeout_s
        self._on_notification = on_notification
        self._ids = itertools.count(1)
        self._pending: dict[str | int, threading.Event] = {}
        self._results: dict[str | int, p.Response] = {}
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._error: BaseException | None = None
        self._reader = threading.Thread(target=self._read_loop, name="rpc-client", daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        try:
            while not self._closed.is_set():
                line = self.transport.read_line()
                if line is None:
                    break
                if not line.strip():
                    continue
                payload = p.decode(line)
                for item in payload if isinstance(payload, list) else [payload]:
                    self._deliver(item)
        except BaseException as exc:  # noqa: BLE001 - surfaced to every waiter below
            self._error = exc
        finally:
            self._closed.set()
            # Release everyone still waiting, so a dead peer fails fast rather
            # than stalling every caller until its timeout.
            with self._lock:
                events = list(self._pending.values())
            for event in events:
                event.set()

    def _deliver(self, payload: Any) -> None:
        if isinstance(payload, dict) and "method" in payload and "id" not in payload:
            if self._on_notification is not None:
                self._on_notification(payload["method"], payload.get("params"))
            return
        response = p.parse_response(payload)
        with self._lock:
            event = self._pending.get(response.id)
            if event is None:
                return  # a reply to a request we are no longer waiting on
            self._results[response.id] = response
            event.set()

    def call(self, method: str, params: Any = None, timeout_s: float | None = None) -> Any:
        """Send a request and wait for its result. Raises on an error reply."""
        response = self.request(method, params, timeout_s)
        if response.is_error:
            error = response.error or {}
            raise p.JsonRpcError(
                error.get("code", p.INTERNAL_ERROR), error.get("message"), error.get("data")
            )
        return response.result

    def request(
        self, method: str, params: Any = None, timeout_s: float | None = None
    ) -> p.Response:
        if self._closed.is_set():
            raise TransportError("connection is closed", cause=repr(self._error))
        identifier = next(self._ids)
        event = threading.Event()
        with self._lock:
            self._pending[identifier] = event
        try:
            self.transport.write_line(
                p.encode(p.Request(method=method, id=identifier, params=params))
            )
            if not event.wait(timeout_s if timeout_s is not None else self.default_timeout_s):
                raise TransportError(f"timed out waiting for a reply to {method!r}", method=method)
            with self._lock:
                response = self._results.pop(identifier, None)
            if response is None:
                raise TransportError(
                    "connection closed before a reply arrived",
                    method=method,
                    cause=repr(self._error) if self._error else None,
                )
            return response
        finally:
            with self._lock:
                self._pending.pop(identifier, None)
                self._results.pop(identifier, None)

    def notify(self, method: str, params: Any = None) -> None:
        self.transport.write_line(p.encode(p.Notification(method=method, params=params)))

    def cancel(self, request_id: str | int) -> None:
        self.notify("$/cancelRequest", {"id": request_id})

    def close(self) -> None:
        self._closed.set()
        self.transport.close()

    def __enter__(self) -> RpcClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
