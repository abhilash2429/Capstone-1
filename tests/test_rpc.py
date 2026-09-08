"""JSON-RPC conformance.

`handle_line` is a pure function from one line of text to zero or one lines,
so the specification's awkward corners are all testable as strings. The cases
below are the ones real implementations get wrong.
"""

from __future__ import annotations

import json
import threading

import pytest

from gantry.errors import CapabilityDenied, ToolValidationError, TransportError
from gantry.rpc import MemoryTransport, RpcClient, RpcServer
from gantry.rpc import protocol as p


@pytest.fixture
def server() -> RpcServer:
    s = RpcServer()
    s.register("add", lambda params, ctx: params["a"] + params["b"])
    s.register("echo", lambda params, ctx: params)
    s.register("null", lambda params, ctx: None)
    s.register("boom", lambda params, ctx: 1 / 0)
    return s


def reply(server: RpcServer, payload: dict | list) -> dict | list | None:
    out = server.handle_line(json.dumps(payload))
    return json.loads(out) if out is not None else None


# --- happy path ------------------------------------------------------------
def test_request_gets_a_result(server):
    assert reply(
        server, {"jsonrpc": "2.0", "method": "add", "params": {"a": 1, "b": 2}, "id": 1}
    ) == {"jsonrpc": "2.0", "id": 1, "result": 3}


def test_string_ids_are_echoed_unchanged(server):
    assert reply(server, {"jsonrpc": "2.0", "method": "null", "id": "abc"})["id"] == "abc"


def test_a_null_result_still_includes_the_result_member(server):
    """Otherwise the peer cannot tell success from a malformed message."""
    assert "result" in reply(server, {"jsonrpc": "2.0", "method": "null", "id": 1})


def test_positional_params_are_accepted(server):
    server.register("first", lambda params, ctx: params[0])
    assert (
        reply(server, {"jsonrpc": "2.0", "method": "first", "params": ["x"], "id": 1})["result"]
        == "x"
    )


# --- notifications ---------------------------------------------------------
def test_a_notification_gets_no_reply(server):
    assert server.handle_line('{"jsonrpc":"2.0","method":"add","params":{"a":1,"b":2}}') is None


def test_a_failing_notification_still_gets_no_reply(server):
    assert server.handle_line('{"jsonrpc":"2.0","method":"boom"}') is None


def test_an_unknown_notification_gets_no_reply(server):
    assert server.handle_line('{"jsonrpc":"2.0","method":"nope"}') is None


def test_null_id_is_a_request_not_a_notification(server):
    """The spec permits a null id and only discourages it. Answering one with
    silence is a real interoperability bug."""
    response = reply(
        server, {"jsonrpc": "2.0", "method": "add", "params": {"a": 1, "b": 2}, "id": None}
    )
    assert response == {"jsonrpc": "2.0", "id": None, "result": 3}


# --- errors ----------------------------------------------------------------
def test_unparseable_input_is_answered_with_a_null_id(server):
    response = json.loads(server.handle_line("{not json"))
    assert response["id"] is None
    assert response["error"]["code"] == p.PARSE_ERROR


def test_unknown_method(server):
    error = reply(server, {"jsonrpc": "2.0", "method": "nope", "id": 1})["error"]
    assert error["code"] == p.METHOD_NOT_FOUND
    assert "add" in error["data"]["available"]


@pytest.mark.parametrize(
    "payload",
    [
        {"method": "add", "id": 1},
        {"jsonrpc": "1.0", "method": "add", "id": 1},
        {"jsonrpc": "2.0", "id": 1},
        {"jsonrpc": "2.0", "method": "", "id": 1},
        {"jsonrpc": "2.0", "method": 7, "id": 1},
        {"jsonrpc": "2.0", "method": "add", "id": True},
    ],
)
def test_malformed_requests_are_invalid(server, payload):
    assert reply(server, payload)["error"]["code"] == p.INVALID_REQUEST


def test_scalar_params_are_invalid_params(server):
    assert (
        reply(server, {"jsonrpc": "2.0", "method": "add", "params": 7, "id": 1})["error"]["code"]
        == p.INVALID_PARAMS
    )


def test_a_recoverable_id_is_echoed_on_an_invalid_request(server):
    """So the peer can correlate the failure with the call it made."""
    assert reply(server, {"jsonrpc": "1.0", "method": "add", "id": 42})["id"] == 42


def test_a_raising_handler_becomes_an_internal_error_not_a_crash(server):
    error = reply(server, {"jsonrpc": "2.0", "method": "boom", "id": 1})["error"]
    assert error["code"] == p.INTERNAL_ERROR
    assert "ZeroDivisionError" in error["message"]


def test_harness_errors_map_to_their_own_codes(server):
    def denied(params, ctx):
        raise CapabilityDenied("no write access", missing=["fs:write"])

    def invalid(params, ctx):
        raise ToolValidationError("bad path", errors=["path: required"])

    server.register("denied", denied)
    server.register("invalid", invalid)
    assert (
        reply(server, {"jsonrpc": "2.0", "method": "denied", "id": 1})["error"]["code"]
        == p.CAPABILITY_DENIED
    )
    assert (
        reply(server, {"jsonrpc": "2.0", "method": "invalid", "id": 2})["error"]["code"]
        == p.INVALID_PARAMS
    )


# --- batches ---------------------------------------------------------------
def test_batch_returns_only_the_requests(server):
    responses = reply(
        server,
        [
            {"jsonrpc": "2.0", "method": "add", "params": {"a": 1, "b": 1}, "id": 1},
            {"jsonrpc": "2.0", "method": "add", "params": {"a": 2, "b": 2}},
            {"jsonrpc": "2.0", "method": "add", "params": {"a": 3, "b": 3}, "id": 2},
        ],
    )
    assert [r["result"] for r in responses] == [2, 6]


def test_an_empty_batch_is_itself_an_invalid_request(server):
    assert reply(server, [])["error"]["code"] == p.INVALID_REQUEST


def test_a_batch_of_only_notifications_gets_no_reply(server):
    assert server.handle_line('[{"jsonrpc":"2.0","method":"add","params":{"a":1,"b":1}}]') is None


def test_a_bad_member_does_not_sink_the_whole_batch(server):
    responses = reply(
        server,
        [
            {"jsonrpc": "2.0", "method": "add", "params": {"a": 1, "b": 1}, "id": 1},
            "not an object",
            {"jsonrpc": "2.0", "method": "add", "params": {"a": 2, "b": 2}, "id": 2},
        ],
    )
    assert len(responses) == 3
    assert [r.get("result") for r in responses] == [2, None, 4]
    assert responses[1]["error"]["code"] == p.INVALID_REQUEST


def test_blank_lines_are_ignored(server):
    assert server.handle_line("   ") is None


# --- cancellation ----------------------------------------------------------
def test_cancellation_marks_an_inflight_request(server):
    started, may_finish = threading.Event(), threading.Event()

    def slow(params, ctx):
        started.set()
        may_finish.wait(2)
        return "done"

    server.register("slow", slow)
    result: list = []
    worker = threading.Thread(
        target=lambda: result.append(server.handle_line('{"jsonrpc":"2.0","method":"slow","id":9}'))
    )
    worker.start()
    assert started.wait(2)
    server.handle_line('{"jsonrpc":"2.0","method":"$/cancelRequest","params":{"id":9}}')
    may_finish.set()
    worker.join(3)
    assert json.loads(result[0])["error"]["code"] == p.REQUEST_CANCELLED


def test_cancelling_an_unknown_id_is_harmless(server):
    assert (
        server.handle_line('{"jsonrpc":"2.0","method":"$/cancelRequest","params":{"id":"gone"}}')
        is None
    )


# --- transport -------------------------------------------------------------
def test_transport_refuses_to_write_an_unframed_line():
    """A payload containing a newline corrupts the *next* message, which then
    surfaces as an unrelated parse error."""
    transport = MemoryTransport()
    with pytest.raises(TransportError, match="line break"):
        transport.write_line('{"a":"one\ntwo"}')


def test_encode_never_emits_a_newline():
    # U+2028 LINE SEPARATOR is included on purpose: it is a line terminator to
    # a JavaScript peer even though Python's splitlines is not the framing
    # here, and ensure_ascii=True is what escapes it.
    payload = "one\ntwo\u2028three\rfour"
    line = p.encode(p.Request(method="m", id=1, params={"text": payload}))
    assert "\n" not in line and "\r" not in line and "\u2028" not in line
    assert json.loads(line)["params"]["text"] == payload


def test_read_line_distinguishes_eof_from_a_blank_line():
    transport = MemoryTransport([""])
    assert transport.read_line() == ""
    assert transport.read_line() is None


def test_serve_forever_stops_at_eof(server):
    transport = MemoryTransport(
        [
            '{"jsonrpc":"2.0","method":"add","params":{"a":1,"b":1},"id":1}',
            '{"jsonrpc":"2.0","method":"add","params":{"a":2,"b":2}}',
        ]
    )
    server.transport = transport
    server.serve_forever()
    assert len(transport.outgoing) == 1
    assert json.loads(transport.outgoing[0])["result"] == 2


# --- client ----------------------------------------------------------------
class LoopbackTransport:
    """An in-memory client/server pair.

    The two directions are deliberately distinct. A line the *client* writes is
    a request for the server to handle; a line the *server* writes is an
    unsolicited notification for the client to read. Conflating them makes a
    server's own notification loop back into itself, which is a bug in the test
    double rather than in the code under test.
    """

    def __init__(self, server: RpcServer) -> None:
        self.server = server
        self._inbound: list[str] = []
        self._ready = threading.Semaphore(0)
        self._closed = False
        self.server_side = _ServerSide(self)
        server.transport = self.server_side

    def _enqueue(self, line: str) -> None:
        self._inbound.append(line)
        self._ready.release()

    # client side
    def write_line(self, line: str) -> None:
        reply = self.server.handle_line(line)
        if reply is not None:
            self._enqueue(reply)

    def read_line(self) -> str | None:
        if not self._ready.acquire(timeout=2) or self._closed:
            return None
        return self._inbound.pop(0)

    def close(self) -> None:
        self._closed = True
        self._ready.release()


class _ServerSide:
    """The server's write end: everything it writes is read by the client."""

    def __init__(self, loopback: LoopbackTransport) -> None:
        self._loopback = loopback

    def write_line(self, line: str) -> None:
        self._loopback._enqueue(line)

    def read_line(self) -> str | None:
        return None

    def close(self) -> None: ...


def test_client_round_trip(server):
    with RpcClient(LoopbackTransport(server), default_timeout_s=2) as client:
        assert client.call("add", {"a": 2, "b": 3}) == 5


def test_client_raises_on_an_error_reply(server):
    with (
        RpcClient(LoopbackTransport(server), default_timeout_s=2) as client,
        pytest.raises(p.JsonRpcError) as exc,
    ):
        client.call("nope")
    assert exc.value.code == p.METHOD_NOT_FOUND


def test_client_surfaces_notifications(server):
    received: list[tuple] = []

    def handler(params, ctx):
        ctx.notify("progress", {"pct": 50})
        return "ok"

    server.register("work", handler)
    transport = LoopbackTransport(server)
    with RpcClient(
        transport, on_notification=lambda m, p_: received.append((m, p_)), default_timeout_s=2
    ) as client:
        assert client.call("work") == "ok"
    assert received == [("progress", {"pct": 50})]


def test_client_times_out_rather_than_hanging(server):
    class Silent:
        def write_line(self, line): ...
        def read_line(self):
            threading.Event().wait(5)
            return None

        def close(self): ...

    with (
        RpcClient(Silent(), default_timeout_s=0.1) as client,
        pytest.raises(TransportError, match="timed out"),
    ):
        client.call("add", {"a": 1, "b": 1})


# --- subprocess round trip -------------------------------------------------
def test_a_real_child_process_speaks_the_protocol():
    """In-memory tests cannot catch buffering, framing or stdout pollution.
    This runs the server as an actual child process over real pipes."""
    import sys
    from pathlib import Path

    from gantry.rpc import SubprocessTransport

    script = Path(__file__).parent / "fixtures" / "stdio_server.py"
    transport = SubprocessTransport([sys.executable, str(script)])
    try:
        with RpcClient(transport, default_timeout_s=10) as client:
            assert client.call("add", {"a": 20, "b": 22}) == 42
            assert client.call("upper", {"text": "gantry"}) == "GANTRY"

            # A handler that prints must not corrupt the stream.
            assert client.call("chatty", {}) == "survived"

            # And the connection survives a handler that raises.
            with pytest.raises(p.JsonRpcError) as exc:
                client.call("boom", {})
            assert exc.value.code == p.INTERNAL_ERROR
            assert client.call("add", {"a": 1, "b": 1}) == 2
    finally:
        transport.close()
