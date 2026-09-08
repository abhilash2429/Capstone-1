"""JSON-RPC 2.0 over newline-delimited stdio."""

from gantry.rpc import protocol
from gantry.rpc.client import RpcClient
from gantry.rpc.protocol import (
    JsonRpcError,
    Notification,
    Request,
    Response,
    decode,
    encode,
    parse_message,
    parse_response,
)
from gantry.rpc.server import RpcContext, RpcServer
from gantry.rpc.transport import (
    MemoryTransport,
    StdioTransport,
    StreamTransport,
    SubprocessTransport,
    Transport,
)

__all__ = [
    "JsonRpcError",
    "MemoryTransport",
    "Notification",
    "Request",
    "Response",
    "RpcClient",
    "RpcContext",
    "RpcServer",
    "StdioTransport",
    "StreamTransport",
    "SubprocessTransport",
    "Transport",
    "decode",
    "encode",
    "parse_message",
    "parse_response",
    "protocol",
]
