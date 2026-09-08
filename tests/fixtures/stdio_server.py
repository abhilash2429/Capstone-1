"""A minimal Gantry JSON-RPC server, run as a child process by the tests.

Exists to prove the transport works in its real deployment shape: a separate
process speaking newline-delimited JSON over stdin and stdout. In-memory tests
cannot catch buffering, framing or stdout-pollution bugs, and those are exactly
the ones that break a stdio agent in production.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gantry.rpc import RpcServer, StdioTransport


def main() -> None:
    # stdout now belongs to the protocol. Anything that would print must not.
    transport = StdioTransport()
    StdioTransport.redirect_stdout_to_stderr()

    server = RpcServer(transport=transport, name="stdio-fixture")
    server.register("add", lambda params, ctx: params["a"] + params["b"])
    server.register("upper", lambda params, ctx: params["text"].upper())
    server.register("boom", lambda params, ctx: 1 / 0)

    def chatty(params, ctx):
        # A tool handler that prints. Without the redirect this corrupts the
        # stream and the peer sees a parse error it cannot explain.
        print("this would corrupt the protocol stream")
        return "survived"

    server.register("chatty", chatty)
    server.serve_forever()


if __name__ == "__main__":
    main()
