"""Newline-delimited framing over byte streams.

Newline-delimited JSON is the framing that stdio agents converge on, and it has
exactly one hazard: a payload containing a raw newline silently corrupts the
next message. :func:`gantry.rpc.protocol.encode` forecloses that by escaping
non-ASCII and never pretty-printing, and the writer here asserts it rather than
trusting it, because a framing bug presents as an unrelated parse error three
messages later.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import threading
from typing import IO, Protocol

from gantry.errors import TransportError


class Transport(Protocol):
    """A bidirectional line-oriented channel."""

    def read_line(self) -> str | None:
        """Return the next line without its terminator, or ``None`` at EOF."""

    def write_line(self, line: str) -> None: ...

    def close(self) -> None: ...


class StreamTransport:
    """Framing over an arbitrary reader and writer.

    Writes are serialised: a JSON-RPC peer may be written to from several
    threads (a response here, a progress notification there) and interleaved
    partial writes would corrupt the stream.
    """

    def __init__(self, reader: IO[str], writer: IO[str]) -> None:
        self._reader = reader
        self._writer = writer
        self._write_lock = threading.Lock()
        self._closed = False

    def read_line(self) -> str | None:
        line = self._reader.readline()
        if line == "":
            return None  # EOF, as distinct from an empty line
        return line.rstrip("\r\n")

    def write_line(self, line: str) -> None:
        if self._closed:
            raise TransportError("transport is closed")
        if "\n" in line or "\r" in line:
            raise TransportError(
                "refusing to write a framed message containing a line break",
                hint="serialise with gantry.rpc.protocol.encode",
            )
        with self._write_lock:
            self._writer.write(line + "\n")
            self._writer.flush()

    def close(self) -> None:
        self._closed = True


class StdioTransport(StreamTransport):
    """The process's own stdin and stdout.

    Standard output belongs to the protocol once this is in use, so anything
    that would otherwise print must go to stderr. :meth:`redirect_stdout_to_stderr`
    makes that failure mode impossible rather than merely documented - a stray
    ``print`` in a tool handler would otherwise be indistinguishable from a
    malformed message to the peer.
    """

    def __init__(self) -> None:
        super().__init__(sys.stdin, sys.stdout)

    @staticmethod
    def redirect_stdout_to_stderr() -> None:
        sys.stdout = sys.stderr


class MemoryTransport:
    """An in-memory transport for tests.

    Lets the whole protocol stack be exercised without pipes, subprocesses or
    timing, which is what keeps the transport test suite fast and deterministic.
    """

    def __init__(self, incoming: list[str] | None = None) -> None:
        self.incoming = list(incoming or [])
        self.outgoing: list[str] = []
        self._closed = False

    def read_line(self) -> str | None:
        if not self.incoming:
            return None
        return self.incoming.pop(0)

    def write_line(self, line: str) -> None:
        if self._closed:
            raise TransportError("transport is closed")
        if "\n" in line or "\r" in line:
            raise TransportError("refusing to write a framed message containing a line break")
        self.outgoing.append(line)

    def close(self) -> None:
        self._closed = True

    def feed(self, *lines: str) -> None:
        self.incoming.extend(lines)


class SubprocessTransport(StreamTransport):
    """Speaks to a child process over its stdin and stdout.

    This is how the harness is driven by, or drives, another stdio agent - the
    same shape MCP servers use.
    """

    def __init__(self, command: list[str], cwd: str | None = None, env: dict | None = None) -> None:
        self.process = subprocess.Popen(  # noqa: S603 - command is caller-supplied by design
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env={**os.environ, **(env or {})},
            text=True,
            bufsize=1,  # line buffered, so a written line is actually sent
        )
        if self.process.stdin is None or self.process.stdout is None:
            raise TransportError("failed to open pipes to the child process")
        super().__init__(self.process.stdout, self.process.stdin)

    def close(self) -> None:
        super().close()
        if self.process.poll() is None:
            try:
                if self.process.stdin and not self.process.stdin.closed:
                    self.process.stdin.close()
                self.process.wait(timeout=5)
            except (subprocess.TimeoutExpired, ValueError, OSError):
                self.process.kill()
                self.process.wait(timeout=5)

    def stderr_text(self) -> str:
        """Drain the child's stderr. Where a crashing server explains itself."""
        if self.process.stderr is None:
            return ""
        try:
            return self.process.stderr.read() or ""
        except (ValueError, OSError, io.UnsupportedOperation):
            return ""
