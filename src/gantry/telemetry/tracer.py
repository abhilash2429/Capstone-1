"""A dependency-free tracer that speaks OpenTelemetry's GenAI vocabulary.

Why not import ``opentelemetry-sdk``? Two reasons that matter here. The value
in this module is *what* gets recorded - token accounting, per-span cost,
sandbox decisions, budget headroom - and that logic is identical whichever SDK
carries it. And a harness anyone can run with a bare Python interpreter, no
collector and no exporter configuration is far more useful than one that needs
an observability stack stood up first.

Spans are shaped exactly like OTel spans, and :mod:`gantry.telemetry.otlp`
serialises them into OTLP/JSON, so exporting to a real collector is a
configuration change rather than a rewrite.
"""

from __future__ import annotations

import contextvars
import secrets
import threading
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol

from gantry.telemetry import semconv as sc
from gantry.telemetry.pricing import PRICE_BOOK, PriceBook

_current_trace: contextvars.ContextVar[Trace | None] = contextvars.ContextVar(
    "gantry_trace", default=None
)
_current_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "gantry_span", default=None
)


def new_trace_id() -> str:
    """W3C trace-context: 16 random bytes as lowercase hex."""
    return secrets.token_hex(16)


def new_span_id() -> str:
    return secrets.token_hex(8)


@dataclass
class Span:
    span_id: str
    trace_id: str
    name: str
    kind: str
    started_at: float
    parent_span_id: str | None = None
    ended_at: float | None = None
    status: str = "unset"
    status_message: str | None = None
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duration_ms(self) -> float | None:
        return None if self.ended_at is None else (self.ended_at - self.started_at) * 1000.0

    def set_attribute(self, key: str, value: Any) -> Span:
        self.attributes[key] = value
        return self

    def set_attributes(self, attributes: dict[str, Any]) -> Span:
        self.attributes.update(attributes)
        return self

    def add_event(self, name: str, **attributes: Any) -> Span:
        self.events.append({"time": time.time(), "name": name, "attributes": attributes})
        return self

    def set_status(self, status: str, message: str | None = None) -> Span:
        self.status = status
        if message:
            self.status_message = message
        return self

    def record_exception(self, exc: BaseException) -> Span:
        self.add_event(
            "exception",
            **{
                "exception.type": type(exc).__name__,
                "exception.message": str(exc),
                "exception.stacktrace": "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                )[-4000:],
            },
        )
        return self.set_status("error", f"{type(exc).__name__}: {exc}")

    def record_usage(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
        price_book: PriceBook | None = None,
    ) -> Span:
        """Attach token usage and the cost it implies."""
        cost = (price_book or PRICE_BOOK).price(
            model, input_tokens, output_tokens, cached_input_tokens
        )
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cost_usd = cost.usd
        self.set_attributes(
            {
                sc.GEN_AI_USAGE_INPUT_TOKENS: input_tokens,
                sc.GEN_AI_USAGE_OUTPUT_TOKENS: output_tokens,
                **cost.as_attributes(),
            }
        )
        if cached_input_tokens:
            self.set_attribute(sc.GEN_AI_USAGE_CACHED_INPUT_TOKENS, cached_input_tokens)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "kind": self.kind,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "status_message": self.status_message,
            "cost_usd": self.cost_usd,
            "attributes": self.attributes,
            "events": self.events,
        }


@dataclass
class Trace:
    trace_id: str
    name: str
    started_at: float
    service: str = "gantry"
    environment: str = "local"
    ended_at: float | None = None
    status: str = "unset"
    attributes: dict[str, Any] = field(default_factory=dict)
    spans: list[Span] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_span(self, span: Span) -> None:
        with self._lock:
            self.spans.append(span)

    @property
    def duration_ms(self) -> float | None:
        return None if self.ended_at is None else (self.ended_at - self.started_at) * 1000.0

    @property
    def cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.spans)

    @property
    def input_tokens(self) -> int:
        return sum(s.input_tokens for s in self.spans)

    @property
    def output_tokens(self) -> int:
        return sum(s.output_tokens for s in self.spans)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "service": self.service,
            "environment": self.environment,
            "cost_usd": self.cost_usd,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "span_count": len(self.spans),
            "attributes": self.attributes,
        }


class Exporter(Protocol):
    def export(self, trace: Trace) -> None: ...


class MemoryExporter:
    """Collects traces in memory. Used by tests and by ``--no-store`` runs."""

    def __init__(self) -> None:
        self.traces: list[Trace] = []
        self._lock = threading.Lock()

    def export(self, trace: Trace) -> None:
        with self._lock:
            self.traces.append(trace)

    def by_id(self, trace_id: str) -> Trace | None:
        return next((t for t in self.traces if t.trace_id == trace_id), None)

    def spans_of_kind(self, kind: str) -> list[Span]:
        return [s for t in self.traces for s in t.spans if s.kind == kind]


class StoreExporter:
    """Writes completed traces to the SQLite telemetry store."""

    def __init__(self, store: Any) -> None:
        self._store = store

    def export(self, trace: Trace) -> None:
        self._store.write_trace(trace.to_dict(), [s.to_dict() for s in trace.spans])


class Tracer:
    """Creates traces and spans and hands finished traces to an exporter.

    Safe to share across threads: trace and span context lives in
    :mod:`contextvars`, and a worker thread starts with its own context, so
    concurrent tool calls never interleave their spans.
    """

    def __init__(
        self,
        exporter: Exporter | None = None,
        service: str = "gantry",
        environment: str = "local",
        enabled: bool = True,
        price_book: PriceBook | None = None,
    ) -> None:
        self.exporter = exporter
        self.service = service
        self.environment = environment
        self.enabled = enabled
        self.price_book = price_book or PRICE_BOOK

    @staticmethod
    def current_trace() -> Trace | None:
        return _current_trace.get()

    @staticmethod
    def current_span() -> Span | None:
        return _current_span.get()

    @staticmethod
    def current_trace_id() -> str | None:
        trace = _current_trace.get()
        return trace.trace_id if trace else None

    @contextmanager
    def trace(self, name: str, **attributes: Any) -> Iterator[Trace]:
        """Start a root trace. A nested call joins the existing trace instead."""
        existing = _current_trace.get()
        if existing is not None:
            with self.span(name, kind=sc.SpanKind.CHAIN, **attributes):
                yield existing
            return

        trace = Trace(
            trace_id=new_trace_id(),
            name=name,
            started_at=time.time(),
            service=self.service,
            environment=self.environment,
            attributes=dict(attributes),
        )
        token = _current_trace.set(trace)
        try:
            with self.span(name, kind=sc.SpanKind.AGENT):
                yield trace
        except BaseException:
            trace.status = "error"
            raise
        else:
            if trace.status == "unset":
                trace.status = "ok"
        finally:
            trace.ended_at = time.time()
            _current_trace.reset(token)
            if self.enabled and self.exporter is not None:
                self.exporter.export(trace)

    @contextmanager
    def span(self, name: str, kind: str = sc.SpanKind.CHAIN, **attributes: Any) -> Iterator[Span]:
        trace = _current_trace.get()
        if trace is None or not self.enabled:
            # Un-traced call sites still receive a Span, so instrumentation code
            # never needs a None check and never changes behaviour when tracing
            # is switched off.
            yield Span(
                span_id=new_span_id(),
                trace_id="",
                name=name,
                kind=kind,
                started_at=time.time(),
                attributes=dict(attributes),
            )
            return

        parent = _current_span.get()
        span = Span(
            span_id=new_span_id(),
            trace_id=trace.trace_id,
            parent_span_id=parent.span_id if parent else None,
            name=name,
            kind=kind,
            started_at=time.time(),
            attributes=dict(attributes),
        )
        trace.add_span(span)
        token = _current_span.set(span)
        try:
            yield span
        except BaseException as exc:
            span.record_exception(exc)
            trace.status = "error"
            raise
        else:
            if span.status == "unset":
                span.status = "ok"
        finally:
            span.ended_at = time.time()
            _current_span.reset(token)

    @contextmanager
    def llm_span(
        self,
        model: str,
        operation: str = sc.Operation.CHAT,
        system: str = "azure.openai",
        deployment: str | None = None,
        **attributes: Any,
    ) -> Iterator[Span]:
        """A GenAI span named ``{operation} {model}`` per the OTel convention."""
        attrs: dict[str, Any] = {
            sc.GEN_AI_SYSTEM: system,
            sc.GEN_AI_OPERATION_NAME: operation,
            sc.GEN_AI_REQUEST_MODEL: model,
            **attributes,
        }
        if deployment:
            attrs[sc.GEN_AI_AZURE_DEPLOYMENT] = deployment
        with self.span(sc.llm_span_name(operation, model), kind=sc.SpanKind.LLM, **attrs) as span:
            yield span

    @contextmanager
    def tool_span(self, tool: str, **attributes: Any) -> Iterator[Span]:
        attrs = {sc.GEN_AI_TOOL_NAME: tool, **attributes}
        with self.span(sc.tool_span_name(tool), kind=sc.SpanKind.TOOL, **attrs) as span:
            yield span


#: Process-wide tracer. Disabled until :func:`configure` runs, so importing
#: Gantry never creates files as a side effect.
_default = Tracer(exporter=None, enabled=False)


def configure(
    exporter: Exporter | None = None,
    service: str = "gantry",
    environment: str = "local",
    enabled: bool = True,
) -> Tracer:
    global _default
    _default = Tracer(exporter=exporter, service=service, environment=environment, enabled=enabled)
    return _default


def get_tracer() -> Tracer:
    return _default
