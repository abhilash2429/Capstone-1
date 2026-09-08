"""Telemetry has to be correct in three ways: the spans must nest the way the
code nested, the cost must be arithmetically right, and the output must be
readable by the tools it claims to speak to (OTLP, Prometheus)."""

from __future__ import annotations

import json
import sqlite3

import pytest

from gantry.telemetry import (
    FanOutExporter,
    MemoryExporter,
    PriceBook,
    SpanKind,
    StoreExporter,
    Tracer,
    metrics,
    trace_to_otlp,
)
from gantry.telemetry.metrics import Counter, Histogram, MetricsRegistry
from gantry.telemetry.pricing import PRICE_BOOK


# --- pricing ---------------------------------------------------------------
def test_cached_input_is_discounted_not_double_counted():
    """Cached tokens are a subset of input tokens, billed at the cached rate."""
    cost = PRICE_BOOK.price(
        "gpt-4o", input_tokens=100_000, output_tokens=20_000, cached_input_tokens=80_000
    )
    expected = (20_000 * 2.50 + 80_000 * 1.25 + 20_000 * 10.00) / 1_000_000
    assert cost.usd == pytest.approx(expected)
    assert cost.priced and cost.verified


def test_unknown_model_is_unpriced_rather_than_free():
    """A silent $0.00 is how cost tracking stops working the day someone adds
    a deployment. An admitted unknown is recoverable."""
    cost = PRICE_BOOK.price("some-model-shipped-tomorrow", 1000, 1000)
    assert cost.usd == 0.0
    assert cost.priced is False


def test_azure_deployment_names_fall_back_to_the_longest_known_prefix():
    assert PRICE_BOOK.price("gpt-4o-mini-prod-eastus", 1_000_000, 0).usd == pytest.approx(0.15)


def test_provisional_prices_are_flagged():
    assert PRICE_BOOK.price("gpt-4.1", 1000, 0).verified is False
    assert "gpt-4.1" in PRICE_BOOK.unverified_models()


def test_pricing_overrides_replace_only_named_models(tmp_path):
    override = tmp_path / "prices.json"
    override.write_text(json.dumps({"models": {"gpt-4o": {"input": 1.0, "output": 1.0}}}))
    book = PriceBook._read(override)
    assert book.price("gpt-4o", 1_000_000, 0).usd == pytest.approx(1.0)


# --- tracing ---------------------------------------------------------------
def test_spans_nest_the_way_the_code_nested(tracer, exporter):
    # Deliberately three separate `with` blocks: the nesting is the assertion.
    with tracer.trace("agent.run"):  # noqa: SIM117
        with tracer.span("step", kind=SpanKind.STEP):
            with tracer.tool_span("read_file"):
                pass
    trace = exporter.traces[0]
    by_name = {s.name: s for s in trace.spans}
    root = by_name["agent.run"]
    step = by_name["step"]
    tool = by_name["execute_tool read_file"]
    assert root.parent_span_id is None
    assert step.parent_span_id == root.span_id
    assert tool.parent_span_id == step.span_id


def test_trace_rolls_up_cost_and_tokens_from_its_spans(tracer, exporter):
    with tracer.trace("agent.run"):
        with tracer.llm_span(model="gpt-4o") as span:
            span.record_usage("gpt-4o", input_tokens=1000, output_tokens=100)
        with tracer.llm_span(model="gpt-4o-mini") as span:
            span.record_usage("gpt-4o-mini", input_tokens=2000, output_tokens=200)
    trace = exporter.traces[0]
    assert trace.input_tokens == 3000
    assert trace.output_tokens == 300
    assert trace.cost_usd == pytest.approx(
        (1000 * 2.50 + 100 * 10.0 + 2000 * 0.15 + 200 * 0.60) / 1_000_000
    )


def test_an_exception_marks_the_span_and_the_trace(tracer, exporter):
    with pytest.raises(RuntimeError), tracer.trace("agent.run"), tracer.tool_span("bash"):
        raise RuntimeError("command failed")
    trace = exporter.traces[0]
    assert trace.status == "error"
    tool = next(s for s in trace.spans if s.kind == SpanKind.TOOL)
    assert tool.status == "error"
    event = tool.events[0]
    assert event["name"] == "exception"
    assert event["attributes"]["exception.type"] == "RuntimeError"


def test_a_nested_trace_call_joins_the_existing_trace(tracer, exporter):
    """Instrumented functions call each other. The inner one must not start a
    second trace and orphan half the spans."""
    with tracer.trace("outer"), tracer.trace("inner"):
        pass
    assert len(exporter.traces) == 1
    assert {s.name for s in exporter.traces[0].spans} == {"outer", "inner"}


def test_spans_still_work_with_tracing_disabled():
    """Instrumentation code must not need a None check or change behaviour."""
    tracer = Tracer(exporter=None, enabled=False)
    with tracer.trace("agent.run"), tracer.llm_span(model="gpt-4o") as span:
        span.record_usage("gpt-4o", input_tokens=10, output_tokens=1)
        assert span.cost_usd > 0


def test_concurrent_runs_do_not_interleave_their_spans():
    """Each thread starts with its own contextvar context, which is what makes
    the dispatcher's thread pool safe to trace."""
    from concurrent.futures import ThreadPoolExecutor

    exporter = MemoryExporter()
    tracer = Tracer(exporter=exporter)

    def run(i: int) -> None:
        with tracer.trace(f"run-{i}"):
            for j in range(5):
                with tracer.tool_span(f"tool-{i}-{j}"):
                    pass

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(run, range(8)))

    assert len(exporter.traces) == 8
    for trace in exporter.traces:
        index = trace.name.split("-")[1]
        tool_spans = [s for s in trace.spans if s.kind == SpanKind.TOOL]
        assert len(tool_spans) == 5
        assert all(s.name.split()[1].split("-")[1] == index for s in tool_spans)


def test_fanout_survives_a_failing_exporter():
    """Telemetry must never break the agent it is observing."""

    class Broken:
        def export(self, trace):
            raise OSError("disk full")

    good = MemoryExporter()
    tracer = Tracer(exporter=FanOutExporter(Broken(), good))
    with tracer.trace("agent.run"):
        pass
    assert len(good.traces) == 1


# --- OTLP ------------------------------------------------------------------
def test_otlp_payload_shape(tracer, exporter):
    with tracer.trace("agent.run"), tracer.llm_span(model="gpt-4o") as span:
        span.record_usage("gpt-4o", input_tokens=10, output_tokens=2)
    payload = trace_to_otlp(exporter.traces[0])
    scope = payload["resourceSpans"][0]["scopeSpans"][0]
    span = next(s for s in scope["spans"] if s["name"] == "chat gpt-4o")
    assert span["kind"] == 3  # CLIENT: the call leaves the process
    assert len(span["traceId"]) == 32 and len(span["spanId"]) == 16
    assert int(span["endTimeUnixNano"]) >= int(span["startTimeUnixNano"])
    attrs = {a["key"]: a["value"] for a in span["attributes"]}
    assert attrs["gen_ai.usage.input_tokens"] == {"intValue": "10"}
    # bool must not be serialised as int, despite being an int subclass
    assert attrs["gantry.cost.priced"] == {"boolValue": True}
    assert json.dumps(payload)  # must be JSON-serialisable end to end


# --- metrics ---------------------------------------------------------------
def test_exposition_format_is_well_formed():
    import re

    registry = MetricsRegistry()
    counter = registry.counter("t_calls_total", "Calls.", ("tool", "outcome"))
    hist = registry.histogram("t_seconds", "Duration.", ("tool",), buckets=(0.1, 1.0))
    counter.inc(tool="read", outcome="ok")
    counter.inc(2, tool="read", outcome="ok")
    hist.observe(0.05, tool="read")
    hist.observe(2.0, tool="read")
    text = registry.render()

    assert 't_calls_total{tool="read",outcome="ok"} 3' in text
    # Buckets are cumulative: le=1.0 includes the 0.05 observation.
    assert 't_seconds_bucket{tool="read",le="0.1"} 1' in text
    assert 't_seconds_bucket{tool="read",le="1"} 1' in text
    assert 't_seconds_bucket{tool="read",le="+Inf"} 2' in text
    assert 't_seconds_count{tool="read"} 2' in text
    pattern = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*(\{[^}]*\})? -?[\d.eE+]+$")
    assert all(pattern.match(line) for line in text.splitlines() if not line.startswith("#"))


def test_label_values_are_escaped():
    registry = MetricsRegistry()
    counter = registry.counter("t_total", "T.", ("cmd",))
    counter.inc(cmd='rm -rf "/tmp"\nx')
    assert 'cmd="rm -rf \\"/tmp\\"\\nx"' in registry.render()


def test_missing_labels_fail_loudly():
    counter = Counter(name="t_total", help="T.", labelnames=("a", "b"))
    with pytest.raises(ValueError, match="missing labels"):
        counter.inc(a="1")


def test_counters_cannot_decrease():
    counter = Counter(name="t_total", help="T.")
    with pytest.raises(ValueError):
        counter.inc(-1)


def test_metric_storage_is_initialised():
    """Regression: @dataclass only wires __post_init__ if the decorated class
    defines it, so a hook declared only on a subclass is never called."""
    Counter(name="c_total", help="C.").inc()
    Histogram("h_seconds", "H.").observe(1.0)


def test_duplicate_metric_registration_is_refused():
    registry = MetricsRegistry()
    registry.counter("t_total", "T.")
    with pytest.raises(ValueError, match="already registered"):
        registry.counter("t_total", "T.")


def test_standard_metrics_cover_the_harness_boundaries():
    text = metrics.REGISTRY.render()
    for name in (
        "gantry_agent_runs_total",
        "gantry_tool_calls_total",
        "gantry_provider_cost_usd_total",
        "gantry_sandbox_decisions_total",
    ):
        assert f"# TYPE {name}" in text


# --- store -----------------------------------------------------------------
def test_traces_and_spans_round_trip(store):
    tracer = Tracer(exporter=StoreExporter(store))
    with (
        tracer.trace("agent.run", **{"gantry.run.task": "fix a test"}) as trace,
        tracer.llm_span(model="gpt-4o") as span,
    ):
        span.record_usage("gpt-4o", input_tokens=1000, output_tokens=100)
    stored = store.get_trace(trace.trace_id)
    assert stored["name"] == "agent.run"
    assert stored["attributes"]["gantry.run.task"] == "fix a test"
    assert stored["cost_usd"] == pytest.approx(trace.cost_usd)
    assert len(store.get_spans(trace.trace_id)) == 2


def test_cost_by_model_aggregates_llm_spans(store):
    tracer = Tracer(exporter=StoreExporter(store))
    for _ in range(3):
        with tracer.trace("agent.run"), tracer.llm_span(model="gpt-4o") as span:
            span.record_usage("gpt-4o", input_tokens=1000, output_tokens=100)
    rows = store.cost_by_model()
    assert rows[0]["model"] == "gpt-4o"
    assert rows[0]["calls"] == 3
    assert rows[0]["input_tokens"] == 3000


def test_latency_percentiles_on_an_empty_store(store):
    assert store.latency_percentiles() == {"p50": 0.0, "p95": 0.0, "p99": 0.0, "n": 0}


def test_agent_runs_round_trip_and_group_by_stop_reason(store):
    for i, reason in enumerate(["completed", "completed", "no_progress"]):
        store.write_run(
            {
                "run_id": f"r{i}",
                "task": "t",
                "started_at": 0.0,
                "ended_at": 1.0,
                "duration_ms": 1000.0,
                "stop_reason": reason,
                "steps": 3,
                "cost_usd": 0.01,
            }
        )
    assert store.get_run("r0")["stop_reason"] == "completed"
    breakdown = {row["stop_reason"]: row["runs"] for row in store.stop_reason_breakdown()}
    assert breakdown == {"completed": 2, "no_progress": 1}


def test_a_failed_span_write_rolls_back_the_trace(store):
    """A half-written trace is worse than none when you are debugging.

    The second span references a trace that does not exist, so the foreign key
    fires mid-transaction after the trace row and the first span are already
    inserted. Both must disappear.
    """
    with pytest.raises(sqlite3.IntegrityError):
        store.write_trace(
            {"trace_id": "t1", "name": "agent.run", "started_at": 0.0},
            [
                {"span_id": "s1", "trace_id": "t1", "name": "ok", "kind": "llm", "started_at": 0.0},
                {
                    "span_id": "s2",
                    "trace_id": "does-not-exist",
                    "name": "orphan",
                    "kind": "llm",
                    "started_at": 0.0,
                },
            ],
        )
    assert store.get_trace("t1") is None
    assert store.get_spans("t1") == []
