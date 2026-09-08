"""Prometheus metrics, in the text exposition format, with no client library.

Traces answer "what happened in this run". Metrics answer "what is happening
across all runs" - the tool failure rate that crept up, the p95 that doubled,
the spend that tripled. Both are needed, and they are cheap to produce together
because the harness already sits at every interesting boundary.

The exposition format is a documented, stable text format, so implementing it
directly avoids a dependency for perhaps eighty lines of code. Everything here
is safe to call from any thread.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Any

#: Seconds. Tuned for agent workloads, where a tool call is milliseconds and a
#: provider call is seconds.
DURATION_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)
#: US dollars, per agent run.
COST_BUCKETS = (0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _render_labels(names: tuple[str, ...], values: tuple[str, ...], extra: str = "") -> str:
    parts = [f'{n}="{_escape_label(str(v))}"' for n, v in zip(names, values, strict=True)]
    if extra:
        parts.append(extra)
    return "{" + ",".join(parts) + "}" if parts else ""


@dataclass
class _Metric:
    name: str
    help: str
    labelnames: tuple[str, ...] = ()
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        """Hook for subclasses to build their storage.

        Declared here even though it does nothing: ``@dataclass`` decides
        whether to emit a ``__post_init__`` call when it processes *this*
        class, so a hook defined only on a subclass is generated but never
        invoked. That failure is silent until the first attribute access.
        """

    def _key(self, labels: dict[str, str]) -> tuple[str, ...]:
        missing = set(self.labelnames) - set(labels)
        if missing:
            raise ValueError(f"metric {self.name}: missing labels {sorted(missing)}")
        return tuple(str(labels[name]) for name in self.labelnames)


class Counter(_Metric):
    """A monotonically increasing value."""

    def __post_init__(self) -> None:
        self._values: dict[tuple[str, ...], float] = {}

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        if amount < 0:
            raise ValueError("counters cannot decrease")
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        with self._lock:
            items = sorted(self._values.items())
        lines.extend(
            f"{self.name}{_render_labels(self.labelnames, key)} {value:g}" for key, value in items
        )
        return lines


class Gauge(_Metric):
    """A value that can go up or down."""

    def __post_init__(self) -> None:
        self._values: dict[tuple[str, ...], float] = {}

    def set(self, value: float, **labels: str) -> None:
        key = self._key(labels)
        with self._lock:
            self._values[key] = value

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} gauge"]
        with self._lock:
            items = sorted(self._values.items())
        lines.extend(
            f"{self.name}{_render_labels(self.labelnames, key)} {value:g}" for key, value in items
        )
        return lines


class Histogram(_Metric):
    """Cumulative buckets plus sum and count, as Prometheus expects."""

    buckets: tuple[float, ...] = DURATION_BUCKETS

    def __init__(
        self,
        name: str,
        help: str,
        labelnames: tuple[str, ...] = (),
        buckets: tuple[float, ...] = DURATION_BUCKETS,
    ) -> None:
        super().__init__(name=name, help=help, labelnames=labelnames)
        self.buckets = tuple(sorted(buckets))
        self._counts: dict[tuple[str, ...], list[int]] = {}
        self._sums: dict[tuple[str, ...], float] = {}
        self._totals: dict[tuple[str, ...], int] = {}

    def observe(self, value: float, **labels: str) -> None:
        key = self._key(labels)
        with self._lock:
            counts = self._counts.setdefault(key, [0] * len(self.buckets))
            for i, bound in enumerate(self.buckets):
                if value <= bound:
                    counts[i] += 1
            self._sums[key] = self._sums.get(key, 0.0) + value
            self._totals[key] = self._totals.get(key, 0) + 1

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        with self._lock:
            keys = sorted(self._counts)
            snapshot = {k: (list(self._counts[k]), self._sums[k], self._totals[k]) for k in keys}
        for key, (counts, total_sum, total_count) in snapshot.items():
            # observe() already increments every bucket whose bound the value
            # falls under, so these counts are cumulative as Prometheus requires.
            for bound, count in zip(self.buckets, counts, strict=True):
                labels = _render_labels(self.labelnames, key, f'le="{bound:g}"')
                lines.append(f"{self.name}_bucket{labels} {count}")
            inf_labels = _render_labels(self.labelnames, key, 'le="+Inf"')
            base_labels = _render_labels(self.labelnames, key)
            lines.append(f"{self.name}_bucket{inf_labels} {total_count}")
            lines.append(f"{self.name}_sum{base_labels} {total_sum:g}")
            lines.append(f"{self.name}_count{base_labels} {total_count}")
        return lines


class MetricsRegistry:
    """Holds metrics and renders the exposition text."""

    def __init__(self) -> None:
        self._metrics: dict[str, Any] = {}
        self._lock = threading.Lock()

    def register(self, metric: Any) -> Any:
        with self._lock:
            if metric.name in self._metrics:
                raise ValueError(f"metric {metric.name} is already registered")
            self._metrics[metric.name] = metric
        return metric

    def counter(self, name: str, help: str, labelnames: tuple[str, ...] = ()) -> Counter:
        return self.register(Counter(name=name, help=help, labelnames=labelnames))

    def gauge(self, name: str, help: str, labelnames: tuple[str, ...] = ()) -> Gauge:
        return self.register(Gauge(name=name, help=help, labelnames=labelnames))

    def histogram(
        self,
        name: str,
        help: str,
        labelnames: tuple[str, ...] = (),
        buckets: tuple[float, ...] = DURATION_BUCKETS,
    ) -> Histogram:
        return self.register(Histogram(name, help, labelnames, buckets))

    def render(self) -> str:
        with self._lock:
            metrics = [self._metrics[name] for name in sorted(self._metrics)]
        lines: list[str] = []
        for metric in metrics:
            lines.extend(metric.render())
        return "\n".join(lines) + "\n"


REGISTRY = MetricsRegistry()

# --- the harness's standard metrics ---------------------------------------
AGENT_RUNS = REGISTRY.counter(
    "gantry_agent_runs_total", "Agent runs by terminal stop reason.", ("stop_reason",)
)
AGENT_RUN_DURATION = REGISTRY.histogram(
    "gantry_agent_run_duration_seconds", "Wall-clock duration of an agent run.", ("stop_reason",)
)
AGENT_RUN_COST = REGISTRY.histogram(
    "gantry_agent_run_cost_usd",
    "Provider spend per agent run.",
    ("stop_reason",),
    buckets=COST_BUCKETS,
)
AGENT_STEPS = REGISTRY.histogram(
    "gantry_agent_steps",
    "Planning steps taken per agent run.",
    ("stop_reason",),
    buckets=(1, 2, 3, 5, 8, 12, 16, 24, 32, 48),
)

TOOL_CALLS = REGISTRY.counter(
    "gantry_tool_calls_total", "Tool invocations by tool and outcome.", ("tool", "outcome")
)
TOOL_DURATION = REGISTRY.histogram(
    "gantry_tool_duration_seconds", "Tool execution time.", ("tool",)
)

PROVIDER_CALLS = REGISTRY.counter(
    "gantry_provider_calls_total", "Provider requests by model and outcome.", ("model", "outcome")
)
PROVIDER_DURATION = REGISTRY.histogram(
    "gantry_provider_duration_seconds", "Provider request latency.", ("model",)
)
PROVIDER_TOKENS = REGISTRY.counter(
    "gantry_provider_tokens_total", "Tokens consumed.", ("model", "direction")
)
PROVIDER_COST = REGISTRY.counter(
    "gantry_provider_cost_usd_total", "Cumulative provider spend.", ("model",)
)
PROVIDER_RETRIES = REGISTRY.counter(
    "gantry_provider_retries_total", "Provider retries by reason.", ("model", "reason")
)

SANDBOX_DECISIONS = REGISTRY.counter(
    "gantry_sandbox_decisions_total",
    "Sandbox admission decisions. 'denied' means an action was blocked.",
    ("decision", "rule"),
)


def observe_tool_call(tool: str, duration_s: float, ok: bool) -> None:
    TOOL_CALLS.inc(tool=tool, outcome="ok" if ok else "error")
    TOOL_DURATION.observe(duration_s, tool=tool)


def observe_provider_call(
    model: str,
    duration_s: float,
    ok: bool,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: float = 0.0,
) -> None:
    PROVIDER_CALLS.inc(model=model, outcome="ok" if ok else "error")
    PROVIDER_DURATION.observe(duration_s, model=model)
    if input_tokens:
        PROVIDER_TOKENS.inc(input_tokens, model=model, direction="input")
    if output_tokens:
        PROVIDER_TOKENS.inc(output_tokens, model=model, direction="output")
    if cost_usd and math.isfinite(cost_usd):
        PROVIDER_COST.inc(cost_usd, model=model)


def observe_agent_run(stop_reason: str, duration_s: float, cost_usd: float, steps: int) -> None:
    AGENT_RUNS.inc(stop_reason=stop_reason)
    AGENT_RUN_DURATION.observe(duration_s, stop_reason=stop_reason)
    AGENT_RUN_COST.observe(cost_usd, stop_reason=stop_reason)
    AGENT_STEPS.observe(steps, stop_reason=stop_reason)
