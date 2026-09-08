"""Serialise Gantry traces into OTLP/JSON.

The escape hatch that keeps the local-first design honest: the same spans that
power the built-in dashboard can be shipped to any OTLP endpoint, so adopting
this harness in a project that already runs Tempo, Honeycomb or Azure Monitor
does not mean throwing the instrumentation away.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from gantry.telemetry.tracer import Span, Trace

log = logging.getLogger("gantry.telemetry")

_STATUS_CODE = {"unset": 0, "ok": 1, "error": 2}
# OTel SpanKind: 1=INTERNAL, 3=CLIENT. Calls that leave the process are CLIENT.
_SPAN_KIND = {"llm": 3, "sandbox": 3}


def _any_value(value: Any) -> dict[str, Any]:
    # bool before int: bool is a subclass of int and would otherwise be
    # serialised as an integer, which loses the type on the wire.
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, (list, tuple, set, frozenset)):
        return {"arrayValue": {"values": [_any_value(v) for v in value]}}
    if isinstance(value, dict):
        return {"stringValue": json.dumps(value, default=str)}
    return {"stringValue": str(value)}


def _attributes(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"key": k, "value": _any_value(v)} for k, v in attrs.items()]


def _nanos(seconds: float | None) -> str:
    return str(int((seconds or 0.0) * 1_000_000_000))


def span_to_otlp(span: Span) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "traceId": span.trace_id,
        "spanId": span.span_id,
        "name": span.name,
        "kind": _SPAN_KIND.get(span.kind, 1),
        "startTimeUnixNano": _nanos(span.started_at),
        "endTimeUnixNano": _nanos(span.ended_at),
        "attributes": _attributes({"gantry.span.kind": span.kind, **span.attributes}),
        "status": {"code": _STATUS_CODE.get(span.status, 0)},
    }
    if span.parent_span_id:
        payload["parentSpanId"] = span.parent_span_id
    if span.status_message:
        payload["status"]["message"] = span.status_message
    if span.events:
        payload["events"] = [
            {
                "timeUnixNano": _nanos(e.get("time")),
                "name": e.get("name", ""),
                "attributes": _attributes(e.get("attributes", {})),
            }
            for e in span.events
        ]
    return payload


def trace_to_otlp(trace: Trace, scope_version: str = "0.1.0") -> dict[str, Any]:
    """An OTLP/JSON ``ExportTraceServiceRequest`` body for one trace."""
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": _attributes(
                        {
                            "service.name": trace.service,
                            "deployment.environment.name": trace.environment,
                        }
                    )
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "gantry", "version": scope_version},
                        "spans": [span_to_otlp(s) for s in trace.spans],
                    }
                ],
            }
        ]
    }


class OtlpFileExporter:
    """Appends each trace to a newline-delimited OTLP/JSON file.

    A file rather than an HTTP client on purpose: shipping the payload is the
    collector's job, and a file can be tailed by the OTel Collector's filelog
    receiver without the harness taking on a network dependency it would then
    have to make reliable.
    """

    def __init__(self, path: str) -> None:
        self.path = path

    def export(self, trace: Trace) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(trace_to_otlp(trace), default=str) + "\n")


class FanOutExporter:
    """Sends every trace to several exporters. One failing never blocks another."""

    def __init__(self, *exporters: Any) -> None:
        self.exporters = list(exporters)

    def export(self, trace: Trace) -> None:
        for exporter in self.exporters:
            try:
                exporter.export(trace)
            except Exception:
                # Telemetry must never break the agent it is observing, so any
                # exporter failure is contained. It is logged rather than
                # swallowed, because a silently dropped trace is how you find
                # out weeks later that an exporter has been broken all along.
                log.warning(
                    "exporter %s failed to export trace %s",
                    type(exporter).__name__,
                    trace.trace_id,
                    exc_info=True,
                )
