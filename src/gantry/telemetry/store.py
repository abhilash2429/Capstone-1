"""SQLite-backed telemetry storage.

SQLite is a deliberate choice, not a placeholder. The point of this harness is
that it runs on a laptop and inside CI with no infrastructure while still
producing durable, queryable telemetry. WAL mode gives concurrent readers
alongside the single writer the exporter uses, which is exactly the access
pattern here.

Connections are thread-local because ``sqlite3`` connections cannot be shared
across threads and the dispatcher fans tool calls out over a pool.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _row(row: sqlite3.Row, json_fields: Sequence[str] = ()) -> dict[str, Any]:
    out = dict(row)
    for field in json_fields:
        if field in out:
            out[field] = _loads(out[field], [] if field == "events" else {})
    return out


class TelemetryStore:
    """Thread-safe handle onto a Gantry telemetry database."""

    def __init__(self, path: str | Path = ".gantry/gantry.db") -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.Lock()
        with self._write_lock:
            self.conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))

    @property
    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(
                str(self.path), timeout=30.0, isolation_level=None, check_same_thread=False
            )
            conn.row_factory = sqlite3.Row
            for pragma in (
                "journal_mode = WAL",
                "synchronous = NORMAL",
                "foreign_keys = ON",
                "busy_timeout = 30000",
            ):
                conn.execute(f"PRAGMA {pragma}")
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def __enter__(self) -> TelemetryStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- writes ---------------------------------------------------------
    def write_trace(self, trace: dict[str, Any], spans: Iterable[dict[str, Any]]) -> None:
        """Persist a finished trace and its spans in one transaction.

        Whole traces rather than individual spans: a half-written trace is worse
        than none when you are debugging, and batching keeps the writer off the
        agent's hot path.
        """
        span_rows = [
            (
                s["span_id"],
                s["trace_id"],
                s.get("parent_span_id"),
                s["name"],
                s["kind"],
                s["started_at"],
                s.get("ended_at"),
                s.get("duration_ms"),
                s.get("status", "unset"),
                s.get("status_message"),
                float(s.get("cost_usd", 0.0)),
                json.dumps(s.get("attributes", {}), default=str),
                json.dumps(s.get("events", []), default=str),
            )
            for s in spans
        ]
        with self._write_lock:
            conn = self.conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    """
                    INSERT INTO traces (trace_id, name, started_at, ended_at, duration_ms, status,
                                        service, environment, cost_usd, input_tokens,
                                        output_tokens, span_count, attributes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (trace_id) DO UPDATE SET
                        ended_at = excluded.ended_at, duration_ms = excluded.duration_ms,
                        status = excluded.status, cost_usd = excluded.cost_usd,
                        input_tokens = excluded.input_tokens, output_tokens = excluded.output_tokens,
                        span_count = excluded.span_count, attributes = excluded.attributes
                    """,
                    (
                        trace["trace_id"],
                        trace["name"],
                        trace["started_at"],
                        trace.get("ended_at"),
                        trace.get("duration_ms"),
                        trace.get("status", "unset"),
                        trace.get("service", "gantry"),
                        trace.get("environment", "local"),
                        float(trace.get("cost_usd", 0.0)),
                        int(trace.get("input_tokens", 0)),
                        int(trace.get("output_tokens", 0)),
                        int(trace.get("span_count", len(span_rows))),
                        json.dumps(trace.get("attributes", {}), default=str),
                    ),
                )
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO spans
                        (span_id, trace_id, parent_span_id, name, kind, started_at, ended_at,
                         duration_ms, status, status_message, cost_usd, attributes, events)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    span_rows,
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def write_run(self, run: dict[str, Any]) -> None:
        with self._write_lock:
            self.conn.execute(
                """
                INSERT INTO agent_runs (run_id, trace_id, task, started_at, ended_at, duration_ms,
                                        stop_reason, stop_detail, steps, tool_calls, tool_errors,
                                        input_tokens, output_tokens, cost_usd, provider, model, config)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (run_id) DO UPDATE SET
                    ended_at = excluded.ended_at, duration_ms = excluded.duration_ms,
                    stop_reason = excluded.stop_reason, stop_detail = excluded.stop_detail,
                    steps = excluded.steps, tool_calls = excluded.tool_calls,
                    tool_errors = excluded.tool_errors, input_tokens = excluded.input_tokens,
                    output_tokens = excluded.output_tokens, cost_usd = excluded.cost_usd
                """,
                (
                    run["run_id"],
                    run.get("trace_id"),
                    run.get("task", ""),
                    run["started_at"],
                    run.get("ended_at"),
                    run.get("duration_ms"),
                    run.get("stop_reason"),
                    run.get("stop_detail", ""),
                    int(run.get("steps", 0)),
                    int(run.get("tool_calls", 0)),
                    int(run.get("tool_errors", 0)),
                    int(run.get("input_tokens", 0)),
                    int(run.get("output_tokens", 0)),
                    float(run.get("cost_usd", 0.0)),
                    run.get("provider", ""),
                    run.get("model", ""),
                    json.dumps(run.get("config", {}), default=str),
                ),
            )

    # -- reads ----------------------------------------------------------
    def list_traces(
        self, limit: int = 50, offset: int = 0, name: str | None = None, status: str | None = None
    ) -> list[dict[str, Any]]:
        where, params = [], []
        if name:
            where.append("name = ?")
            params.append(name)
        if status:
            where.append("status = ?")
            params.append(status)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        # The interpolated fragment is assembled from the literals above; every
        # caller-supplied value travels as a bound parameter.
        rows = self.conn.execute(
            f"SELECT * FROM traces {clause} ORDER BY started_at DESC LIMIT ? OFFSET ?",  # noqa: S608
            (*params, limit, offset),
        ).fetchall()
        return [_row(r, ("attributes",)) for r in rows]

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM traces WHERE trace_id = ?", (trace_id,)).fetchone()
        return _row(row, ("attributes",)) if row else None

    def get_spans(self, trace_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM spans WHERE trace_id = ? ORDER BY started_at ASC", (trace_id,)
        ).fetchall()
        return [_row(r, ("attributes", "events")) for r in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM agent_runs WHERE run_id = ?", (run_id,)).fetchone()
        return _row(row, ("config",)) if row else None

    def list_runs(self, limit: int = 50, stop_reason: str | None = None) -> list[dict[str, Any]]:
        if stop_reason:
            rows = self.conn.execute(
                "SELECT * FROM agent_runs WHERE stop_reason = ? ORDER BY started_at DESC LIMIT ?",
                (stop_reason, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM agent_runs ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_row(r, ("config",)) for r in rows]

    # -- aggregates -----------------------------------------------------
    def latency_percentiles(self, name: str | None = None) -> dict[str, float]:
        clause = "WHERE duration_ms IS NOT NULL"
        params: list[Any] = []
        if name:
            clause += " AND name = ?"
            params.append(name)
        values = [
            r[0]
            for r in self.conn.execute(
                # Same as list_traces: fixed fragments, bound values.
                f"SELECT duration_ms FROM traces {clause} ORDER BY duration_ms ASC",  # noqa: S608
                params,
            )
        ]
        if not values:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "n": 0}

        def pct(p: float) -> float:
            return round(values[min(len(values) - 1, max(0, round(p * (len(values) - 1))))], 2)

        return {"p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99), "n": len(values)}

    def cost_by_model(self) -> list[dict[str, Any]]:
        """Spend grouped by the model recorded on each LLM span."""
        rows = self.conn.execute(
            """
            SELECT json_extract(attributes, '$."gen_ai.request.model"') AS model,
                   COUNT(*) AS calls,
                   SUM(cost_usd) AS cost_usd,
                   SUM(COALESCE(json_extract(attributes, '$."gen_ai.usage.input_tokens"'), 0)) AS input_tokens,
                   SUM(COALESCE(json_extract(attributes, '$."gen_ai.usage.output_tokens"'), 0)) AS output_tokens
              FROM spans
             WHERE kind = 'llm'
             GROUP BY model
             ORDER BY cost_usd DESC
            """
        ).fetchall()
        return [dict(r) for r in rows if r["model"]]

    def stop_reason_breakdown(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT stop_reason, COUNT(*) AS runs, AVG(cost_usd) AS avg_cost_usd,
                   AVG(steps) AS avg_steps
              FROM agent_runs WHERE stop_reason IS NOT NULL
             GROUP BY stop_reason ORDER BY runs DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]
