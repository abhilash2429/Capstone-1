-- Gantry telemetry schema.
--
-- Three record families share one database and join on ids:
--   traces / spans  - what the harness did, span by span
--   agent_runs      - one row per agent run, with its terminal state
--   eval_* (M6)     - what the evaluation harness measured
-- Sharing a database means an eval failure opens as a full execution trace
-- without a second storage system to stand up.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS traces (
    trace_id      TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    started_at    REAL NOT NULL,
    ended_at      REAL,
    duration_ms   REAL,
    status        TEXT NOT NULL DEFAULT 'unset',
    service       TEXT NOT NULL DEFAULT 'gantry',
    environment   TEXT NOT NULL DEFAULT 'local',
    cost_usd      REAL NOT NULL DEFAULT 0.0,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    span_count    INTEGER NOT NULL DEFAULT 0,
    attributes    TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_traces_started ON traces (started_at DESC);
CREATE INDEX IF NOT EXISTS idx_traces_name    ON traces (name, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_traces_status  ON traces (status, started_at DESC);

CREATE TABLE IF NOT EXISTS spans (
    span_id        TEXT PRIMARY KEY,
    trace_id       TEXT NOT NULL,
    parent_span_id TEXT,
    name           TEXT NOT NULL,
    kind           TEXT NOT NULL,
    started_at     REAL NOT NULL,
    ended_at       REAL,
    duration_ms    REAL,
    status         TEXT NOT NULL DEFAULT 'unset',
    status_message TEXT,
    cost_usd       REAL NOT NULL DEFAULT 0.0,
    attributes     TEXT NOT NULL DEFAULT '{}',
    events         TEXT NOT NULL DEFAULT '[]',
    FOREIGN KEY (trace_id) REFERENCES traces (trace_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_spans_trace  ON spans (trace_id, started_at);
CREATE INDEX IF NOT EXISTS idx_spans_parent ON spans (parent_span_id);
CREATE INDEX IF NOT EXISTS idx_spans_kind   ON spans (kind, started_at DESC);

CREATE TABLE IF NOT EXISTS agent_runs (
    run_id        TEXT PRIMARY KEY,
    trace_id      TEXT,
    task          TEXT NOT NULL DEFAULT '',
    started_at    REAL NOT NULL,
    ended_at      REAL,
    duration_ms   REAL,
    stop_reason   TEXT,
    stop_detail   TEXT NOT NULL DEFAULT '',
    steps         INTEGER NOT NULL DEFAULT 0,
    tool_calls    INTEGER NOT NULL DEFAULT 0,
    tool_errors   INTEGER NOT NULL DEFAULT 0,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd      REAL NOT NULL DEFAULT 0.0,
    provider      TEXT NOT NULL DEFAULT '',
    model         TEXT NOT NULL DEFAULT '',
    config        TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_runs_started ON agent_runs (started_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_stop    ON agent_runs (stop_reason, started_at DESC);
