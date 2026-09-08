# Gantry

An agent harness: the runtime a coding agent runs *on*.

Most agent projects are a prompt wrapped around an API call. Gantry is the layer
underneath that — the loop contract, tool registry, transport, dispatcher,
sandbox, verification gates, eval harness and telemetry that decide whether an
agent is reliable or merely impressive in a demo.

**Status:** in development. Design and build plan below.

## Subsystems

| # | Subsystem | What it does |
|---|-----------|--------------|
| 1 | Loop contract | The typed state machine every agent turn passes through |
| 2 | Tool registry | JSON Schema validation, versioning, capability scoping |
| 3 | Transport | JSON-RPC 2.0 over newline-delimited stdio |
| 4 | Dispatcher | Routes model tool calls to handlers, parallel execution, typed errors |
| 5 | Plan-execute | Plan/act/observe control flow with replanning |
| 6 | Verification gates | Post-conditions an action must satisfy before it counts as done |
| 7 | Observation budget | Bounded context growth under long tool-use loops |
| 8 | Sandbox | Path jail, command denylist, resource limits |
| 9 | Eval harness | Fixture tasks, deterministic scoring, regression gate |
| 10 | Observability | OpenTelemetry GenAI spans, Prometheus metrics, cost accounting |

## Model access

Built against **Azure OpenAI**. A deterministic offline provider ships alongside
it so the test suite and CI run with no credentials and no spend.

## License

MIT
