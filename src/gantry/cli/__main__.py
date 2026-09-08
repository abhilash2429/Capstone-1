"""``gantry`` - drive the harness from a terminal.

Four subcommands, each mapping to something a reviewer would want to check
without reading the source:

* ``gantry run`` runs an agent against a workspace and prints what happened.
* ``gantry tools`` shows the tools a given capability grant exposes.
* ``gantry doctor`` reports what is configured and what is missing.
* ``gantry trace`` reads a recorded run back out of the telemetry store.

The default provider is the offline one, so ``gantry run`` works on a fresh
clone with no credentials. That is the point: an agent harness whose demo
requires a paid key is a harness nobody evaluates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from gantry.config import Config
from gantry.contract import LoopContract, StopReason
from gantry.providers.base import Provider
from gantry.providers.offline import OfflineProvider, Script, Turn
from gantry.telemetry import tracer as tracing
from gantry.telemetry.store import TelemetryStore
from gantry.telemetry.tracer import StoreExporter
from gantry.toolkit import build_coding_agent, build_toolkit
from gantry.tools import Grant
from gantry.verify import GateSet, PythonSyntaxGate, gates_from_spec

GRANTS = {
    "read-only": Grant.read_only,
    "developer": Grant.developer,
    "unrestricted": Grant.unrestricted,
}

EXIT_OK = 0
EXIT_INCOMPLETE = 1
EXIT_USAGE = 2
EXIT_ERROR = 3


# --- provider selection ---------------------------------------------------
def build_provider(name: str, config: Config) -> Provider:
    """Resolve a provider by name, failing with a usable message.

    The offline provider is scripted to stop immediately. It exists so that
    every other part of the CLI - tool registration, the sandbox, the gates,
    the trace - can be exercised without a network call, and a run that used
    it says so in its output rather than looking like a real one.
    """
    if name == "offline":
        return OfflineProvider(
            script=Script(
                turns=[Turn(text="Offline provider: no model is configured for this run.")]
            )
        )
    if name == "azure":
        from gantry.providers.azure import AzureOpenAIProvider

        if not config.azure.configured:
            raise SystemExit(
                "Azure is not configured. Set AZURE_OPENAI_ENDPOINT, "
                "AZURE_OPENAI_API_KEY and GANTRY_DEPLOYMENT_MAIN in .env, "
                "then re-run `gantry doctor` to check."
            )
        return AzureOpenAIProvider(config.azure)
    raise SystemExit(f"unknown provider {name!r}; choose from: offline, azure")


def build_gates(specs: list[str], syntax: bool) -> GateSet:
    """Turn ``--gate name=command`` arguments into a gate set."""
    described: list[dict[str, Any]] = []
    for raw in specs:
        name, _, command = raw.partition("=")
        if not command:
            name, command = "check", raw
        described.append({"type": "command", "name": name, "command": command})
    gates = gates_from_spec(described)
    if syntax:
        gates.gates.insert(0, PythonSyntaxGate())
    return gates


# --- commands -------------------------------------------------------------
def cmd_run(args: argparse.Namespace) -> int:
    config = Config.from_env()
    config = _with_overrides(config, args)
    store = TelemetryStore(args.db) if args.db else None
    tracer = tracing.configure(
        exporter=StoreExporter(store) if store else None, environment=config.environment
    )

    agent = build_coding_agent(
        build_provider(args.provider, config),
        args.workspace,
        config=config,
        gates=build_gates(args.gate, syntax=not args.no_syntax_gate),
        grant=GRANTS[args.grant](),
        contract=LoopContract(config.budget),
        tracer=tracer,
        store=store,
        shell=args.grant != "read-only",
    )
    result = agent.run(args.task)

    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
    else:
        _print_result(result)
    if store is not None:
        store.close()
    return EXIT_OK if result.stop_reason is StopReason.COMPLETED else EXIT_INCOMPLETE


def cmd_tools(args: argparse.Namespace) -> int:
    toolkit = build_toolkit(args.workspace, config=Config.from_env())
    grant = GRANTS[args.grant]()
    if args.json:
        print(json.dumps(toolkit.registry.describe(grant), indent=2))
        return EXIT_OK
    print(f"Tools available under the '{args.grant}' grant:\n")
    for spec in toolkit.registry.list():
        allowed = grant.permits(spec)
        marker = " " if allowed else "x"
        caps = ",".join(sorted(str(c) for c in spec.capabilities)) or "-"
        print(f" [{marker}] {spec.name:<12} {caps:<32} timeout {spec.timeout_s:g}s")
        if not allowed:
            missing = ",".join(sorted(str(c) for c in grant.missing_for(spec)))
            print(f"      refused: needs {missing}")
    print("\n[x] marks a tool the grant refuses; it is never described to the model.")
    return EXIT_OK


def cmd_doctor(args: argparse.Namespace) -> int:
    config = Config.from_env()
    checks: list[tuple[str, bool, str]] = []

    checks.append(("python", sys.version_info >= (3, 11), sys.version.split()[0]))
    checks.append(
        (
            "azure endpoint",
            bool(config.azure.endpoint),
            config.azure.endpoint or "unset (AZURE_OPENAI_ENDPOINT)",
        )
    )
    key_set = bool(_env(config.azure.api_key_env))
    # The value is never printed, only whether the variable is populated.
    checks.append(
        ("azure api key", key_set, f"{config.azure.api_key_env} is {'set' if key_set else 'unset'}")
    )
    for role, deployment in config.azure.deployments.items():
        checks.append((f"deployment '{role}'", bool(deployment), deployment or "unset"))
    try:
        import openai  # noqa: F401

        checks.append(("openai sdk", True, "installed"))
    except ImportError:
        checks.append(("openai sdk", False, "not installed (pip install '.[azure]')"))

    from gantry.telemetry.pricing import PriceBook

    book = PriceBook.load()
    unverified = book.unverified_models()
    checks.append(
        (
            "price book",
            True,
            f"loaded, last verified {book.meta.get('last_verified', 'unknown')}"
            + (f"; provisional: {', '.join(unverified)}" if unverified else ""),
        )
    )

    width = max(len(name) for name, _, _ in checks)
    for name, ok, detail in checks:
        print(f"  {'ok  ' if ok else 'MISS'}  {name:<{width}}  {detail}")
    missing = [name for name, ok, _ in checks if not ok]
    if missing:
        print(f"\n{len(missing)} item(s) unset. Offline runs work regardless.")
    return EXIT_OK


def cmd_trace(args: argparse.Namespace) -> int:
    store = TelemetryStore(args.db)
    try:
        runs = store.list_runs(limit=args.limit)
        if args.json:
            print(json.dumps(runs, indent=2))
            return EXIT_OK
        if not runs:
            print(f"No runs recorded in {args.db}.")
            return EXIT_OK
        print(f"{'run':<12} {'stop reason':<22} {'steps':>5} {'cost':>9}  task")
        for run in runs:
            print(
                f"{run['run_id'][:10]:<12} {run['stop_reason']:<22} "
                f"{run['steps']:>5} {run['cost_usd']:>9.4f}  {run['task'][:48]}"
            )
        return EXIT_OK
    finally:
        store.close()


# --- helpers --------------------------------------------------------------
def _env(name: str) -> str:
    import os

    return os.environ.get(name, "")


def _with_overrides(config: Config, args: argparse.Namespace) -> Config:
    from dataclasses import replace

    budget = config.budget
    if args.max_steps:
        budget = replace(budget, max_steps=args.max_steps)
    if args.max_cost:
        budget = replace(budget, max_cost_usd=args.max_cost)
    sandbox = replace(config.sandbox, root=str(args.workspace))
    if args.container:
        sandbox = replace(sandbox, use_container=True)
    return replace(config, budget=budget, sandbox=sandbox)


def _print_result(result: Any) -> None:
    print(f"\nrun      {result.run_id}")
    print(f"outcome  {result.stop_reason}")
    if result.detail:
        print(f"detail   {result.detail}")
    if result.summary:
        print(f"summary  {result.summary}")
    usage = result.usage or {}
    print(
        f"usage    {usage.get('steps', 0)} steps, {usage.get('tool_calls', 0)} tool calls, "
        f"{usage.get('input_tokens', 0)}+{usage.get('output_tokens', 0)} tokens, "
        f"${usage.get('cost_usd', 0.0):.4f}, {result.duration_ms / 1000:.1f}s"
    )
    if result.gate_report is not None:
        for check in result.gate_report.verifications:
            state = "pass" if check.passed else ("skip" if check.inconclusive else "FAIL")
            print(f"gate     {state}  {check.name}")


# --- entry point ----------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gantry", description="An agent harness: loop, tools, sandbox, telemetry."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    run = subcommands.add_parser("run", help="run an agent against a workspace")
    run.add_argument("task", help="what the agent should do")
    run.add_argument("-w", "--workspace", type=Path, default=Path.cwd())
    run.add_argument("-p", "--provider", default="offline", choices=["offline", "azure"])
    run.add_argument("-g", "--grant", default="developer", choices=sorted(GRANTS))
    run.add_argument(
        "--gate",
        action="append",
        default=[],
        metavar="NAME=COMMAND",
        help="a command that must pass before completion is accepted; repeatable",
    )
    run.add_argument("--no-syntax-gate", action="store_true")
    run.add_argument("--max-steps", type=int, default=0)
    run.add_argument("--max-cost", type=float, default=0.0)
    run.add_argument("--container", action="store_true", help="run commands in a container")
    run.add_argument("--db", type=Path, default=None, help="record the run to this SQLite file")
    run.add_argument("--json", action="store_true")
    run.set_defaults(func=cmd_run)

    tools = subcommands.add_parser("tools", help="list the tools a grant exposes")
    tools.add_argument("-w", "--workspace", type=Path, default=Path.cwd())
    tools.add_argument("-g", "--grant", default="developer", choices=sorted(GRANTS))
    tools.add_argument("--json", action="store_true")
    tools.set_defaults(func=cmd_tools)

    doctor = subcommands.add_parser("doctor", help="report what is configured")
    doctor.set_defaults(func=cmd_doctor)

    trace = subcommands.add_parser("trace", help="list recorded runs")
    trace.add_argument("--db", type=Path, default=Path(".gantry/gantry.db"))
    trace.add_argument("-n", "--limit", type=int, default=20)
    trace.add_argument("--json", action="store_true")
    trace.set_defaults(func=cmd_trace)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover - exercised through the console script
    raise SystemExit(main())
