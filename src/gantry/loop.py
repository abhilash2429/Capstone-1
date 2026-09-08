"""The agent loop.

Everything the rest of this package builds is here to make this file boring,
which is the goal. The loop plans, acts, observes and verifies, and it holds no
opinions about *when to stop* - that belongs to the contract - or about *what
is allowed* - that belongs to the registry and the sandbox. What is left is
sequencing, and sequencing is testable.

Three decisions worth naming.

**Completion is a tool call, not a sentence.** The agent finishes by calling
``finish`` with a status, so the harness never has to guess whether a paragraph
of prose meant "done", "I refuse" or "I am stuck". Sniffing text for completion
is how a run ends at the wrong moment. A bare text reply is still handled - it
is treated as an implicit completion claim - but the explicit path is the one
the prompt asks for.

**A completion claim is checked before it is believed.** Claiming completion
runs the gates. If they fail, the failure goes back as the tool result and the
loop continues, so the agent gets the compiler error rather than a verdict.

**Refusal is a successful outcome.** An agent that declines an ambiguous or
inappropriate task and says why has behaved correctly. Recording that as a
distinct stop reason rather than a failure is what lets an eval suite contain
tasks whose right answer is "no".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gantry.budget import BudgetCounters, ObservationBudget
from gantry.config import Config
from gantry.contract import LoopContract, Phase, RunState, StopReason
from gantry.dispatch import Dispatcher
from gantry.errors import ProviderError
from gantry.messages import Completion, Message, ToolCall
from gantry.providers.base import CompletionRequest, Provider
from gantry.sandbox import PathJail, SandboxRunner
from gantry.telemetry import metrics
from gantry.telemetry import semconv as sc
from gantry.telemetry.tracer import Tracer, get_tracer
from gantry.tools import Grant, ToolContext, ToolRegistry, ToolResult, ToolSpec
from gantry.verify import GateContext, GateReport, GateSet

FINISH_TOOL_NAME = "finish"

FINISH_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["completed", "refused", "blocked"],
            "description": (
                "completed: the task is done and you believe the checks will pass. "
                "refused: you decline the task and have said why. "
                "blocked: you cannot proceed and need something you do not have."
            ),
        },
        "summary": {
            "type": "string",
            "description": "What you did, or why you are not doing it. Two or three sentences.",
        },
    },
    "required": ["status", "summary"],
    "additionalProperties": False,
}

DEFAULT_SYSTEM_PROMPT = """\
You are a software engineering agent working inside a sandboxed workspace.

How this works:
- Use the tools to inspect and change files. You cannot see the workspace any
  other way, so read before you edit.
- Every command runs confined to the workspace. Paths outside it, network
  access and privileged commands are refused, and being refused is information,
  not an obstacle to route around.
- When you believe the task is done, call `finish` with status "completed".
  Your work is then checked. If a check fails you will be told which one and
  why, and you should fix the cause.
- If the task is ambiguous, inappropriate, or asks for something you should not
  do, call `finish` with status "refused" and explain. Declining is a correct
  outcome, not a failure.
- If you are blocked on something you genuinely cannot obtain, call `finish`
  with status "blocked" and say what is missing.

Repeating a failed command unchanged will not make it succeed. If something
fails twice, change your approach.\
"""


@dataclass
class AgentResult:
    """Everything one run produced."""

    run_id: str
    task: str
    stop_reason: StopReason
    detail: str = ""
    summary: str = ""
    trace_id: str | None = None
    messages: list[Message] = field(default_factory=list)
    gate_report: GateReport | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    budget: BudgetCounters | None = None

    @property
    def succeeded(self) -> bool:
        return self.stop_reason is StopReason.COMPLETED

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task": self.task,
            "stop_reason": str(self.stop_reason),
            "detail": self.detail,
            "summary": self.summary,
            "trace_id": self.trace_id,
            "succeeded": self.succeeded,
            "duration_ms": round(self.duration_ms, 2),
            "usage": self.usage,
            "gates": self.gate_report.as_dict() if self.gate_report else None,
            "turns": len(self.messages),
        }


class Agent:
    """Runs a task to a terminal state."""

    def __init__(
        self,
        provider: Provider,
        registry: ToolRegistry,
        jail: PathJail,
        contract: LoopContract | None = None,
        runner: SandboxRunner | None = None,
        gates: GateSet | None = None,
        budget: ObservationBudget | None = None,
        grant: Grant | None = None,
        dispatcher: Dispatcher | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        tracer: Tracer | None = None,
        store: Any = None,
        config: Config | None = None,
        max_gate_retries: int = 3,
        max_output_tokens: int = 4096,
        temperature: float | None = 0.0,
    ) -> None:
        self.provider = provider
        self.jail = jail
        self.runner = runner
        self.gates = gates or GateSet()
        self.budget = budget or ObservationBudget()
        self.grant = grant or Grant.developer()
        self.contract = contract or LoopContract((config or Config()).budget)
        self.tracer = tracer or get_tracer()
        self.store = store
        self.system_prompt = system_prompt
        self.max_gate_retries = max_gate_retries
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

        # The finish tool is registered here rather than expected from the
        # caller: a loop whose termination signal is optional is a loop that
        # sometimes has no termination signal.
        # Collaborators trace into the agent's tracer. Without this, a runner
        # or gate set constructed with the default tracer emits its spans
        # somewhere else, and a run's trace is silently missing the sandbox and
        # verification steps depending on construction order - which is the
        # kind of gap you only notice when you need the trace.
        if self.runner is not None:
            self.runner.tracer = self.tracer
        self.gates.tracer = self.tracer

        self.registry = registry
        self._register_finish_tool()
        self.dispatcher = dispatcher or Dispatcher(
            registry=self.registry, grant=self.grant, contract=self.contract, tracer=self.tracer
        )

    def _register_finish_tool(self) -> None:
        if FINISH_TOOL_NAME in self.registry:
            return
        self.registry.register(
            ToolSpec(
                name=FINISH_TOOL_NAME,
                version="1.0.0",
                description=(
                    "Declare that you are done. Use status 'completed' when the task is "
                    "finished, 'refused' when you decline it, 'blocked' when you cannot "
                    "proceed. Your work is checked before a completion is accepted."
                ),
                input_schema=FINISH_SCHEMA,
                # Intercepted by the loop before dispatch; this handler exists
                # so the tool is a first-class registry entry rather than a
                # name the loop treats specially by convention.
                handler=lambda args, ctx: ToolResult.success(args.get("summary", "")),
                capabilities=frozenset(),
                concurrency_safe=False,
            )
        )

    # -- prompt ----------------------------------------------------------
    def build_system_prompt(self) -> str:
        """The base prompt plus the acceptance criteria.

        Telling the agent up front what will be checked is worth several turns.
        Without it the first completion claim is a guess, and the gate report
        is the first time it learns what "done" meant.
        """
        parts = [self.system_prompt]
        if len(self.gates):
            lines = ["", "Your work will be checked against:"]
            lines += [f"- {g['name']}: {g['description']}" for g in self.gates.describe()]
            parts.append("\n".join(lines))
        parts.append(
            f"\nThe workspace root is {self.jail.root.name}/ and every path is relative to it."
        )
        return "\n".join(parts)

    # -- the loop --------------------------------------------------------
    def run(self, task: str, run_id: str | None = None) -> AgentResult:
        started = time.monotonic()
        state = self.contract.start_run(task=task, run_id=run_id)
        messages: list[Message] = [
            Message.system(self.build_system_prompt()),
            Message.user(task),
        ]
        counters = BudgetCounters()
        gate_report: GateReport | None = None
        gate_attempts = 0
        summary = ""

        with self.tracer.trace(
            "agent.run", **{sc.RUN_ID: state.run_id, sc.RUN_TASK: task[:500]}
        ) as trace:
            state.trace_id = trace.trace_id
            tool_context = ToolContext(
                run_id=state.run_id,
                trace_id=trace.trace_id,
                workspace=self.jail.root,
                cancellation=self.contract.cancellation,
            )

            while True:
                decision = self.contract.before_step(state)
                if not decision.allowed:
                    self.contract.finish(state, decision.stop_reason, decision.detail)
                    break

                self.contract.record_step(state)
                state.phase = Phase.PLAN

                fitted, elision = self.budget.fit(messages, trace.trace_id)
                counters.record(elision)
                if elision.changed:
                    messages = fitted
                    self.tracer.current_span().set_attributes(elision.as_attributes())

                try:
                    completion = self._complete(fitted, state)
                except ProviderError as exc:
                    self.contract.finish(state, StopReason.PROVIDER_ERROR, exc.message)
                    break

                messages.append(completion.message)

                if not completion.wants_tools:
                    # A bare reply is an implicit completion claim.
                    gate_attempts += 1
                    summary = completion.text
                    gate_report = self._verify(tool_context)
                    if gate_report.passed:
                        self.contract.finish(state, StopReason.COMPLETED, "verification passed")
                        break
                    if gate_attempts > self.max_gate_retries:
                        self.contract.finish(
                            state,
                            StopReason.VERIFICATION_FAILED,
                            f"checks still failing after {gate_attempts} attempts: "
                            + ", ".join(v.name for v in gate_report.failures),
                        )
                        break
                    messages.append(Message.user(gate_report.feedback()))
                    continue

                state.phase = Phase.ACT
                finish_call, tool_calls = self._split_finish(completion.message.tool_calls)

                if tool_calls:
                    outcome = self.dispatcher.dispatch(list(tool_calls), state, tool_context)
                    messages.extend(
                        Message.tool(r.call.id, r.result.content) for r in outcome.results
                    )
                    if outcome.should_stop:
                        self.contract.finish(state, outcome.stop.stop_reason, outcome.stop.detail)
                        break

                if finish_call is None:
                    state.phase = Phase.OBSERVE
                    continue

                # -- a completion claim ---------------------------------
                state.phase = Phase.VERIFY
                arguments = self._finish_arguments(finish_call)
                status = arguments.get("status", "completed")
                summary = arguments.get("summary", "")

                if status in ("refused", "blocked"):
                    messages.append(Message.tool(finish_call.id, "Acknowledged."))
                    self.contract.finish(state, StopReason.REFUSED, summary)
                    break

                gate_attempts += 1
                gate_report = self._verify(tool_context)
                if gate_report.passed:
                    messages.append(
                        Message.tool(finish_call.id, "All checks passed. The task is complete.")
                    )
                    self.contract.finish(state, StopReason.COMPLETED, "verification passed")
                    break

                messages.append(Message.tool(finish_call.id, gate_report.feedback()))
                if gate_attempts > self.max_gate_retries:
                    self.contract.finish(
                        state,
                        StopReason.VERIFICATION_FAILED,
                        f"checks still failing after {gate_attempts} attempts: "
                        + ", ".join(v.name for v in gate_report.failures),
                    )
                    break

            trace.attributes[sc.RUN_STOP_REASON] = str(state.stop_reason)
            trace.attributes[sc.RUN_STEP] = state.usage.steps
            if state.stop_reason is not StopReason.COMPLETED:
                trace.status = "error" if not _is_clean_stop(state.stop_reason) else trace.status

        duration_ms = (time.monotonic() - started) * 1000.0
        result = AgentResult(
            run_id=state.run_id,
            task=task,
            stop_reason=state.stop_reason or StopReason.INTERNAL_ERROR,
            detail=state.stop_detail,
            summary=summary,
            trace_id=state.trace_id,
            messages=messages,
            gate_report=gate_report,
            usage=state.usage.as_dict(),
            duration_ms=duration_ms,
            budget=counters,
        )
        self._record(result, state)
        return result

    # -- steps -----------------------------------------------------------
    def _complete(self, messages: list[Message], state: RunState) -> Completion:
        request = CompletionRequest.of(
            messages,
            tools=self.registry.to_openai_tools(self.grant),
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            metadata={"run_id": state.run_id},
        )
        completion = self.provider.complete(request)
        self.contract.record_provider_call(
            state,
            input_tokens=completion.usage.input_tokens,
            output_tokens=completion.usage.output_tokens,
            cost_usd=_span_cost(),
        )
        return completion

    @staticmethod
    def _split_finish(
        calls: tuple[ToolCall, ...],
    ) -> tuple[ToolCall | None, tuple[ToolCall, ...]]:
        """Separate a completion claim from real work in the same turn.

        A model will happily call a tool and declare victory in one turn. The
        work runs first, so the gates judge the finished state rather than the
        state before the last edit landed.
        """
        finish = next((c for c in calls if c.name == FINISH_TOOL_NAME), None)
        return finish, tuple(c for c in calls if c.name != FINISH_TOOL_NAME)

    @staticmethod
    def _finish_arguments(call: ToolCall) -> dict[str, Any]:
        try:
            return call.parse_arguments()
        except Exception:  # noqa: BLE001 - a malformed claim is still a claim
            return {"status": "completed", "summary": ""}

    def _verify(self, context: ToolContext) -> GateReport:
        return self.gates.run(
            GateContext(jail=self.jail, runner=self.runner, task=context.metadata.get("task", ""))
        )

    def _record(self, result: AgentResult, state: RunState) -> None:
        metrics.observe_agent_run(
            stop_reason=str(result.stop_reason),
            duration_s=result.duration_ms / 1000.0,
            cost_usd=state.usage.cost_usd,
            steps=state.usage.steps,
        )
        if self.store is None:
            return
        self.store.write_run(
            {
                "run_id": result.run_id,
                "trace_id": result.trace_id,
                "task": result.task,
                "started_at": time.time() - result.duration_ms / 1000.0,
                "ended_at": time.time(),
                "duration_ms": result.duration_ms,
                "stop_reason": str(result.stop_reason),
                "stop_detail": result.detail,
                "steps": state.usage.steps,
                "tool_calls": state.usage.tool_calls,
                "tool_errors": state.usage.tool_errors,
                "input_tokens": state.usage.input_tokens,
                "output_tokens": state.usage.output_tokens,
                "cost_usd": state.usage.cost_usd,
                "provider": self.provider.name,
                "model": self.provider.model_for("main"),
                "config": {"gates": self.gates.describe()},
            }
        )


def _is_clean_stop(reason: StopReason | None) -> bool:
    """Refusing and running out of budget are outcomes, not errors."""
    return reason in (StopReason.COMPLETED, StopReason.REFUSED) or (
        reason is not None and reason.is_budget
    )


def _span_cost() -> float:
    """The cost the provider just recorded on its span.

    Read back from the span rather than recomputed, so the loop's running total
    and the trace can never disagree about what a run cost.
    """
    span = get_tracer().current_span()
    return float(span.attributes.get(sc.COST_USD, 0.0)) if span else 0.0


def build_workspace_agent(
    provider: Provider,
    workspace: str | Path,
    registry: ToolRegistry,
    gates: GateSet | None = None,
    config: Config | None = None,
    **kwargs: Any,
) -> Agent:
    """Assemble an agent over a workspace with the sandbox wired in."""
    config = config or Config()
    jail = PathJail(workspace)
    runner = SandboxRunner(jail, config=config.sandbox, tracer=kwargs.get("tracer"))
    return Agent(
        provider=provider,
        registry=registry,
        jail=jail,
        runner=runner,
        gates=gates,
        config=config,
        **kwargs,
    )
