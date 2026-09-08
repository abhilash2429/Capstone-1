"""OpenTelemetry GenAI semantic-convention keys.

Instrumentation is only worth writing if something else can read it. Rather
than inventing attribute names, Gantry emits the OpenTelemetry GenAI semantic
conventions, so the same spans can be shipped to Tempo, Honeycomb, Jaeger or
Azure Monitor without a translation layer.

Concepts OpenTelemetry has not standardised - cost, sandbox decisions, loop
budget - sit under a clearly separate ``gantry.*`` namespace, so adopting a
future upstream convention is a rename rather than a redesign.
"""

from __future__ import annotations

# --- GenAI ----------------------------------------------------------------
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
GEN_AI_RESPONSE_ID = "gen_ai.response.id"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"
GEN_AI_TOOL_DESCRIPTION = "gen_ai.tool.description"

# Azure-specific, mirrored by the common instrumentation libraries.
GEN_AI_AZURE_DEPLOYMENT = "gen_ai.azure.deployment"
GEN_AI_USAGE_CACHED_INPUT_TOKENS = "gen_ai.usage.cached_input_tokens"

# --- Gantry extensions -----------------------------------------------------
COST_USD = "gantry.cost.usd"
COST_PRICED = "gantry.cost.priced"
COST_VERIFIED = "gantry.cost.verified"

RUN_ID = "gantry.run.id"
RUN_TASK = "gantry.run.task"
RUN_STEP = "gantry.run.step"
RUN_PHASE = "gantry.run.phase"
RUN_STOP_REASON = "gantry.run.stop_reason"

TOOL_VERSION = "gantry.tool.version"
TOOL_CAPABILITIES = "gantry.tool.capabilities"
TOOL_OUTCOME = "gantry.tool.outcome"
TOOL_TRUNCATED = "gantry.tool.truncated"
TOOL_ACTION_FINGERPRINT = "gantry.tool.action_fingerprint"

SANDBOX_DECISION = "gantry.sandbox.decision"
SANDBOX_MODE = "gantry.sandbox.mode"
SANDBOX_EXIT_CODE = "gantry.sandbox.exit_code"

BUDGET_REMAINING_STEPS = "gantry.budget.remaining_steps"
BUDGET_REMAINING_USD = "gantry.budget.remaining_usd"

RETRY_COUNT = "gantry.retry.count"
CACHE_HIT = "gantry.cache.hit"

EVAL_RUN_ID = "gantry.eval.run_id"
EVAL_CASE_ID = "gantry.eval.case_id"


class Operation:
    """Values for :data:`GEN_AI_OPERATION_NAME`."""

    CHAT = "chat"
    EMBEDDINGS = "embeddings"
    EXECUTE_TOOL = "execute_tool"


class SpanKind:
    """Gantry's coarse span taxonomy, used for filtering and cost rollups."""

    AGENT = "agent"
    STEP = "step"
    LLM = "llm"
    TOOL = "tool"
    SANDBOX = "sandbox"
    VERIFY = "verify"
    CHAIN = "chain"


def llm_span_name(operation: str, model: str) -> str:
    """OpenTelemetry names GenAI spans ``{operation} {model}``."""
    return f"{operation} {model}"


def tool_span_name(tool: str) -> str:
    return f"{Operation.EXECUTE_TOOL} {tool}"
