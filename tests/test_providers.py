"""Provider layer.

The Azure provider is exercised against fake SDK objects rather than the
network. That is not a compromise: the interesting logic is entirely in the
translation layer - which exceptions are retryable, which parameters a
deployment will reject, how usage maps to cost - and a live call tests none of
it while costing money and flaking.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from gantry.config import AzureConfig
from gantry.errors import (
    ConfigError,
    ProviderAuthError,
    ProviderBadRequest,
    ProviderError,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
    ToolValidationError,
)
from gantry.messages import Completion, FinishReason, Message, ToolCall, Usage
from gantry.providers import (
    AzureOpenAIProvider,
    CacheMiss,
    CacheMode,
    CachingProvider,
    CompletionRequest,
    FailingProvider,
    OfflineProvider,
    ResponseCache,
    RetryPolicy,
    Script,
    Turn,
)


@pytest.fixture
def request_() -> CompletionRequest:
    return CompletionRequest.of(
        [Message.system("You are an agent."), Message.user("Fix the failing test.")],
        tools=[{"type": "function", "function": {"name": "read_file"}}],
        temperature=0.0,
    )


# --- messages --------------------------------------------------------------
def test_an_assistant_turn_that_only_calls_tools_sends_null_content():
    """The API wants an explicit null here, not an empty string."""
    wire = Message.assistant(tool_calls=(ToolCall("c1", "read", '{"p":"a"}'),)).to_wire()
    assert wire["content"] is None
    assert wire["tool_calls"][0]["function"]["arguments"] == '{"p":"a"}'


def test_tool_results_carry_their_call_id():
    assert Message.tool("c1", "contents").to_wire() == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": "contents",
    }


def test_dict_arguments_are_serialised_for_the_wire():
    assert ToolCall("c1", "read", {"p": "a"}).to_wire()["function"]["arguments"] == '{"p":"a"}'


def test_malformed_arguments_are_recoverable_not_fatal():
    with pytest.raises(ToolValidationError, match="not valid JSON"):
        ToolCall("c1", "read", '{"p": ').parse_arguments()


def test_an_unrecognised_finish_reason_does_not_crash_a_run():
    assert FinishReason.parse("something_new_next_year") is FinishReason.UNKNOWN


def test_a_length_finish_is_reported_as_truncated():
    """A reply cut off by the output cap often carries a half-written tool
    call, so the loop needs to know."""
    assert Completion(Message.assistant("half"), FinishReason.LENGTH).truncated


# --- offline provider ------------------------------------------------------
def test_a_script_replays_in_order(tracer):
    provider = OfflineProvider(
        script=Script.calling("read_file", {"path": "a.py"}, then="Fixed."), tracer=tracer
    )
    request = CompletionRequest.of([Message.user("go")])
    first, second = provider.complete(request), provider.complete(request)
    assert first.tool_calls[0].name == "read_file"
    assert first.finish_reason is FinishReason.TOOL_CALLS
    assert second.text == "Fixed."
    assert second.finish_reason is FinishReason.STOP


def test_an_exhausted_script_terminates_rather_than_hanging(tracer):
    """A script that simply stopped would leave the loop waiting forever."""
    provider = OfflineProvider(script=Script.of("only turn"), tracer=tracer)
    request = CompletionRequest.of([Message.user("go")])
    provider.complete(request)
    last = provider.complete(request)
    assert "exhausted" in last.text
    assert last.finish_reason is FinishReason.STOP


def test_the_same_conversation_always_gives_the_same_answer(tracer):
    """Reproducibility is the point: an eval number that moves on its own is
    not a measurement."""
    request = CompletionRequest.of([Message.user("go")])
    outputs = []
    for _ in range(3):
        provider = OfflineProvider(script=Script.calling("read_file", {"p": "a"}), tracer=tracer)
        completion = provider.complete(request)
        outputs.append((completion.tool_calls[0].id, completion.text, completion.usage.as_dict()))
    assert len(set(map(str, outputs))) == 1


def test_a_policy_can_react_to_what_the_tools_returned(tracer):
    def policy(request, index):
        last = request.messages[-1]
        if last.role == "tool" and "PASS" in last.content:
            return Turn(text="Tests pass.")
        return Turn(calls=(("run_tests", {}),))

    provider = OfflineProvider(policy=policy, tracer=tracer)
    first = provider.complete(CompletionRequest.of([Message.user("check")]))
    assert first.tool_calls[0].name == "run_tests"
    second = provider.complete(
        CompletionRequest.of([Message.user("check"), Message.tool("c1", "PASS")])
    )
    assert second.text == "Tests pass."


def test_offline_runs_report_zero_cost_as_a_fact_not_a_gap(tracer, exporter):
    provider = OfflineProvider(script=Script.of("hello"), tracer=tracer)
    with tracer.trace("run"):
        provider.complete(CompletionRequest.of([Message.user("hi")]))
    span = exporter.traces[0].spans[-1]
    assert span.cost_usd == 0.0
    assert span.attributes["gantry.cost.priced"] is True


def test_recorded_traffic_can_be_replayed_from_jsonl(tmp_path, tracer):
    path = tmp_path / "recording.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"message": Message.assistant(text).to_wire(), "finish_reason": "stop"})
            for text in ("first", "second")
        )
    )
    provider = OfflineProvider.from_jsonl(path, tracer=tracer)
    request = CompletionRequest.of([Message.user("go")])
    assert [provider.complete(request).text for _ in range(2)] == ["first", "second"]


# --- instrumentation and retry ---------------------------------------------
def test_every_call_is_traced_priced_and_counted(tracer, exporter):
    provider = OfflineProvider(script=Script.of("done"), tracer=tracer)
    with tracer.trace("agent.run"):
        provider.complete(CompletionRequest.of([Message.user("hi")], temperature=0.2))
    span = next(s for s in exporter.traces[0].spans if s.kind == "llm")
    assert span.attributes["gen_ai.system"] == "offline"
    assert span.attributes["gen_ai.request.temperature"] == 0.2
    assert span.attributes["gen_ai.response.finish_reasons"] == ["stop"]
    assert "gen_ai.usage.input_tokens" in span.attributes


def test_a_retryable_failure_is_retried_and_each_attempt_is_recorded(tracer, exporter):
    """Invisible retries make a latency regression unattributable."""
    provider = FailingProvider(
        error=ProviderRateLimited("slow down"),
        fail_after=0,
        tracer=tracer,
        retry=RetryPolicy(max_attempts=3, base_delay_s=0),
        sleep=lambda _s: None,
    )
    with pytest.raises(ProviderRateLimited), tracer.trace("run"):
        provider.complete(CompletionRequest.of([Message.user("hi")]))
    assert provider.attempts == 3
    span = next(s for s in exporter.traces[0].spans if s.kind == "llm")
    assert [e["name"] for e in span.events].count("provider_retry") == 2
    assert span.status == "error"


def test_a_non_retryable_failure_is_not_retried(tracer):
    provider = FailingProvider(
        error=ProviderAuthError("bad key"), tracer=tracer, sleep=lambda _s: None
    )
    with pytest.raises(ProviderAuthError), tracer.trace("run"):
        provider.complete(CompletionRequest.of([Message.user("hi")]))
    assert provider.attempts == 1


def test_retry_after_takes_precedence_over_backoff():
    policy = RetryPolicy(base_delay_s=10, max_delay_s=30)
    assert policy.delay_for(attempt=5, retry_after_s=2.0) == 2.0
    # Full jitter, so several rate-limited runs do not retry in lockstep.
    assert 0 <= policy.delay_for(attempt=0) <= 10
    assert policy.delay_for(attempt=99) <= 30


# --- caching ---------------------------------------------------------------
def test_record_then_replay_reproduces_the_answer(tmp_path, tracer, request_):
    cache = ResponseCache(tmp_path)
    recorder = CachingProvider(
        OfflineProvider(script=Script.of("the recorded answer")),
        cache,
        mode=CacheMode.RECORD,
        tracer=tracer,
    )
    recorder.complete(request_)

    inner = OfflineProvider(script=Script.of("this must not be used"))
    player = CachingProvider(inner, ResponseCache(tmp_path), mode=CacheMode.REPLAY, tracer=tracer)
    replayed = player.complete(request_)
    assert replayed.text == "the recorded answer"
    assert replayed.cached is True
    assert inner.calls == []


def test_replay_mode_refuses_to_fall_through_to_the_provider(tmp_path, tracer, request_):
    """A silent fall-through would let CI spend money and report numbers that
    cannot be reproduced."""
    player = CachingProvider(
        OfflineProvider(script=Script.of("live")),
        ResponseCache(tmp_path),
        mode=CacheMode.REPLAY,
        tracer=tracer,
    )
    with pytest.raises(CacheMiss, match="replay mode forbids"):
        player.complete(request_)


def test_read_mode_falls_through_and_then_records(tmp_path, tracer, request_):
    cache = ResponseCache(tmp_path)
    provider = CachingProvider(
        OfflineProvider(script=Script.of("live answer")),
        cache,
        mode=CacheMode.READ,
        tracer=tracer,
    )
    assert provider.complete(request_).text == "live answer"
    assert cache.get(request_.cache_key(provider.model_for("main"))) is not None


def test_a_cache_hit_is_still_traced(tmp_path, tracer, exporter, request_):
    cache = ResponseCache(tmp_path)
    CachingProvider(
        OfflineProvider(script=Script.of("x")), cache, mode=CacheMode.RECORD, tracer=tracer
    ).complete(request_)
    player = CachingProvider(
        OfflineProvider(), ResponseCache(tmp_path), mode=CacheMode.REPLAY, tracer=tracer
    )
    with tracer.trace("run"):
        player.complete(request_)
    span = next(s for s in exporter.traces[-1].spans if s.kind == "llm")
    assert span.attributes["gantry.cache.hit"] is True


def test_the_cache_key_covers_what_changes_the_answer():
    base = CompletionRequest.of([Message.user("hi")], temperature=0.0)
    assert base.cache_key("m") == CompletionRequest.of(
        [Message.user("hi")], temperature=0.0
    ).cache_key("m")
    assert base.cache_key("m") != base.cache_key("other-model")
    assert base.cache_key("m") != CompletionRequest.of(
        [Message.user("hi")], temperature=0.7
    ).cache_key("m")
    assert base.cache_key("m") != CompletionRequest.of([Message.user("bye")]).cache_key("m")


def test_run_metadata_does_not_defeat_the_cache():
    """Run ids and timestamps change every time; keying on them would make
    every request a miss."""
    a = CompletionRequest.of([Message.user("hi")], metadata={"run_id": "r1"})
    b = CompletionRequest.of([Message.user("hi")], metadata={"run_id": "r2"})
    assert a.cache_key("m") == b.cache_key("m")


def test_an_interrupted_recording_leaves_no_broken_cassette(tmp_path):
    cache = ResponseCache(tmp_path)
    cache.put("abc123", Completion(Message.assistant("x"), FinishReason.STOP))
    assert not list(tmp_path.rglob("*.tmp"))
    assert cache.get("abc123").text == "x"


# --- azure: fake SDK objects -----------------------------------------------
def azure_response(content="ok", tool_calls=None, finish="stop", model="gpt-4o-2024-11-20"):
    message = SimpleNamespace(
        role="assistant", content=content, tool_calls=tool_calls, tool_call_id=None
    )
    return SimpleNamespace(
        id="chatcmpl-123",
        model=model,
        choices=[SimpleNamespace(message=message, finish_reason=finish)],
        usage=SimpleNamespace(
            prompt_tokens=1200,
            completion_tokens=340,
            prompt_tokens_details=SimpleNamespace(cached_tokens=800),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=64),
        ),
    )


class FakeCompletions:
    def __init__(self, responses=None, errors=None):
        self.responses = list(responses or [])
        self.errors = list(errors or [])
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.errors:
            error = self.errors.pop(0)
            if error is not None:
                raise error
        return self.responses.pop(0) if self.responses else azure_response()


class FakeClient:
    def __init__(self, completions):
        self.chat = SimpleNamespace(completions=completions)
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def azure_config() -> AzureConfig:
    return AzureConfig(
        endpoint="https://example.openai.azure.com",
        api_version="2024-10-21",
        deployments={"main": "gpt-4o-prod", "fast": "gpt-4o-mini-prod"},
    )


def test_azure_normalises_a_response(azure_config, tracer, request_):
    completions = FakeCompletions([azure_response("all done")])
    provider = AzureOpenAIProvider(azure_config, client=FakeClient(completions), tracer=tracer)
    completion = provider.complete(request_)
    assert completion.text == "all done"
    # The response reports the real model, the only way to learn what sits
    # behind a deployment name.
    assert completion.model == "gpt-4o-2024-11-20"
    assert completion.usage == Usage(1200, 340, cached_input_tokens=800, reasoning_tokens=64)


def test_azure_costs_a_call_using_the_cached_token_split(azure_config, tracer, exporter, request_):
    provider = AzureOpenAIProvider(
        azure_config, client=FakeClient(FakeCompletions()), tracer=tracer
    )
    with tracer.trace("run"):
        provider.complete(request_)
    span = next(s for s in exporter.traces[0].spans if s.kind == "llm")
    expected = (400 * 2.50 + 800 * 1.25 + 340 * 10.00) / 1_000_000
    assert span.cost_usd == pytest.approx(expected)


def test_azure_sends_tool_calls_back_in_normalised_form(azure_config, tracer, request_):
    raw_call = SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(name="read_file", arguments='{"path":"a.py"}'),
    )
    completions = FakeCompletions([azure_response(None, [raw_call], finish="tool_calls")])
    provider = AzureOpenAIProvider(azure_config, client=FakeClient(completions), tracer=tracer)
    completion = provider.complete(request_)
    assert completion.wants_tools
    assert completion.tool_calls[0].name == "read_file"
    assert completion.finish_reason is FinishReason.TOOL_CALLS


def test_azure_talks_in_roles_not_model_ids(azure_config, tracer):
    provider = AzureOpenAIProvider(
        azure_config, client=FakeClient(FakeCompletions()), tracer=tracer
    )
    assert provider.model_for("main") == "gpt-4o-prod"
    assert provider.model_for("fast") == "gpt-4o-mini-prod"
    # An unknown role falls back to main rather than failing a live run.
    assert provider.model_for("nonexistent") == "gpt-4o-prod"


def test_a_missing_deployment_is_a_config_error_not_a_runtime_surprise():
    provider = AzureOpenAIProvider(AzureConfig(endpoint="https://x", deployments={}))
    with pytest.raises(ConfigError, match="no Azure deployment"):
        provider.model_for("main")


def test_a_missing_endpoint_is_reported_before_any_call(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "sk-test")
    with pytest.raises(ConfigError, match="AZURE_OPENAI_ENDPOINT"):
        _ = AzureOpenAIProvider(AzureConfig(endpoint="")).client


def test_a_missing_key_is_reported_before_any_call(monkeypatch, azure_config):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="API_KEY"):
        _ = AzureOpenAIProvider(azure_config).client


def test_no_choices_is_an_error_rather_than_an_index_crash(azure_config, tracer, request_):
    empty = SimpleNamespace(id="x", model="m", choices=[], usage=None)
    provider = AzureOpenAIProvider(
        azure_config, client=FakeClient(FakeCompletions([empty])), tracer=tracer
    )
    with pytest.raises(ProviderError, match="no choices"):
        provider.complete(request_)


# --- azure: error translation ----------------------------------------------
def make_sdk_error(cls, message="boom", status=500, headers=None, body=None):
    import httpx2

    response = httpx2.Response(
        status, headers=headers or {}, request=httpx2.Request("POST", "https://example")
    )
    return cls(message, response=response, body=body)


@pytest.mark.parametrize(
    ("sdk_name", "status", "expected", "retryable"),
    [
        ("AuthenticationError", 401, ProviderAuthError, False),
        ("PermissionDeniedError", 403, ProviderAuthError, False),
        ("RateLimitError", 429, ProviderRateLimited, True),
        ("NotFoundError", 404, ProviderBadRequest, False),
        ("BadRequestError", 400, ProviderBadRequest, False),
        ("InternalServerError", 500, ProviderUnavailable, True),
    ],
)
def test_sdk_errors_map_to_typed_retry_aware_errors(
    azure_config, tracer, request_, sdk_name, status, expected, retryable
):
    """A single broad catch would lose the only distinction that matters:
    whether trying again could possibly help."""
    import openai

    error = make_sdk_error(getattr(openai, sdk_name), status=status)
    provider = AzureOpenAIProvider(
        azure_config,
        client=FakeClient(FakeCompletions(errors=[error] * 4)),
        tracer=tracer,
        retry=RetryPolicy(max_attempts=1),
        sleep=lambda _s: None,
    )
    with pytest.raises(expected) as exc:
        provider.complete(request_)
    assert exc.value.retryable is retryable


def test_a_timeout_is_retryable(azure_config, tracer, request_):
    import httpx2
    import openai

    provider = AzureOpenAIProvider(
        azure_config,
        client=FakeClient(
            FakeCompletions(errors=[openai.APITimeoutError(httpx2.Request("POST", "https://x"))])
        ),
        tracer=tracer,
        retry=RetryPolicy(max_attempts=1),
        sleep=lambda _s: None,
    )
    with pytest.raises(ProviderTimeout) as exc:
        provider.complete(request_)
    assert exc.value.retryable


def test_a_rate_limit_carries_retry_after_into_the_backoff(azure_config, tracer, request_):
    import openai

    error = make_sdk_error(openai.RateLimitError, status=429, headers={"retry-after": "7"})
    provider = AzureOpenAIProvider(
        azure_config,
        client=FakeClient(FakeCompletions(errors=[error])),
        tracer=tracer,
        retry=RetryPolicy(max_attempts=1),
        sleep=lambda _s: None,
    )
    with pytest.raises(ProviderRateLimited) as exc:
        provider.complete(request_)
    assert exc.value.details["retry_after_s"] == 7.0


# --- azure: parameter adaptation -------------------------------------------
def test_a_deployment_rejecting_temperature_is_retried_without_it(azure_config, tracer, request_):
    """The same request that works against one deployment is a 400 against
    another. The operator should not have to discover that in production."""
    import openai

    error = make_sdk_error(
        openai.BadRequestError,
        status=400,
        body={
            "error": {
                "message": "Unsupported value: 'temperature' is not supported with this model."
            }
        },
    )
    completions = FakeCompletions(errors=[error, None], responses=[azure_response("recovered")])
    provider = AzureOpenAIProvider(azure_config, client=FakeClient(completions), tracer=tracer)

    assert provider.complete(request_).text == "recovered"
    assert "temperature" in completions.calls[0]
    assert "temperature" not in completions.calls[1]
    # Remembered, so it costs one extra call per process rather than per turn.
    provider.complete(request_)
    assert "temperature" not in completions.calls[2]
    assert provider.describe()["adapted_away"]["gpt-4o-prod"] == ["temperature"]


def test_the_output_cap_parameter_swaps_in_either_direction(azure_config, tracer, request_):
    import openai

    error = make_sdk_error(
        openai.BadRequestError,
        status=400,
        body={
            "error": {"message": "Unrecognized request argument supplied: max_completion_tokens"}
        },
    )
    completions = FakeCompletions(errors=[error, None], responses=[azure_response("recovered")])
    provider = AzureOpenAIProvider(azure_config, client=FakeClient(completions), tracer=tracer)
    provider.complete(request_)
    assert "max_completion_tokens" in completions.calls[0]
    assert "max_tokens" in completions.calls[1]
    assert "max_completion_tokens" not in completions.calls[1]


def test_an_unrelated_bad_request_is_not_adapted_away(azure_config, tracer, request_):
    import openai

    error = make_sdk_error(
        openai.BadRequestError,
        status=400,
        body={"error": {"message": "content filter triggered"}},
    )
    completions = FakeCompletions(errors=[error] * 3)
    provider = AzureOpenAIProvider(
        azure_config,
        client=FakeClient(completions),
        tracer=tracer,
        retry=RetryPolicy(max_attempts=1),
        sleep=lambda _s: None,
    )
    with pytest.raises(ProviderBadRequest):
        provider.complete(request_)
    assert len(completions.calls) == 1


def test_describe_exposes_no_secrets(azure_config, tracer):
    described = AzureOpenAIProvider(azure_config, tracer=tracer).describe()
    assert "key" not in json.dumps(described).lower()
    assert described["endpoint"] == "https://example.openai.azure.com"


def test_sdk_retries_are_disabled_so_attempts_stay_visible(azure_config, monkeypatch):
    """Retrying happens in Provider.complete, where each attempt is recorded."""
    captured = {}

    class FakeAzureOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "sk-test")
    monkeypatch.setitem(
        __import__("sys").modules, "openai", SimpleNamespace(AzureOpenAI=FakeAzureOpenAI)
    )
    _ = AzureOpenAIProvider(azure_config).client  # the property builds the client
    assert captured["max_retries"] == 0
    assert captured["azure_endpoint"] == "https://example.openai.azure.com"


# --- live credentials ------------------------------------------------------
@pytest.mark.live
def test_a_real_azure_deployment_answers_and_calls_a_tool(tracer):
    """Runs only with real credentials: `pytest -m live`.

    Deselected by default so the suite stays free and deterministic. Its job is
    to catch the things fakes cannot - a wrong API version, a deployment that
    does not exist, a model that rejects a parameter this code has not learned
    about yet.
    """
    import os

    from gantry.config import Config

    config = Config.from_env()
    if not config.azure.configured:
        pytest.skip("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY are not set")

    provider = AzureOpenAIProvider(config.azure, tracer=tracer)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_status",
                "description": "Return the status of a named service.",
                "parameters": {
                    "type": "object",
                    "properties": {"service": {"type": "string"}},
                    "required": ["service"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }
    ]
    completion = provider.complete(
        CompletionRequest.of(
            [Message.user("Use the get_status tool to check the service named 'billing'.")],
            tools=tools,
            max_output_tokens=256,
        )
    )
    assert completion.usage.input_tokens > 0
    if completion.wants_tools:
        call = completion.tool_calls[0]
        assert call.name == "get_status"
        assert call.parse_arguments()["service"] == "billing"
    else:
        assert completion.text
    print(
        f"\nlive: model={completion.model} usage={completion.usage.as_dict()} "
        f"latency={completion.latency_ms:.0f}ms"
    )
    os.environ.setdefault("GANTRY_LIVE_OK", "1")
