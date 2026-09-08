"""The provider interface, and the instrumentation every provider inherits.

Two things live here rather than in each provider.

**Instrumentation.** Tracing, cost attribution, metrics and retry are
implemented once, in :meth:`Provider.complete`, and every provider gets them by
implementing :meth:`Provider._complete`. Repeating that per provider is how
telemetry ends up subtly different between the code path you tested and the one
you shipped.

**Retry.** The SDK's own retries are switched off in favour of retrying here.
That is a deliberate trade: the SDK's implementation is fine, but its attempts
are invisible, and "the p99 tripled" is unanswerable when three quarters of a
request's latency is retries nobody recorded. Every attempt here is a span
event and a metric.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from gantry.errors import ProviderError
from gantry.messages import Completion, Message, Usage, to_wire
from gantry.telemetry import metrics
from gantry.telemetry import semconv as sc
from gantry.telemetry.pricing import PRICE_BOOK, PriceBook
from gantry.telemetry.tracer import Tracer, get_tracer


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_s: float = 0.5
    max_delay_s: float = 20.0

    def delay_for(self, attempt: int, retry_after_s: float | None = None) -> float:
        """Full jitter, with the provider's ``Retry-After`` taking precedence.

        Full jitter rather than plain exponential backoff: several agent runs
        rate-limited at the same moment otherwise retry in lockstep and
        rate-limit each other again.
        """
        if retry_after_s is not None:
            return min(retry_after_s, self.max_delay_s)
        ceiling = min(self.base_delay_s * (2**attempt), self.max_delay_s)
        return random.uniform(0, ceiling)  # noqa: S311 - jitter, not cryptography


@dataclass(frozen=True)
class CompletionRequest:
    """Everything needed to make one provider call.

    Frozen and hashable so it can key a response cache, which is what makes
    record-and-replay possible: a request that has been seen before must be
    recognisable as the same request.
    """

    messages: tuple[Message, ...] = ()
    tools: tuple[dict[str, Any], ...] = ()
    tool_choice: str | dict[str, Any] = "auto"
    max_output_tokens: int = 4096
    temperature: float | None = None
    top_p: float | None = None
    parallel_tool_calls: bool | None = None
    seed: int | None = None
    #: Which deployment role to use: "main" for reasoning, "fast" for cheap
    #: auxiliary calls. A role rather than a model id, because the mapping is
    #: deployment configuration.
    role: str = "main"
    timeout_s: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    @classmethod
    def of(cls, messages: list[Message], **kwargs: Any) -> CompletionRequest:
        tools = kwargs.pop("tools", None)
        return cls(messages=tuple(messages), tools=tuple(tools or ()), **kwargs)

    def cache_key(self, model: str) -> str:
        """A stable digest of everything that can change the answer.

        ``metadata`` is excluded on purpose - it carries run ids and timestamps,
        which differ every time and would make every request a cache miss.
        """
        payload = json.dumps(
            {
                "model": model,
                "messages": to_wire(list(self.messages)),
                "tools": list(self.tools),
                "tool_choice": self.tool_choice,
                "max_output_tokens": self.max_output_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "parallel_tool_calls": self.parallel_tool_calls,
                "seed": self.seed,
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()


class Provider(ABC):
    """Base class for anything that can turn messages into a completion."""

    #: Short identifier used in telemetry, e.g. "azure.openai".
    name: str = "provider"

    def __init__(
        self,
        tracer: Tracer | None = None,
        price_book: PriceBook | None = None,
        retry: RetryPolicy | None = None,
        sleep=time.sleep,
    ) -> None:
        self.tracer = tracer or get_tracer()
        self.price_book = price_book or PRICE_BOOK
        self.retry = retry or RetryPolicy()
        self._sleep = sleep

    # -- to implement ----------------------------------------------------
    @abstractmethod
    def _complete(self, request: CompletionRequest, model: str) -> Completion:
        """Make one attempt. Raise :class:`ProviderError` subclasses on failure."""

    @abstractmethod
    def model_for(self, role: str) -> str:
        """Resolve a role to the concrete model or deployment name."""

    # -- public ----------------------------------------------------------
    def complete(self, request: CompletionRequest) -> Completion:
        """Make a completion request, traced, priced, retried and counted."""
        model = self.model_for(request.role)
        attributes: dict[str, Any] = {
            sc.GEN_AI_REQUEST_MAX_TOKENS: request.max_output_tokens,
        }
        if request.temperature is not None:
            attributes[sc.GEN_AI_REQUEST_TEMPERATURE] = request.temperature
        if request.top_p is not None:
            attributes[sc.GEN_AI_REQUEST_TOP_P] = request.top_p

        started = time.monotonic()
        with self.tracer.llm_span(model=model, system=self.name, **attributes) as span:
            completion = self._attempt_with_retries(request, model, span)
            duration_s = time.monotonic() - started

            usage = completion.usage
            span.record_usage(
                model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cached_input_tokens=usage.cached_input_tokens,
                price_book=self.price_book,
            )
            span.set_attributes(
                {
                    sc.GEN_AI_RESPONSE_MODEL: completion.model or model,
                    sc.GEN_AI_RESPONSE_FINISH_REASONS: [str(completion.finish_reason)],
                    sc.CACHE_HIT: completion.cached,
                }
            )
            if completion.response_id:
                span.set_attribute(sc.GEN_AI_RESPONSE_ID, completion.response_id)
            if completion.truncated:
                # A reply cut off by the output cap often carries a half-written
                # tool call. Flagging it is how the loop avoids acting on one.
                span.add_event("output_truncated", max_tokens=request.max_output_tokens)

            metrics.observe_provider_call(
                model,
                duration_s,
                ok=True,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cost_usd=span.cost_usd,
            )
            completion.latency_ms = duration_s * 1000.0
            return completion

    def _attempt_with_retries(
        self, request: CompletionRequest, model: str, span: Any
    ) -> Completion:
        last: ProviderError | None = None
        for attempt in range(self.retry.max_attempts):
            try:
                return self._complete(request, model)
            except ProviderError as exc:
                last = exc
                if not exc.retryable or attempt == self.retry.max_attempts - 1:
                    span.set_status("error", exc.code)
                    metrics.PROVIDER_CALLS.inc(model=model, outcome="error")
                    raise
                delay = self.retry.delay_for(attempt, exc.details.get("retry_after_s"))
                span.add_event(
                    "provider_retry",
                    attempt=attempt + 1,
                    delay_s=round(delay, 3),
                    reason=exc.code,
                )
                metrics.PROVIDER_RETRIES.inc(model=model, reason=exc.code)
                self._sleep(delay)
        raise last or ProviderError("provider produced no completion")

    # -- helpers for subclasses ------------------------------------------
    @staticmethod
    def estimate_usage(request: CompletionRequest, output_text: str) -> Usage:
        """A crude deterministic token estimate.

        Used by providers that do not report usage. Four characters per token
        is wrong in detail and right in magnitude, which is what a fallback
        should be; the real numbers come from the provider when it sends them.
        """
        prompt_chars = sum(len(m.content or "") for m in request.messages)
        prompt_chars += sum(len(json.dumps(t, default=str)) for t in request.tools)
        return Usage(input_tokens=prompt_chars // 4, output_tokens=len(output_text) // 4)

    def close(self) -> None:
        """Release any connections. Safe to call more than once.

        Concrete rather than abstract: most providers hold nothing to release,
        and forcing every one of them to write an empty override adds noise
        without adding safety.
        """
        return None
