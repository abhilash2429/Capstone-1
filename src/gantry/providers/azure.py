"""Azure OpenAI provider.

Three things here are shaped by how Azure actually behaves rather than by how
the documentation reads.

**Deployments, not models.** An Azure deployment name is chosen by whoever
provisioned the resource, so ``gpt-4o-mini-prod-eastus`` is a normal thing to
see and the model id is not recoverable from it. The harness therefore talks in
*roles* - ``main`` for reasoning turns, ``fast`` for cheap auxiliary ones - and
the role-to-deployment mapping is configuration.

**Parameters differ by model behind the deployment.** Newer models reject
``max_tokens`` in favour of ``max_completion_tokens``, and reasoning models
reject ``temperature`` outright. The same request that works against one
deployment is a 400 against another. Rather than making the operator find that
out in production, an unsupported-parameter 400 is detected once, the parameter
is dropped, and the request is retried; the adaptation is remembered for the
rest of the process and recorded on the span so it is visible rather than
mysterious.

**The SDK's retries are off.** Retrying happens in :class:`Provider` instead,
where every attempt becomes a span event and a metric. Invisible retries make
latency regressions unattributable.
"""

from __future__ import annotations

import os
import re
from typing import Any

from gantry.config import AzureConfig
from gantry.errors import (
    ConfigError,
    ProviderAuthError,
    ProviderBadRequest,
    ProviderError,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
)
from gantry.messages import Completion, FinishReason, Message, Usage, to_wire
from gantry.providers.base import CompletionRequest, Provider
from gantry.telemetry import semconv as sc

#: Parameters Azure deployments are known to reject depending on the model
#: behind them. Detected from the error body rather than guessed from the
#: deployment name, which carries no reliable model information.
ADAPTABLE_PARAMS = ("temperature", "top_p", "parallel_tool_calls", "seed")

_UNSUPPORTED_RE = re.compile(
    r"unsupported (?:value|parameter)|is not supported with this model"
    r"|unrecognized request argument|not supported in this version",
    re.IGNORECASE,
)


class AzureOpenAIProvider(Provider):
    """Chat Completions against an Azure OpenAI resource."""

    name = "azure.openai"

    def __init__(self, config: AzureConfig, client: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = config
        self._client = client
        #: Parameters this process has learned a deployment will not accept.
        self._unsupported: dict[str, set[str]] = {}
        #: Deployments that want max_completion_tokens rather than max_tokens.
        self._token_param: dict[str, str] = {}

    # -- client ----------------------------------------------------------
    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _build_client(self) -> Any:
        try:
            from openai import AzureOpenAI
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise ConfigError(
                "the Azure provider needs the openai package",
                hint='pip install "gantry-harness[azure]"',
            ) from exc

        if not self.config.endpoint:
            raise ConfigError(
                "AZURE_OPENAI_ENDPOINT is not set",
                hint="copy .env.example to .env and fill it in",
            )
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise ConfigError(
                f"{self.config.api_key_env} is not set",
                hint="the key belongs in .env or the environment, never in the repository",
            )
        return AzureOpenAI(
            azure_endpoint=self.config.endpoint,
            api_version=self.config.api_version,
            api_key=api_key,
            timeout=self.config.timeout_s,
            # Retries are handled by Provider.complete so that each attempt is
            # recorded. See this module's docstring.
            max_retries=0,
        )

    def model_for(self, role: str) -> str:
        return self.config.deployment(role)

    def close(self) -> None:
        if self._client is not None and hasattr(self._client, "close"):
            self._client.close()

    # -- request ---------------------------------------------------------
    def _build_kwargs(self, request: CompletionRequest, deployment: str) -> dict[str, Any]:
        unsupported = self._unsupported.get(deployment, set())
        token_param = self._token_param.get(deployment, "max_completion_tokens")

        kwargs: dict[str, Any] = {
            "model": deployment,
            "messages": to_wire(list(request.messages)),
            token_param: request.max_output_tokens,
        }
        if request.tools:
            kwargs["tools"] = list(request.tools)
            kwargs["tool_choice"] = request.tool_choice
        optional = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "seed": request.seed,
            "parallel_tool_calls": request.parallel_tool_calls if request.tools else None,
        }
        for key, value in optional.items():
            if value is not None and key not in unsupported:
                kwargs[key] = value
        if request.timeout_s is not None:
            kwargs["timeout"] = request.timeout_s
        return kwargs

    def _complete(self, request: CompletionRequest, model: str) -> Completion:
        kwargs = self._build_kwargs(request, model)
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            adapted = self._adapt(exc, model, kwargs)
            if adapted is None:
                raise self._translate(exc) from exc
            # One retry with the offending parameter removed. The adaptation is
            # remembered, so this costs at most one extra call per process.
            response = self.client.chat.completions.create(**adapted)
        return self._normalise(response, model)

    # -- adaptation ------------------------------------------------------
    def _adapt(self, exc: Exception, deployment: str, kwargs: dict[str, Any]) -> dict | None:
        """Rewrite a request a deployment rejected, once, and remember why.

        Returns the adjusted keyword arguments, or ``None`` when the error is
        not something a parameter change can fix.
        """
        from openai import BadRequestError

        if not isinstance(exc, BadRequestError):
            return None
        body = f"{getattr(exc, 'body', '') or ''} {exc}"
        if not _UNSUPPORTED_RE.search(body):
            return None

        # The two output-cap parameter names are mutually exclusive and which
        # one a deployment wants depends on the model behind it, so the swap has
        # to work in both directions rather than only the modern one.
        for sent, wanted in (
            ("max_completion_tokens", "max_tokens"),
            ("max_tokens", "max_completion_tokens"),
        ):
            if sent in kwargs and sent in body:
                self._token_param[deployment] = wanted
                retry = dict(kwargs)
                retry[wanted] = retry.pop(sent)
                return retry

        offending = next((p for p in ADAPTABLE_PARAMS if p in kwargs and p in body), None)
        if offending is None:
            return None
        self._unsupported.setdefault(deployment, set()).add(offending)
        retry = dict(kwargs)
        retry.pop(offending, None)
        return retry

    # -- errors ----------------------------------------------------------
    @staticmethod
    def _retry_after(exc: Any) -> float | None:
        headers = getattr(getattr(exc, "response", None), "headers", None)
        if not headers:
            return None
        raw = headers.get("retry-after") or headers.get("Retry-After")
        try:
            return float(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    def _translate(self, exc: Exception) -> ProviderError:
        """Map SDK exceptions onto the harness's typed, retry-aware errors.

        Ordered most specific first. A single broad catch would lose the only
        distinction that matters to the caller: whether trying again could
        possibly help.
        """
        import openai

        request_id = getattr(exc, "request_id", None)
        status = getattr(exc, "status_code", None)
        details: dict[str, Any] = {"request_id": request_id, "status_code": status}

        if isinstance(exc, openai.AuthenticationError | openai.PermissionDeniedError):
            return ProviderAuthError(
                "Azure OpenAI rejected the credentials or denied access to the deployment",
                **details,
            )
        if isinstance(exc, openai.RateLimitError):
            return ProviderRateLimited(
                "Azure OpenAI rate limit reached",
                retry_after_s=self._retry_after(exc),
                **details,
            )
        if isinstance(exc, openai.APITimeoutError):
            return ProviderTimeout("Azure OpenAI request timed out", **details)
        if isinstance(exc, openai.APIConnectionError):
            return ProviderUnavailable(f"could not reach Azure OpenAI: {exc}", **details)
        if isinstance(exc, openai.NotFoundError):
            return ProviderBadRequest(
                "deployment not found; check the deployment name and API version", **details
            )
        if isinstance(exc, openai.BadRequestError | openai.UnprocessableEntityError):
            return ProviderBadRequest(f"Azure OpenAI rejected the request: {exc}", **details)
        if isinstance(exc, openai.InternalServerError):
            return ProviderUnavailable(f"Azure OpenAI server error: {exc}", **details)
        if isinstance(exc, openai.APIStatusError):
            return ProviderError(f"Azure OpenAI returned {status}: {exc}", **details)
        return ProviderError(f"{type(exc).__name__}: {exc}", **details)

    # -- response --------------------------------------------------------
    @staticmethod
    def _normalise(response: Any, deployment: str) -> Completion:
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise ProviderError(
                "Azure OpenAI returned no choices",
                response_id=getattr(response, "id", None),
            )
        choice = choices[0]
        message = Message.from_wire(choice.message)

        raw_usage = getattr(response, "usage", None)
        prompt_details = getattr(raw_usage, "prompt_tokens_details", None)
        completion_details = getattr(raw_usage, "completion_tokens_details", None)
        usage = Usage(
            input_tokens=getattr(raw_usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(raw_usage, "completion_tokens", 0) or 0,
            cached_input_tokens=getattr(prompt_details, "cached_tokens", 0) or 0,
            reasoning_tokens=getattr(completion_details, "reasoning_tokens", 0) or 0,
        )
        return Completion(
            message=message,
            finish_reason=FinishReason.parse(getattr(choice, "finish_reason", None)),
            # The response reports the underlying model, which is the only way
            # to learn what actually sits behind a deployment name.
            model=getattr(response, "model", "") or deployment,
            usage=usage,
            response_id=getattr(response, "id", "") or "",
        )

    def describe(self) -> dict[str, Any]:
        """Connection facts for the dashboard, with no secrets."""
        return {
            "provider": self.name,
            "endpoint": self.config.endpoint,
            "api_version": self.config.api_version,
            "deployments": dict(self.config.deployments),
            "configured": self.config.configured,
            "adapted_away": {k: sorted(v) for k, v in self._unsupported.items()},
            sc.GEN_AI_SYSTEM: self.name,
        }
