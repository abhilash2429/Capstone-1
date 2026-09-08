"""Providers: the boundary between the harness and a language model."""

from __future__ import annotations

import os
from typing import Any

from gantry.config import Config
from gantry.providers.azure import AzureOpenAIProvider
from gantry.providers.base import CompletionRequest, Provider, RetryPolicy
from gantry.providers.cache import CacheMiss, CacheMode, CachingProvider, ResponseCache
from gantry.providers.offline import (
    OFFLINE_MODEL,
    FailingProvider,
    OfflineProvider,
    Script,
    Turn,
)

__all__ = [
    "OFFLINE_MODEL",
    "AzureOpenAIProvider",
    "CacheMiss",
    "CacheMode",
    "CachingProvider",
    "CompletionRequest",
    "FailingProvider",
    "OfflineProvider",
    "Provider",
    "ResponseCache",
    "RetryPolicy",
    "Script",
    "Turn",
    "build_provider",
]


def build_provider(config: Config, **kwargs: Any) -> Provider:
    """Construct the configured provider, wrapped in a cache when asked.

    Defaults to the offline provider. Reaching a paid API should be something
    the operator opted into, not something that happens because a variable was
    left unset.
    """
    provider: Provider
    if config.provider == "azure":
        provider = AzureOpenAIProvider(config.azure, **kwargs)
    else:
        provider = OfflineProvider(**kwargs)

    mode = CacheMode(os.environ.get("GANTRY_CACHE_MODE", "off"))
    if mode is CacheMode.OFF:
        return provider
    directory = os.environ.get("GANTRY_CASSETTES", ".gantry/cassettes")
    return CachingProvider(provider, ResponseCache(directory), mode=mode, **kwargs)
