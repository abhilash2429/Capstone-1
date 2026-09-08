"""Record and replay for provider traffic.

An evaluation number is only meaningful if it can be reproduced, and a
provider is the least reproducible thing in the system: temperature, model
updates behind a deployment name, and outright non-determinism all move the
result. Recording real traffic once and replaying it turns an agent benchmark
from an anecdote into a regression test.

The mode matters more than the mechanism:

* ``off`` - always call the provider.
* ``read`` - use a recording when there is one, call the provider otherwise.
* ``record`` - call the provider and save every response.
* ``replay`` - **never** call the provider; a miss is an error.

``replay`` is the one CI uses. A cache that silently falls through on a miss
would let a pull request quietly spend money and produce numbers that cannot
be reproduced, which is exactly the failure this exists to prevent.
"""

from __future__ import annotations

import json
import threading
from enum import StrEnum
from pathlib import Path
from typing import Any

from gantry.errors import ProviderError
from gantry.messages import Completion, FinishReason, Message, Usage
from gantry.providers.base import CompletionRequest, Provider


class CacheMode(StrEnum):
    OFF = "off"
    READ = "read"
    RECORD = "record"
    REPLAY = "replay"


class CacheMiss(ProviderError):
    code = "provider.cache_miss"


def _to_payload(completion: Completion) -> dict[str, Any]:
    return {
        "message": completion.message.to_wire(),
        "finish_reason": str(completion.finish_reason),
        "model": completion.model,
        "usage": completion.usage.as_dict(),
        "response_id": completion.response_id,
    }


def _from_payload(payload: dict[str, Any]) -> Completion:
    return Completion(
        message=Message.from_wire(payload.get("message", {})),
        finish_reason=FinishReason.parse(payload.get("finish_reason")),
        model=payload.get("model", ""),
        usage=Usage(**(payload.get("usage") or {})),
        response_id=payload.get("response_id", ""),
        cached=True,
    )


class ResponseCache:
    """A content-addressed store of completions, one JSON file per request."""

    def __init__(self, directory: str | Path = ".gantry/cassettes") -> None:
        self.directory = Path(directory)
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        # Two-character shard, so a large cassette directory stays navigable
        # and does not put tens of thousands of files in one folder.
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> Completion | None:
        path = self._path(key)
        if not path.is_file():
            with self._lock:
                self.misses += 1
            return None
        with self._lock:
            self.hits += 1
        return _from_payload(json.loads(path.read_text(encoding="utf-8")))

    def put(self, key: str, completion: Completion) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temporary file then rename: an interrupted recording must
        # not leave a half-written cassette that fails to parse on replay.
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(_to_payload(completion), indent=2), encoding="utf-8")
        temp.replace(path)

    def stats(self) -> dict[str, int]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": round(100 * self.hits / total) if total else 0,
        }


class CachingProvider(Provider):
    """Wraps a provider with record-and-replay.

    Subclasses :class:`Provider` so a cache hit is traced, priced and counted
    exactly like a live call, with ``gantry.cache.hit`` distinguishing the two.
    A replayed run that reported no telemetry would be useless for the very
    comparisons it exists to support.
    """

    def __init__(
        self,
        inner: Provider,
        cache: ResponseCache | None = None,
        mode: CacheMode | str = CacheMode.OFF,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.inner = inner
        self.cache = cache or ResponseCache()
        self.mode = CacheMode(mode)
        self.name = f"{inner.name}+cache"

    def model_for(self, role: str) -> str:
        return self.inner.model_for(role)

    def close(self) -> None:
        self.inner.close()

    def _complete(self, request: CompletionRequest, model: str) -> Completion:
        if self.mode is CacheMode.OFF:
            return self.inner._complete(request, model)

        key = request.cache_key(model)
        if self.mode in (CacheMode.READ, CacheMode.REPLAY):
            hit = self.cache.get(key)
            if hit is not None:
                return hit
            if self.mode is CacheMode.REPLAY:
                raise CacheMiss(
                    "no recorded response for this request and replay mode forbids "
                    "calling the provider",
                    key=key,
                    model=model,
                    hint="re-record with GANTRY_CACHE_MODE=record",
                )

        completion = self.inner._complete(request, model)
        if self.mode in (CacheMode.RECORD, CacheMode.READ):
            self.cache.put(key, completion)
        return completion
