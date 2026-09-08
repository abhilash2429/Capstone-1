"""Token pricing and cost attribution.

Cost is treated as a first-class evaluation signal here, not a footnote. An
agent change that raises the pass rate by two points while tripling spend is a
regression, and you can only see that if every provider call is priced at the
moment it happens.

Three decisions worth naming:

* Prices live in ``pricing.json``, not in code, so updating them is a data
  change with a diff rather than a code change with a release.
* Every entry carries a ``verified`` flag. Published rates change and a
  confidently wrong dollar figure is worse than an admitted unknown, so the
  distinction survives all the way to the dashboard.
* An unknown model yields ``priced=False`` rather than ``$0.00``. Silent zeros
  are how cost tracking quietly stops working the day someone adds a
  deployment.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

_DATA_PATH = Path(__file__).with_name("pricing.json")

#: Cached input tokens are billed at a discount. Used when a model has no
#: explicit ``cached_input`` rate.
DEFAULT_CACHED_INPUT_RATIO = 0.25


@dataclass(frozen=True)
class ModelPrice:
    """USD per million tokens."""

    input: float
    output: float
    cached_input: float | None = None
    verified: bool = False

    def cached_rate(self) -> float:
        if self.cached_input is not None:
            return self.cached_input
        return self.input * DEFAULT_CACHED_INPUT_RATIO


@dataclass(frozen=True)
class Cost:
    """The outcome of pricing one provider call."""

    usd: float = 0.0
    #: False when the model has no price entry at all.
    priced: bool = False
    #: False when the price entry exists but is marked provisional.
    verified: bool = False
    model: str = ""

    def as_attributes(self) -> dict[str, object]:
        return {
            "gantry.cost.usd": round(self.usd, 8),
            "gantry.cost.priced": self.priced,
            "gantry.cost.verified": self.verified,
        }


class PriceBook:
    """Resolves a model or deployment id to a price."""

    def __init__(
        self, prices: dict[str, ModelPrice] | None = None, meta: dict | None = None
    ) -> None:
        self._prices = dict(prices or {})
        self.meta = dict(meta or {})

    @classmethod
    def load(cls, path: str | Path | None = None) -> PriceBook:
        """Load from the bundled table, then apply ``GANTRY_PRICING_FILE``."""
        book = cls._read(path or _DATA_PATH)
        override = os.environ.get("GANTRY_PRICING_FILE")
        if override and Path(override).is_file():
            book._prices.update(cls._read(override)._prices)
        return book

    @classmethod
    def _read(cls, path: str | Path) -> PriceBook:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        models = raw.get("models", raw)
        prices = {
            name: ModelPrice(
                input=float(entry["input"]),
                output=float(entry["output"]),
                cached_input=(
                    float(entry["cached_input"]) if entry.get("cached_input") is not None else None
                ),
                verified=bool(entry.get("verified", False)),
            )
            for name, entry in models.items()
        }
        return cls(prices, raw.get("_meta", {}))

    def get(self, model: str) -> ModelPrice | None:
        """Resolve a price, tolerating Azure deployment naming.

        Azure deployment names are chosen by whoever provisioned the resource,
        so ``gpt-4o-mini-prod-eastus`` is a normal thing to see. Falling back to
        the longest matching known prefix keeps cost tracking working without
        forcing every deployment into the table by hand.
        """
        if model in self._prices:
            return self._prices[model]
        normalised = model.strip().lower()
        if normalised in self._prices:
            return self._prices[normalised]
        candidates = [name for name in self._prices if normalised.startswith(name)]
        if not candidates:
            return None
        return self._prices[max(candidates, key=len)]

    def price(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
    ) -> Cost:
        """Cost one provider call.

        ``cached_input_tokens`` is the subset of ``input_tokens`` served from
        the prompt cache; it is billed at the cached rate and subtracted from
        the full-rate portion rather than added on top.
        """
        entry = self.get(model)
        if entry is None:
            return Cost(usd=0.0, priced=False, verified=False, model=model)
        billable_input = max(0, input_tokens - cached_input_tokens)
        usd = (
            billable_input * entry.input
            + cached_input_tokens * entry.cached_rate()
            + output_tokens * entry.output
        ) / 1_000_000
        return Cost(usd=usd, priced=True, verified=entry.verified, model=model)

    def unverified_models(self) -> list[str]:
        """Models whose rates are provisional. Surfaced in the dashboard."""
        return sorted(name for name, price in self._prices.items() if not price.verified)


PRICE_BOOK = PriceBook.load()
