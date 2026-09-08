from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gantry.telemetry import MemoryExporter, TelemetryStore, Tracer
from gantry.tools import ToolRegistry, ToolResult, ToolSpec


class FakeClock:
    """A clock the test drives, so deadline paths are covered without sleeping."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += seconds
        return self.now


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def exporter() -> MemoryExporter:
    return MemoryExporter()


@pytest.fixture
def tracer(exporter: MemoryExporter) -> Tracer:
    return Tracer(exporter=exporter, environment="test")


@pytest.fixture
def store(tmp_path: Path) -> TelemetryStore:
    with TelemetryStore(tmp_path / "telemetry.db") as store:
        yield store


def echo_schema(*names: str) -> dict:
    return {
        "type": "object",
        "properties": {name: {"type": "string"} for name in names},
        "required": list(names),
        "additionalProperties": False,
    }


def make_spec(name: str = "echo", **overrides) -> ToolSpec:
    defaults = dict(
        name=name,
        version="1.0.0",
        description=f"Echo tool {name}.",
        input_schema=echo_schema("text"),
        handler=lambda args, ctx: ToolResult.success(args.get("text", "")),
    )
    defaults.update(overrides)
    return ToolSpec(**defaults)


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry()
