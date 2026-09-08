"""Configuration, resolved from environment with explicit defaults.

Configuration is a frozen dataclass built once at startup rather than a module
of globals read at each call site. That makes a run reproducible: the exact
config is snapshotted into the trace and the eval record, so a result from three
weeks ago can be explained.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from gantry.errors import ConfigError

_TRUTHY = {"1", "true", "yes", "on"}


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    return default if raw is None else raw.strip().lower() in _TRUTHY


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer, got {raw!r}") from exc


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be a number, got {raw!r}") from exc


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal ``.env`` loader.

    Deliberately dependency-free and non-overriding: a value already present in
    the real environment always wins, so a CI secret is never shadowed by a
    stale file someone left in the working tree.
    """
    file = Path(path)
    if not file.is_file():
        return
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class AzureConfig:
    """Azure OpenAI connection settings.

    ``deployments`` maps a *role* the harness cares about (``main`` for
    reasoning-heavy turns, ``fast`` for cheap auxiliary calls) onto an Azure
    deployment name. Roles rather than model ids, because the deployment name is
    chosen by whoever provisioned the resource and rarely matches the model.
    """

    endpoint: str = ""
    api_version: str = "2024-10-21"
    api_key_env: str = "AZURE_OPENAI_API_KEY"
    deployments: dict[str, str] = field(default_factory=lambda: {"main": "", "fast": ""})
    timeout_s: float = 120.0
    max_retries: int = 3

    @property
    def configured(self) -> bool:
        return bool(self.endpoint and os.environ.get(self.api_key_env))

    def deployment(self, role: str = "main") -> str:
        name = self.deployments.get(role) or self.deployments.get("main", "")
        if not name:
            raise ConfigError(
                f"no Azure deployment configured for role {role!r}",
                hint="set GANTRY_DEPLOYMENT_MAIN in the environment or .env",
            )
        return name


@dataclass(frozen=True)
class BudgetConfig:
    """Hard ceilings on a single agent run."""

    max_steps: int = 24
    max_tool_calls: int = 80
    max_tokens: int = 400_000
    max_cost_usd: float = 2.00
    wall_clock_s: float = 900.0
    max_consecutive_tool_errors: int = 4
    #: How many times the same (tool, arguments) pair may repeat before the run
    #: is stopped for lack of progress.
    max_repeat_actions: int = 3


@dataclass(frozen=True)
class SandboxConfig:
    root: str = "."
    #: Wall-clock ceiling for a single sandboxed command.
    timeout_s: float = 60.0
    max_output_bytes: int = 256 * 1024
    max_cpu_seconds: int = 30
    max_memory_mb: int = 1024
    max_processes: int = 64
    max_file_size_mb: int = 64
    allow_network: bool = False
    #: When true, commands run in a disposable container instead of a subprocess.
    use_container: bool = False
    container_image: str = "python:3.11-slim"


@dataclass(frozen=True)
class TelemetryConfig:
    db_path: str = ".gantry/gantry.db"
    enabled: bool = True
    otlp_file: str = ""
    #: Record prompt and completion text on spans. Off by default because
    #: traces routinely contain source code and user data.
    capture_content: bool = False


@dataclass(frozen=True)
class Config:
    provider: str = "offline"
    environment: str = "local"
    log_level: str = "info"
    azure: AzureConfig = field(default_factory=AzureConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)

    @classmethod
    def from_env(cls, dotenv: str | Path | None = ".env") -> Config:
        if dotenv:
            load_dotenv(dotenv)
        azure = AzureConfig(
            endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            deployments={
                "main": os.environ.get("GANTRY_DEPLOYMENT_MAIN", ""),
                "fast": os.environ.get("GANTRY_DEPLOYMENT_FAST", ""),
            },
            timeout_s=_env_float("GANTRY_AZURE_TIMEOUT_S", 120.0),
            max_retries=_env_int("GANTRY_AZURE_MAX_RETRIES", 3),
        )
        budget = BudgetConfig(
            max_steps=_env_int("GANTRY_MAX_STEPS", 24),
            max_tool_calls=_env_int("GANTRY_MAX_TOOL_CALLS", 80),
            max_tokens=_env_int("GANTRY_MAX_TOKENS", 400_000),
            max_cost_usd=_env_float("GANTRY_MAX_COST_USD", 2.00),
            wall_clock_s=_env_float("GANTRY_WALL_CLOCK_S", 900.0),
        )
        sandbox = SandboxConfig(
            root=os.environ.get("GANTRY_SANDBOX_ROOT", "."),
            timeout_s=_env_float("GANTRY_SANDBOX_TIMEOUT_S", 60.0),
            allow_network=_env_bool("GANTRY_SANDBOX_ALLOW_NETWORK", False),
            use_container=_env_bool("GANTRY_SANDBOX_CONTAINER", False),
            container_image=os.environ.get("GANTRY_SANDBOX_IMAGE", "python:3.11-slim"),
        )
        telemetry = TelemetryConfig(
            db_path=os.environ.get("GANTRY_DB", ".gantry/gantry.db"),
            enabled=_env_bool("GANTRY_TELEMETRY", True),
            otlp_file=os.environ.get("GANTRY_OTLP_FILE", ""),
            capture_content=_env_bool("GANTRY_CAPTURE_CONTENT", False),
        )
        provider = os.environ.get("GANTRY_PROVIDER", "offline").strip().lower()
        if provider not in {"offline", "azure"}:
            raise ConfigError(f"unknown provider {provider!r}", expected=["offline", "azure"])
        return cls(
            provider=provider,
            environment=os.environ.get("GANTRY_ENV", "local"),
            log_level=os.environ.get("GANTRY_LOG_LEVEL", "info").lower(),
            azure=azure,
            budget=budget,
            sandbox=sandbox,
            telemetry=telemetry,
        )

    def with_overrides(self, **changes: Any) -> Config:
        return replace(self, **changes)

    def snapshot(self) -> dict[str, Any]:
        """A redacted, JSON-safe copy for embedding in traces and eval records."""
        data = asdict(self)
        data["azure"].pop("api_key_env", None)
        return data
