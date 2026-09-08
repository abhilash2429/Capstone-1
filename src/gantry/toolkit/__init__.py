"""The workspace toolkit: the tools a coding agent actually runs.

Everything in this package is deliberately thin. The harness is the product -
the contract, the registry, the dispatcher, the sandbox, the budget, the
gates - and this package is the demonstration that the harness is enough to
hang a real agent on. Six tools, no framework, no prompt engineering beyond
what the tool descriptions carry.

The one non-obvious idea here is the :class:`~gantry.toolkit.ledger.FileLedger`.
Read, write and edit share it, so the toolkit can refuse an edit whose basis
has gone stale rather than silently applying it to a file that changed. That
single piece of state is the difference between an agent that loses work and
one that notices it would have.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gantry.config import Config
from gantry.sandbox.jail import PathJail
from gantry.sandbox.runner import SandboxRunner
from gantry.toolkit.common import ToolkitLimits
from gantry.toolkit.files import FileTools
from gantry.toolkit.ledger import FileLedger, Observation
from gantry.toolkit.search import SearchTools
from gantry.toolkit.shell import ShellTools
from gantry.tools import ToolRegistry, ToolSpec

__all__ = [
    "FileLedger",
    "FileTools",
    "Observation",
    "SearchTools",
    "ShellTools",
    "Toolkit",
    "ToolkitLimits",
    "build_coding_agent",
    "build_toolkit",
]


@dataclass
class Toolkit:
    """One workspace's tools, plus the state they share."""

    jail: PathJail
    registry: ToolRegistry
    ledger: FileLedger
    limits: ToolkitLimits
    files: FileTools
    search: SearchTools
    runner: SandboxRunner | None = None
    shell: ShellTools | None = None
    names: tuple[str, ...] = field(default=())

    def specs(self) -> list[ToolSpec]:
        return [self.registry.get(name) for name in self.names]

    def describe(self) -> list[dict[str, Any]]:
        return self.registry.describe()


def build_toolkit(
    workspace: str | Path | PathJail,
    *,
    runner: SandboxRunner | None = None,
    registry: ToolRegistry | None = None,
    ledger: FileLedger | None = None,
    limits: ToolkitLimits | None = None,
    config: Config | None = None,
    shell: bool = True,
) -> Toolkit:
    """Register the workspace tools and hand back everything they share.

    Pass ``shell=False`` for a read-and-write-only agent. The capability model
    would refuse ``bash`` under a grant without ``proc:exec`` anyway, but not
    registering it at all keeps it out of the tool descriptions, and a tool the
    model never sees is a tool it never tries and never has to be told no about.
    """
    jail = workspace if isinstance(workspace, PathJail) else PathJail(workspace)
    registry = registry if registry is not None else ToolRegistry()
    ledger = ledger or FileLedger()
    limits = limits or ToolkitLimits()

    files = FileTools(jail, ledger=ledger, limits=limits)
    search = SearchTools(jail, limits=limits)
    specs = [*files.specs(), *search.specs()]

    shell_tools: ShellTools | None = None
    if shell:
        runner = runner or SandboxRunner(jail, config=(config or Config()).sandbox)
        shell_tools = ShellTools(runner, limits=limits)
        specs += shell_tools.specs()

    for spec in specs:
        registry.register(spec, replace=True)

    return Toolkit(
        jail=jail,
        registry=registry,
        ledger=ledger,
        limits=limits,
        files=files,
        search=search,
        runner=runner if shell else None,
        shell=shell_tools,
        names=tuple(spec.name for spec in specs),
    )


def build_coding_agent(
    provider: Any,
    workspace: str | Path,
    *,
    config: Config | None = None,
    limits: ToolkitLimits | None = None,
    registry: ToolRegistry | None = None,
    shell: bool = True,
    **kwargs: Any,
) -> Any:
    """A coding agent over a workspace: toolkit, sandbox and loop, wired up.

    Imported lazily so that the toolkit stays usable - and testable - without
    dragging in the provider and loop machinery.
    """
    from gantry.loop import Agent

    config = config or Config()
    jail = PathJail(workspace)
    runner = SandboxRunner(jail, config=config.sandbox, tracer=kwargs.get("tracer"))
    toolkit = build_toolkit(
        jail, runner=runner, registry=registry, limits=limits, config=config, shell=shell
    )
    agent = Agent(
        provider=provider,
        registry=toolkit.registry,
        jail=jail,
        runner=runner,
        config=config,
        **kwargs,
    )
    agent.toolkit = toolkit  # type: ignore[attr-defined]
    return agent
