"""Running commands, through the sandbox rather than around it.

The tool itself is thin because every hard decision - what may run, what the
environment contains, what happens when it hangs - already belongs to
:mod:`gantry.sandbox`. What this module adds is the framing the model needs:
a non-zero exit is reported as a failed tool call with the full output
attached, because an agent that is told only "the command failed" will re-run
it to find out why, and that is a wasted turn against a paid provider.
"""

from __future__ import annotations

from typing import Any

from gantry.errors import ToolExecutionError
from gantry.sandbox.runner import SandboxRunner
from gantry.toolkit.common import ToolkitLimits, number, object_schema, optional, string
from gantry.tools import Capability, ToolContext, ToolResult, ToolSpec


class ShellTools:
    """The command-execution half of the toolkit."""

    def __init__(self, runner: SandboxRunner, limits: ToolkitLimits | None = None) -> None:
        self.runner = runner
        self.limits = limits or ToolkitLimits()

    @property
    def timeout_s(self) -> float:
        """How long the dispatcher waits for a bash call.

        Strictly longer than the sandbox's own ceiling. If the dispatcher gave
        up first it would abandon the waiting thread while the child process
        kept running, and the harness would report a timeout for a command
        that was still writing to the workspace - the worst of both, since
        nothing would then kill the process group.
        """
        return self.runner.config.timeout_s + self.limits.shell_timeout_margin_s

    def bash(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        command = args["command"]
        if not command.strip():
            raise ToolExecutionError("command is empty.")
        requested = args.get("timeout_s")
        timeout = min(float(requested), self.runner.config.timeout_s) if requested else None

        result = self.runner.run(command, cwd=args.get("cwd") or None, timeout_s=timeout)
        payload = {
            "command": command,
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "duration_ms": round(result.duration_ms, 1),
            "mode": result.mode,
        }

        if result.denied is not None:
            return ToolResult.failure(
                f"Refused by the command policy ({result.denied.rule}): {result.denied.reason}. "
                "This command will not run. Achieve the task another way.",
                error_code="sandbox.command_denied",
                rule=result.denied.rule,
                **payload,
            )
        text = result.combined_output(max_chars=24_000)
        if result.ok:
            return ToolResult.success(text or "[no output, exit code 0]", **payload)
        return ToolResult.failure(
            text,
            error_code="sandbox.timeout" if result.timed_out else "sandbox.nonzero_exit",
            **payload,
        )

    def specs(self) -> list[ToolSpec]:
        network = "off" if not self.runner.config.allow_network else "on"
        return [
            ToolSpec(
                name="bash",
                version="1.0.0",
                description=(
                    "Run a shell command inside the workspace sandbox. The working "
                    "directory is the workspace root, the environment is stripped to a "
                    f"minimal allowlist, and the network is {network}. Commands are "
                    "checked against a policy and dangerous ones are refused. Output is "
                    "captured and truncated."
                ),
                input_schema=object_schema(
                    command=string("The command line to run, e.g. 'python -m pytest -q'."),
                    cwd=optional(
                        string("Workspace-relative working directory"), "the workspace root"
                    ),
                    timeout_s=optional(
                        number("Seconds to allow before the command is killed"),
                        "the sandbox default",
                    ),
                ),
                handler=self.bash,
                capabilities=frozenset(
                    {Capability.PROC_EXEC, Capability.FS_READ, Capability.FS_WRITE}
                ),
                timeout_s=self.timeout_s,
                # A command can create files, start servers or consume a
                # budget. Re-running one after a lost answer is not safe.
                idempotent=False,
                destructive=True,
                concurrency_safe=False,
                max_output_chars=24_000,
            )
        ]
