"""Tool definitions, validation and capability scoping."""

from gantry.tools.registry import ToolRegistry, check_strict_schema, format_validation_error
from gantry.tools.spec import (
    Capability,
    Grant,
    ToolContext,
    ToolHandler,
    ToolResult,
    ToolSpec,
)

__all__ = [
    "Capability",
    "Grant",
    "ToolContext",
    "ToolHandler",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "check_strict_schema",
    "format_validation_error",
]
