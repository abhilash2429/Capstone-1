"""Shared pieces of the workspace toolkit: limits, schema helpers, decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gantry.errors import ToolExecutionError

#: Directories never worth walking. Skipping them is not only a speed
#: optimisation: a grep that returns forty hits from ``.venv`` has spent the
#: agent's context on vendored code it cannot change.
IGNORED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "dist",
        "build",
        ".tox",
        ".idea",
        ".gantry",
    }
)

#: Per-line ceiling for search output. A minified bundle on one line would
#: otherwise spend an entire context window on a single match.
MAX_LINE_CHARS = 400


@dataclass(frozen=True)
class ToolkitLimits:
    """Ceilings the tools enforce on themselves.

    Every one of these exists because the alternative is an agent that fills
    its context with a single tool result and then has no room left to act on
    it. They are conservative on purpose and adjustable per workspace.
    """

    #: Largest file ``read_file`` will open at all.
    max_read_bytes: int = 400_000
    #: Lines returned by one ``read_file`` call when no limit is given.
    max_read_lines: int = 1_500
    #: Largest file ``write_file`` will produce.
    max_write_bytes: int = 2_000_000
    #: Files larger than this are skipped by search rather than scanned.
    max_search_file_bytes: int = 1_000_000
    max_search_files: int = 20_000
    max_grep_matches: int = 100
    max_glob_results: int = 200
    #: Added to the sandbox's own timeout to get the dispatcher's. See
    #: :func:`gantry.toolkit.shell.ShellTools.specs` for why the two differ.
    shell_timeout_margin_s: float = 15.0


def looks_binary(raw: bytes) -> bool:
    """A NUL byte in the first block is the cheap, reliable binary tell."""
    return b"\x00" in raw[:8192]


def decode(raw: bytes, path: str) -> str:
    """Decode file bytes as UTF-8, refusing anything that is not text.

    Refusing is the useful behaviour. Lossy decoding hands the model a wall of
    replacement characters that it will try to reason about, and any edit
    written against that text would corrupt the file on the way back out.
    """
    if looks_binary(raw):
        raise ToolExecutionError(
            f"{path} is a binary file and cannot be read as text.",
            path=path,
            bytes=len(raw),
        )
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ToolExecutionError(
            f"{path} is not valid UTF-8 (byte {exc.start}: {exc.reason}).",
            path=path,
        ) from exc


# -- schema helpers ------------------------------------------------------
# Strict function calling requires every property to appear in `required`;
# optionality is expressed by admitting null. These helpers make that the
# default shape rather than something each tool has to remember.


def string(description: str, **extra: Any) -> dict[str, Any]:
    return {"type": "string", "description": description, **extra}


def integer(description: str, **extra: Any) -> dict[str, Any]:
    return {"type": "integer", "description": description, **extra}


def number(description: str, **extra: Any) -> dict[str, Any]:
    return {"type": "number", "description": description, **extra}


def boolean(description: str, **extra: Any) -> dict[str, Any]:
    return {"type": "boolean", "description": description, **extra}


def optional(schema: dict[str, Any], default: str = "the default") -> dict[str, Any]:
    """Make a property nullable, and say so where the model will read it."""
    kind = schema["type"]
    kinds = list(kind) if isinstance(kind, list) else [kind]
    if "null" not in kinds:
        kinds.append("null")
    return {
        **schema,
        "type": kinds,
        "description": f"{schema['description'].rstrip('.')}. Pass null for {default}.",
    }


def object_schema(**properties: dict[str, Any]) -> dict[str, Any]:
    """A strict-mode object schema over the given properties."""
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(properties),
        "additionalProperties": False,
    }
