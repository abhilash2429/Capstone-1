"""Reading, writing and editing files inside the workspace.

Three tools, and most of the code here is about the ways each one fails.
That ratio is the point. A file tool that works on the happy path is an
afternoon; what decides whether an agent finishes a task is whether a failed
edit comes back as *"no match; the file uses CRLF line endings"* or as
*"replacement failed"*. The first costs one turn, the second costs the run.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Any

from gantry.errors import CapabilityDenied, ToolExecutionError, ToolNotFound
from gantry.sandbox.jail import PathJail
from gantry.toolkit.common import (
    ToolkitLimits,
    boolean,
    decode,
    integer,
    object_schema,
    optional,
    string,
)
from gantry.toolkit.ledger import FileLedger
from gantry.tools import Capability, ToolContext, ToolResult, ToolSpec

#: Longest single line ``read_file`` will show before clipping it. Generous,
#: because the model copies from this output when composing an edit, and a
#: clipped line it copies will not match.
MAX_DISPLAY_LINE = 2_000

#: Paths the agent has no business writing to even inside its own workspace.
#: Rewriting ``.git`` is not a plausible task and is an expensive accident.
PROTECTED_PARTS: frozenset[str] = frozenset({".git"})

#: A line-number prefix as ``read_file`` renders it. Matching this in a failed
#: edit tells us the model copied the display format rather than the content.
NUMBERED_LINE = re.compile(r"(?m)^\s*\d+\t")


def _clip(line: str, limit: int = MAX_DISPLAY_LINE) -> str:
    if len(line) <= limit:
        return line
    return line[:limit] + f" ... [line clipped, {len(line) - limit} more characters]"


def _line_of(text: str, index: int) -> int:
    """1-based line number of a character offset."""
    return text.count("\n", 0, index) + 1


def _occurrence_lines(haystack: str, needle: str, limit: int = 5) -> list[int]:
    lines: list[int] = []
    start = 0
    while len(lines) < limit:
        found = haystack.find(needle, start)
        if found < 0:
            break
        lines.append(_line_of(haystack, found))
        start = found + max(1, len(needle))
    return lines


def _flatten(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines())


def _explain_miss(current: str, old: str) -> str:
    """Say *why* a replacement found nothing, in the order that is most likely.

    Each branch is a mistake seen in practice, and each one has a different
    fix, so collapsing them into "no match" throws away the only information
    the next turn needs.
    """
    if NUMBERED_LINE.search(old):
        return (
            "The text you gave includes the line numbers that read_file adds for display. "
            "They are not in the file. Send only the text after the tab."
        )
    if "\r\n" in current and "\r\n" not in old and old.replace("\n", "\r\n") in current:
        return (
            "This file uses CRLF (\\r\\n) line endings and your text uses LF. "
            "Match the file's endings, or edit a single line that has no newline in it."
        )
    stripped = old.strip()
    if stripped and stripped in current:
        return (
            "The text matches except for leading or trailing whitespace. "
            "Send the exact substring, without extra blank lines around it."
        )
    if stripped and _flatten(old) in _flatten(current):
        return (
            "The text matches except for indentation. Copy the lines exactly as read_file "
            "showed them, including the leading spaces or tabs."
        )
    first = old.splitlines()[0].strip() if old.splitlines() else ""
    if len(first) > 3 and first in current:
        hits = _occurrence_lines(current, first)
        return (
            f"Its first line appears at line(s) {', '.join(map(str, hits))}, so the "
            "difference is further down the block. Re-read that region and copy it exactly."
        )
    return "No part of it appears in the file. Read the file again before editing."


class FileTools:
    """The file half of the toolkit, bound to one workspace."""

    def __init__(
        self,
        jail: PathJail,
        ledger: FileLedger | None = None,
        limits: ToolkitLimits | None = None,
    ) -> None:
        self.jail = jail
        self.ledger = ledger or FileLedger()
        self.limits = limits or ToolkitLimits()

    # -- helpers ---------------------------------------------------------
    def _resolve_existing(self, raw: str) -> Path:
        resolved = self.jail.resolve(raw)
        if resolved.is_dir():
            raise ToolExecutionError(
                f"{raw} is a directory, not a file. Use glob to list what is inside it.",
                path=raw,
            )
        if not resolved.exists():
            raise ToolNotFound(
                f"{raw} does not exist. {self._suggest(resolved)}",
                path=raw,
            )
        return resolved

    def _suggest(self, missing: Path) -> str:
        """Turn a missing path into a next action rather than a dead end."""
        parent = missing.parent
        if not parent.is_dir():
            return f"The directory {self._show(parent)} does not exist either."
        names = sorted(entry.name for entry in parent.iterdir())
        close = difflib.get_close_matches(missing.name, names, n=3, cutoff=0.6)
        if close:
            return f"Did you mean: {', '.join(close)}?"
        if names:
            shown = ", ".join(names[:8]) + ("..." if len(names) > 8 else "")
            return f"{self._show(parent)} contains: {shown}"
        return f"{self._show(parent)} is empty."

    def _show(self, path: Path) -> str:
        """Workspace-relative rendering, so no host paths reach the model."""
        try:
            return str(self.jail.relative(path)) or "."
        except Exception:  # noqa: BLE001 - display only, never worth failing a call
            return path.name

    def _read_bytes(self, resolved: Path) -> bytes:
        with self.jail.open(resolved, "rb") as handle:
            return handle.read()

    def _write_bytes(self, path: str | Path, content: str) -> None:
        """Write bytes, not text.

        Text mode applies universal-newline translation on the way out, which
        rewrites the line endings of a file the agent only meant to edit one
        line of. Encoding here keeps the write byte-exact.
        """
        with self.jail.open(path, "wb", create_parents=True) as handle:
            handle.write(content.encode("utf-8"))

    def _check_writable(self, raw: str, resolved: Path) -> None:
        blocked = set(self.jail.relative(resolved).parts) & PROTECTED_PARTS
        if blocked:
            raise CapabilityDenied(
                f"Refusing to write to {raw}: it is inside a protected directory "
                f"({', '.join(sorted(blocked))}).",
                path=raw,
            )

    # -- read ------------------------------------------------------------
    def read_file(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raw_path = args["path"]
        resolved = self._resolve_existing(raw_path)
        size = resolved.stat().st_size
        if size > self.limits.max_read_bytes:
            raise ToolExecutionError(
                f"{raw_path} is {size:,} bytes, over the {self.limits.max_read_bytes:,} byte "
                "read limit. Use grep to find the part you need.",
                path=raw_path,
                bytes=size,
            )
        content = decode(self._read_bytes(resolved), self._show(resolved))
        if not content:
            self.ledger.record(resolved, content)
            return ToolResult.success(
                f"{self._show(resolved)} is empty (0 bytes).",
                path=self._show(resolved),
                total_lines=0,
                complete=True,
            )

        lines = content.splitlines()
        offset = max(1, int(args.get("offset") or 1))
        limit = int(args.get("limit") or self.limits.max_read_lines)
        limit = max(1, min(limit, self.limits.max_read_lines))
        start = min(offset - 1, len(lines))
        end = min(start + limit, len(lines))
        complete = start == 0 and end == len(lines)

        body = "\n".join(
            f"{n:>6}\t{_clip(line)}" for n, line in enumerate(lines[start:end], start + 1)
        )
        header = f"{self._show(resolved)} ({len(lines)} lines, {size:,} bytes)"
        if not complete:
            header += f", showing lines {start + 1}-{end}"
        footer = ""
        if end < len(lines):
            footer = (
                f"\n... {len(lines) - end} more lines. Call read_file again with offset={end + 1}."
            )

        # Only a complete read licenses a later edit: see FileLedger.
        self.ledger.record(resolved, content, complete=complete)
        return ToolResult.success(
            f"{header}\n{body}{footer}",
            path=self._show(resolved),
            total_lines=len(lines),
            lines_shown=end - start,
            bytes=size,
            complete=complete,
        )

    # -- write -----------------------------------------------------------
    def write_file(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raw_path = args["path"]
        content = args["content"]
        resolved = self.jail.resolve(raw_path)
        self._check_writable(raw_path, resolved)
        if resolved.is_dir():
            raise ToolExecutionError(f"{raw_path} is a directory.", path=raw_path)
        encoded = len(content.encode("utf-8"))
        if encoded > self.limits.max_write_bytes:
            raise ToolExecutionError(
                f"Refusing to write {encoded:,} bytes, over the "
                f"{self.limits.max_write_bytes:,} byte limit.",
                path=raw_path,
            )

        existed = resolved.exists()
        if existed:
            # Overwriting a file sight-unseen is how an agent deletes work it
            # did not know was there. The ledger check is the whole reason
            # write_file reads before it writes.
            current = decode(self._read_bytes(resolved), self._show(resolved))
            reason = self.ledger.stale_reason(resolved, current)
            if reason is not None:
                return ToolResult.failure(
                    f"Refusing to overwrite {self._show(resolved)}. {reason}",
                    error_code="tool.stale_write",
                    path=self._show(resolved),
                )

        self._write_bytes(raw_path, content)
        self.ledger.record(resolved, content)
        lines = content.count("\n") + (0 if content.endswith("\n") or not content else 1)
        verb = "Replaced" if existed else "Created"
        return ToolResult.success(
            f"{verb} {self._show(resolved)} ({lines} lines, {encoded:,} bytes).",
            path=self._show(resolved),
            created=not existed,
            bytes=encoded,
            lines=lines,
        )

    # -- edit ------------------------------------------------------------
    def edit_file(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raw_path = args["path"]
        old = args["old_string"]
        new = args["new_string"]
        replace_all = bool(args.get("replace_all") or False)

        resolved = self._resolve_existing(raw_path)
        self._check_writable(raw_path, resolved)
        if old == new:
            return ToolResult.failure(
                "old_string and new_string are identical, so this edit would change nothing.",
                error_code="tool.invalid_arguments",
            )
        if not old:
            return ToolResult.failure(
                "old_string is empty. Use write_file to create a file or replace it wholesale.",
                error_code="tool.invalid_arguments",
            )

        current = decode(self._read_bytes(resolved), self._show(resolved))
        reason = self.ledger.stale_reason(resolved, current)
        if reason is not None:
            return ToolResult.failure(
                f"Refusing to edit {self._show(resolved)}. {reason}",
                error_code="tool.stale_write",
                path=self._show(resolved),
            )

        count = current.count(old)
        if count == 0:
            return ToolResult.failure(
                f"No match in {self._show(resolved)}. {_explain_miss(current, old)}",
                error_code="tool.no_match",
                path=self._show(resolved),
            )
        if count > 1 and not replace_all:
            hits = _occurrence_lines(current, old)
            listed = ", ".join(str(n) for n in hits)
            more = "" if count <= len(hits) else f" (first {len(hits)} of {count})"
            return ToolResult.failure(
                f"{count} matches in {self._show(resolved)} at line(s) {listed}{more}. "
                "Include more surrounding context to identify one, or set replace_all "
                "to change every occurrence.",
                error_code="tool.ambiguous_match",
                path=self._show(resolved),
                matches=count,
            )

        first_line = _line_of(current, current.find(old))
        updated = current.replace(old, new) if replace_all else current.replace(old, new, 1)
        self._write_bytes(raw_path, updated)
        self.ledger.record(resolved, updated)

        delta = updated.count("\n") - current.count("\n")
        plural = "" if count == 1 or not replace_all else f" ({count} occurrences)"
        return ToolResult.success(
            f"Edited {self._show(resolved)} at line {first_line}{plural}. "
            f"Line count changed by {delta:+d}.",
            path=self._show(resolved),
            replacements=count if replace_all else 1,
            first_line=first_line,
            line_delta=delta,
        )

    # -- registration ----------------------------------------------------
    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="read_file",
                version="1.0.0",
                description=(
                    "Read a UTF-8 text file from the workspace. Output is prefixed with "
                    "line numbers for reference; those numbers are display only and are "
                    "not part of the file. Read a file before editing it."
                ),
                input_schema=object_schema(
                    path=string("Workspace-relative path to the file."),
                    offset=optional(integer("1-based line to start from"), "the start of the file"),
                    limit=optional(
                        integer("Maximum number of lines to return"), "the default page"
                    ),
                ),
                handler=self.read_file,
                capabilities=frozenset({Capability.FS_READ}),
                timeout_s=20.0,
                idempotent=True,
                concurrency_safe=True,
                max_output_chars=60_000,
            ),
            ToolSpec(
                name="write_file",
                version="1.0.0",
                description=(
                    "Create a file, or replace one you have already read, with exactly the "
                    "given content. Parent directories are created as needed. Prefer "
                    "edit_file for changes to an existing file."
                ),
                input_schema=object_schema(
                    path=string("Workspace-relative path to write."),
                    content=string("The complete new contents of the file."),
                ),
                handler=self.write_file,
                capabilities=frozenset({Capability.FS_READ, Capability.FS_WRITE}),
                timeout_s=20.0,
                idempotent=True,
                destructive=True,
                concurrency_safe=False,
            ),
            ToolSpec(
                name="edit_file",
                version="1.0.0",
                description=(
                    "Replace an exact substring in a file you have already read. The match "
                    "must be unique unless replace_all is true. Whitespace and indentation "
                    "must match the file exactly."
                ),
                input_schema=object_schema(
                    path=string("Workspace-relative path to edit."),
                    old_string=string("Exact text to find, including indentation."),
                    new_string=string("Text to put in its place. May be empty to delete."),
                    replace_all=optional(
                        boolean("Replace every occurrence instead of requiring a unique match"),
                        "a single unique match",
                    ),
                ),
                handler=self.edit_file,
                capabilities=frozenset({Capability.FS_READ, Capability.FS_WRITE}),
                timeout_s=20.0,
                # Applying the same replacement twice is not the same as once:
                # the second attempt finds nothing and fails. An automatic
                # retry after a timeout would report a failure for an edit that
                # in fact landed.
                idempotent=False,
                destructive=True,
                concurrency_safe=False,
            ),
        ]
