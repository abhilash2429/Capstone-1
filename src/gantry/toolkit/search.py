"""Finding things: content search and path matching.

Both tools are implemented in Python rather than by shelling out to ``grep``
and ``find``. That is a deliberate trade of a little speed for three
properties the harness needs more: the results are identical on every machine
regardless of which ``grep`` is installed, they cannot be affected by the
sandbox's command policy, and search keeps working in container mode where
the image may carry no such binaries at all. Deterministic search also means
an eval fixture that depends on it scores the same way every run.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from gantry.errors import ToolExecutionError
from gantry.sandbox.jail import PathJail
from gantry.toolkit.common import (
    IGNORED_DIRS,
    MAX_LINE_CHARS,
    ToolkitLimits,
    boolean,
    integer,
    looks_binary,
    object_schema,
    optional,
    string,
)
from gantry.tools import Capability, ToolContext, ToolResult, ToolSpec


def _ignored(relative: Path) -> bool:
    return bool(set(relative.parts[:-1]) & IGNORED_DIRS)


class SearchTools:
    """The search half of the toolkit, bound to one workspace."""

    def __init__(self, jail: PathJail, limits: ToolkitLimits | None = None) -> None:
        self.jail = jail
        self.limits = limits or ToolkitLimits()

    # -- helpers ---------------------------------------------------------
    def _scope(self, raw: str | None, pattern: str) -> tuple[Path, str]:
        """Turn an optional subdirectory plus a glob into one rooted pattern."""
        if not raw or raw in {".", "./"}:
            return self.jail.root, pattern
        base = self.jail.resolve(raw)
        if not base.exists():
            raise ToolExecutionError(f"{raw} does not exist in the workspace.", path=raw)
        if not base.is_dir():
            raise ToolExecutionError(
                f"{raw} is a file, not a directory. Pass its directory instead.", path=raw
            )
        prefix = base.relative_to(self.jail.root).as_posix()
        return base, f"{prefix}/{pattern}"

    def _candidates(self, pattern: str) -> list[Path]:
        files: list[Path] = []
        for path in self.jail.iter_files(pattern, limit=self.limits.max_search_files):
            if not _ignored(path.relative_to(self.jail.root)):
                files.append(path)
        return files

    # -- grep ------------------------------------------------------------
    def grep(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        pattern = args["pattern"]
        flags = 0 if args.get("case_sensitive") else re.IGNORECASE
        try:
            regex = re.compile(pattern, flags)
        except re.error as exc:
            # The model wrote this pattern; handing back Python's own message
            # (with the offending position) is what lets it fix it in one turn.
            return ToolResult.failure(
                f"Invalid regular expression {pattern!r}: {exc}",
                error_code="tool.invalid_arguments",
            )

        _, rooted = self._scope(args.get("path"), args.get("glob") or "**/*")
        limit = max(
            1,
            min(
                int(args.get("max_results") or self.limits.max_grep_matches),
                self.limits.max_grep_matches,
            ),
        )

        matches: list[str] = []
        files_searched = 0
        files_matched: set[str] = set()
        capped = False
        for file in sorted(self._candidates(rooted)):
            if len(matches) >= limit:
                capped = True
                break
            try:
                if file.stat().st_size > self.limits.max_search_file_bytes:
                    continue
                raw = file.read_bytes()
            except OSError:
                continue  # a file that vanished mid-walk is not an error worth raising
            if looks_binary(raw):
                continue
            files_searched += 1
            shown = self.jail.relative(file).as_posix()
            for number, line in enumerate(raw.decode("utf-8", "replace").splitlines(), 1):
                if not regex.search(line):
                    continue
                files_matched.add(shown)
                text = line.strip()
                if len(text) > MAX_LINE_CHARS:
                    text = text[:MAX_LINE_CHARS] + " ..."
                matches.append(f"{shown}:{number}: {text}")
                if len(matches) >= limit:
                    capped = True
                    break

        if not matches:
            return ToolResult.success(
                f"No match for {pattern!r} in {files_searched} file(s).",
                pattern=pattern,
                matches=0,
                files_searched=files_searched,
            )
        header = f"{len(matches)} match(es) for {pattern!r} in {len(files_matched)} file(s)"
        if capped:
            header += f" (stopped at the {limit}-result cap; narrow the pattern or path)"
        return ToolResult.success(
            header + ":\n" + "\n".join(matches),
            pattern=pattern,
            matches=len(matches),
            files_matched=sorted(files_matched),
            files_searched=files_searched,
            capped=capped,
        )

    # -- glob ------------------------------------------------------------
    def glob(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        pattern = args["pattern"]
        _, rooted = self._scope(args.get("path"), pattern)
        limit = max(
            1,
            min(
                int(args.get("max_results") or self.limits.max_glob_results),
                self.limits.max_glob_results,
            ),
        )

        found = self._candidates(rooted)
        # Newest first. In a repository the agent has been working in, recency
        # is a better relevance signal than alphabetical order.
        found.sort(key=lambda p: (-p.stat().st_mtime, p.as_posix()))
        capped = len(found) > limit
        listed = [self.jail.relative(p).as_posix() for p in found[:limit]]
        if not listed:
            return ToolResult.success(f"No file matches {pattern!r}.", pattern=pattern, count=0)
        header = f"{len(listed)} file(s) matching {pattern!r}"
        if capped:
            header += f" (of {len(found)}, newest first)"
        return ToolResult.success(
            header + ":\n" + "\n".join(listed),
            pattern=pattern,
            count=len(listed),
            total=len(found),
            capped=capped,
            paths=listed,
        )

    # -- registration ----------------------------------------------------
    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="grep",
                version="1.0.0",
                description=(
                    "Search file contents with a Python regular expression. Returns "
                    "path:line: text for each match. Case-insensitive by default. "
                    "Binary files, and directories such as .git and node_modules, are skipped."
                ),
                input_schema=object_schema(
                    pattern=string("Python regular expression to search for."),
                    path=optional(
                        string("Workspace-relative directory to search in"), "the whole workspace"
                    ),
                    glob=optional(
                        string("Glob restricting which files are searched, e.g. '**/*.py'"),
                        "every file",
                    ),
                    case_sensitive=optional(
                        boolean("Match case exactly"), "a case-insensitive search"
                    ),
                    max_results=optional(integer("Maximum matches to return"), "the default cap"),
                ),
                handler=self.grep,
                capabilities=frozenset({Capability.FS_READ}),
                timeout_s=45.0,
                idempotent=True,
                concurrency_safe=True,
                max_output_chars=32_000,
            ),
            ToolSpec(
                name="glob",
                version="1.0.0",
                description=(
                    "List workspace files matching a glob such as 'src/**/*.py'. "
                    "Results are newest first."
                ),
                input_schema=object_schema(
                    pattern=string("Glob pattern, relative to the search path."),
                    path=optional(
                        string("Workspace-relative directory to search in"), "the workspace root"
                    ),
                    max_results=optional(integer("Maximum paths to return"), "the default cap"),
                ),
                handler=self.glob,
                capabilities=frozenset({Capability.FS_READ}),
                timeout_s=30.0,
                idempotent=True,
                concurrency_safe=True,
            ),
        ]
