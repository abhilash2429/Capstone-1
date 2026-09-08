"""Path containment.

The agent proposes paths; this decides whether they are inside the workspace.
Getting it right is mostly about refusing to be clever:

* Resolve fully before deciding. String prefix checks are defeated by ``..``,
  by a symlink inside the workspace pointing out of it, and by ``/tmp`` itself
  being a symlink on some systems - which is why the root is resolved once at
  construction rather than compared as written.
* An absolute path is treated as an absolute *host* path and rejected if it
  falls outside, rather than being quietly reinterpreted as workspace-relative.
  Silently rewriting ``/etc/passwd`` into ``<workspace>/etc/passwd`` would turn
  an attempted escape into a confusing success.
* Paths that do not exist yet still have to be checked, because that is what a
  write is.

**What this is not.** Resolving a path and then opening it is a time-of-check to
time-of-use race: a symlink swapped in between the two wins. :meth:`PathJail.open`
closes that for the final component with ``O_NOFOLLOW``, and container mode
closes it properly. A jail is a guardrail against an agent wandering, not a
boundary against an attacker who already runs code inside it.
"""

from __future__ import annotations

import errno
import os
from collections.abc import Iterator
from pathlib import Path

from gantry.errors import PathEscape


class PathJail:
    """Confines file access to a workspace root."""

    def __init__(self, root: str | Path, deny_symlinks: bool = False) -> None:
        # Resolved once: the root itself may be reached through a symlink, and
        # comparing a resolved path against an unresolved root rejects
        # everything legitimate on such a system.
        self.root = Path(root).expanduser().resolve()
        self.deny_symlinks = deny_symlinks
        if not self.root.is_dir():
            raise PathEscape(f"workspace root does not exist: {self.root}", root=str(self.root))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"PathJail({str(self.root)!r})"

    def contains(self, path: str | Path) -> bool:
        try:
            self.resolve(path)
        except PathEscape:
            return False
        return True

    def literal(self, candidate: str | Path) -> Path:
        """The path as written, anchored to the workspace, without resolving.

        Needed because ``O_NOFOLLOW`` has to be applied to the component the
        caller actually named. Handing it a resolved path defeats it entirely:
        resolution has already followed the symlink, so the flag inspects a
        regular file and the protection silently does nothing.
        """
        target = Path(str(candidate)).expanduser()
        return target if target.is_absolute() else self.root / target

    def resolve(self, candidate: str | Path) -> Path:
        """Return the resolved absolute path, or raise :class:`PathEscape`."""
        raw = str(candidate)
        if "\x00" in raw:
            raise PathEscape("path contains a null byte", path=raw)
        if not raw.strip():
            raise PathEscape("path is empty", path=raw)

        target = Path(raw).expanduser()
        if target.is_absolute():
            resolved = target.resolve()
            if not self._inside(resolved):
                raise PathEscape(
                    f"absolute path {raw!r} is outside the workspace",
                    path=raw,
                    workspace=str(self.root),
                    hint="use a path relative to the workspace root",
                )
        else:
            resolved = (self.root / target).resolve()
            if not self._inside(resolved):
                # Reached only via `..` or a symlink pointing out of the tree.
                raise PathEscape(
                    f"path {raw!r} resolves outside the workspace",
                    path=raw,
                    resolved=str(resolved),
                    workspace=str(self.root),
                )

        if self.deny_symlinks:
            self._reject_symlinks(raw, target if target.is_absolute() else self.root / target)
        return resolved

    def _inside(self, resolved: Path) -> bool:
        return resolved == self.root or self.root in resolved.parents

    def _reject_symlinks(self, raw: str, literal: Path) -> None:
        """Refuse any path traversing a symlink, even one staying inside.

        Walks the *literal* path rather than the resolved one. Resolution has
        already followed every link by then, so inspecting the result finds
        nothing - the check would pass silently while doing nothing at all.

        Off by default, since plenty of legitimate workspaces contain symlinks.
        Worth switching on when the workspace holds untrusted content: a link
        that points inside today can be repointed outside tomorrow.
        """
        for part in self._literal_ancestors(literal):
            if part.is_symlink():
                raise PathEscape(
                    f"path {raw!r} traverses the symlink {part}",
                    path=raw,
                    symlink=str(part),
                )

    def _literal_ancestors(self, path: Path) -> Iterator[Path]:
        """Each component of the path as written, from the leaf up to the root."""
        current = Path(os.path.normpath(path))
        while current != self.root and self.root in current.parents:
            yield current
            current = current.parent

    def relative(self, path: str | Path) -> Path:
        """The workspace-relative form, for messages and telemetry.

        Absolute host paths leak the machine's directory layout into traces and
        into the model's context, so everything user-facing uses this.
        """
        return self.resolve(path).relative_to(self.root)

    # -- I/O -------------------------------------------------------------
    def open(self, path: str | Path, mode: str = "r", *, create_parents: bool = False):
        """Open a file inside the jail.

        Writes use ``O_NOFOLLOW`` on the final component, closing the window
        between resolving a path and opening it - the one an attacker with
        write access to the workspace would use to redirect a write onto
        something outside it.

        Reads deliberately do not. Resolution has already confirmed the target
        is inside the workspace, and refusing every symlinked file would break
        ordinary trees (vendored dependencies, checked-out worktrees) for a
        race that only leaks a file already reachable. Where that residual
        window matters, ``deny_symlinks=True`` or container mode closes it.

        What remains open even for writes: an *intermediate* directory swapped
        for a symlink between the check and the open. Closing that needs
        per-component ``openat`` or Linux's ``RESOLVE_BENEATH``, and container
        mode is the answer this project gives instead.
        """
        resolved = self.resolve(path)  # containment check, follows symlinks
        writing = any(flag in mode for flag in ("w", "a", "x", "+"))
        if writing and create_parents:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            # The new parents must themselves still be inside.
            self.resolve(resolved.parent)

        # Open the path as written, not the resolved one, so O_NOFOLLOW applies
        # to the component the caller named. Containment was established above.
        opened = self.literal(path) if writing else resolved

        flags = getattr(os, "O_CLOEXEC", 0)
        if writing:
            flags |= os.O_NOFOLLOW
        if "a" in mode:
            flags |= os.O_WRONLY | os.O_CREAT | os.O_APPEND
        elif "x" in mode:
            flags |= os.O_WRONLY | os.O_CREAT | os.O_EXCL
        elif writing:
            flags |= os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        else:
            flags |= os.O_RDONLY

        try:
            fd = os.open(opened, flags, 0o644)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                # O_NOFOLLOW reports ELOOP when the final component is a
                # symlink. That is the race this call exists to close, so it is
                # an escape attempt rather than an ordinary I/O failure.
                raise PathEscape(
                    f"refusing to write through the symlink at {path!r}",
                    path=str(path),
                    resolved=str(resolved),
                ) from exc
            raise
        encoding = None if "b" in mode else "utf-8"
        return os.fdopen(fd, mode, encoding=encoding)

    def read_text(self, path: str | Path) -> str:
        with self.open(path, "r") as handle:
            return handle.read()

    def write_text(self, path: str | Path, content: str, create_parents: bool = True) -> Path:
        resolved = self.resolve(path)
        with self.open(path, "w", create_parents=create_parents) as handle:
            handle.write(content)
        return resolved

    def iter_files(self, pattern: str = "**/*", limit: int = 10_000) -> Iterator[Path]:
        """Walk the workspace, skipping anything that escapes it.

        A symlinked directory pointing outside would otherwise make a glob
        enumerate the whole filesystem.
        """
        count = 0
        for path in self.root.glob(pattern):
            if count >= limit:
                return
            try:
                resolved = path.resolve()
            except OSError:
                continue  # a broken symlink is not worth failing the walk over
            if self._inside(resolved) and resolved.is_file():
                count += 1
                yield resolved
