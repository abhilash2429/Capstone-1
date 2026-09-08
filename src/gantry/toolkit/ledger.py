"""What the agent has read, and what it read.

An agent that edits a file it has never looked at is guessing, and an agent
that edits a file it read four steps ago is working from a snapshot that a
test run or a shell command may already have invalidated. The ledger closes
both gaps: it records every observation, and it refuses an edit whose basis
has changed underneath it.

Freshness is decided by hashing the content, not by comparing timestamps.
That is a deliberate reversal of the obvious implementation. CPython's own
bytecode cache trusts (mtime-in-seconds, size) and is wrong for exactly the
edits an agent makes - a same-length change landing inside the same second -
which is a bug this project has already been bitten by once. A digest costs
microseconds on files of this size and cannot be fooled that way.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path


def digest_of(content: str) -> str:
    """A stable content fingerprint. Not a security primitive, a change detector."""
    return hashlib.sha256(content.encode("utf-8", "surrogatepass")).hexdigest()[:16]


@dataclass(frozen=True)
class Observation:
    """One recorded look at a file."""

    path: str
    digest: str
    size: int
    lines: int
    at: float
    #: False when the agent saw only part of the file, which is enough to
    #: reason about but not enough to justify overwriting the whole thing.
    complete: bool = True


@dataclass
class FileLedger:
    """The set of files this run has observed, keyed by absolute path."""

    entries: dict[str, Observation] = field(default_factory=dict)
    clock: object = time.time

    def record(self, path: Path, content: str, *, complete: bool = True) -> Observation:
        observation = Observation(
            path=str(path),
            digest=digest_of(content),
            size=len(content),
            lines=content.count("\n") + (0 if content.endswith("\n") or not content else 1),
            at=self.clock(),  # type: ignore[operator]
            complete=complete,
        )
        self.entries[str(path)] = observation
        return observation

    def observed(self, path: Path) -> Observation | None:
        return self.entries.get(str(path))

    def forget(self, path: Path) -> None:
        self.entries.pop(str(path), None)

    def stale_reason(self, path: Path, current: str) -> str | None:
        """Why an edit to *path* cannot be trusted, or ``None`` if it can.

        The messages are written to be acted on rather than merely understood:
        each one names the tool call that fixes it, because the reader is a
        model deciding what to do next and a diagnosis it cannot act on costs
        a turn.
        """
        observation = self.observed(path)
        if observation is None:
            return (
                f"You have not read {path.name} in this run. Call read_file on it first: "
                "editing a file whose current contents you have not seen risks silently "
                "discarding work."
            )
        if not observation.complete:
            return (
                f"You have only read part of {path.name}. Read it in full before editing, "
                "so the replacement is matched against the whole file."
            )
        if observation.digest != digest_of(current):
            return (
                f"{path.name} has changed on disk since you read it - a command you ran, "
                "or a process outside this run, has modified it. Call read_file again and "
                "base the edit on the current contents."
            )
        return None
