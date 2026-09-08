"""Command admission policy.

**A denylist is not a security boundary.** It cannot be, because the space of
dangerous commands is unbounded and shell quoting is adversarially flexible.
Anything that treats a pattern list as containment is wrong, and it is worth
saying so in the code rather than only in a threat model nobody reads.

What this *is* worth: an agent that reaches for ``sudo``, pipes a download into
a shell, or force-pushes to a remote is almost always mistaken rather than
malicious, and stopping it costs one turn instead of an incident. The real
boundary is the layer below - path containment, resource limits, no ambient
credentials, and container isolation when the work is untrusted.

Two modes. Denylist is the default for a developer workspace, where the useful
command set is open-ended. Allowlist is available for anything closer to
production, where it is enumerable and a deny-by-default posture is affordable.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

#: Tokens that only mean anything to a shell. Checked *after* parsing, as whole
#: tokens, which is the difference between correct and merely strict.
#:
#: Scanning the raw string for these characters looks safer and is wrong: it
#: rejects `python -c "a; b"` and `sed 's/x/y/;s/a/b/'`, where the character is
#: inside a quoted argument. And because nothing here ever runs a shell, a
#: metacharacter inside an argument is inert - `echo 'a;rm -rf .'` prints a
#: string. What is worth refusing is a *bare* operator token, because it means
#: the model expected shell semantics and would otherwise get the operator
#: passed through as a literal argument, silently doing the wrong thing.
SHELL_OPERATORS: frozenset[str] = frozenset(
    {";", "&", "&&", "|", "||", ">", ">>", "<", "<<", "<<<", "2>", "2>&1", "|&"}
)

#: Command substitution anywhere in a token is refused: unlike an operator it
#: is not inert-looking to a reader, and a model writing it is always assuming
#: a shell that is not there.
SUBSTITUTION = re.compile(r"\$\(|`")


class PolicyMode(StrEnum):
    DENYLIST = "denylist"
    ALLOWLIST = "allowlist"


@dataclass(frozen=True)
class Rule:
    """One named pattern, matched against the normalised command line."""

    name: str
    pattern: re.Pattern[str]
    reason: str

    @classmethod
    def of(cls, name: str, pattern: str, reason: str) -> Rule:
        return cls(name=name, pattern=re.compile(pattern, re.IGNORECASE), reason=reason)


@dataclass(frozen=True)
class Decision:
    allowed: bool
    rule: str = ""
    reason: str = ""
    argv: tuple[str, ...] = ()

    @classmethod
    def allow(cls, argv: Sequence[str]) -> Decision:
        return cls(allowed=True, rule="default", argv=tuple(argv))

    @classmethod
    def deny(cls, rule: str, reason: str, argv: Sequence[str] = ()) -> Decision:
        return cls(allowed=False, rule=rule, reason=reason, argv=tuple(argv))


#: Ordered rules covering the mistakes an agent actually makes. Each exists
#: because the consequence is irreversible, reaches outside the workspace, or
#: hands execution to content fetched at runtime.
DEFAULT_RULES: tuple[Rule, ...] = (
    Rule.of(
        "privilege-escalation",
        r"^\s*(sudo|doas|su|pkexec)\b",
        "elevating privileges is never part of a workspace task",
    ),
    Rule.of(
        "remote-code-execution",
        r"\b(curl|wget|fetch)\b.*\|\s*(ba|z|k)?sh\b",
        "piping a download into a shell executes code fetched at runtime",
    ),
    Rule.of(
        "decoded-execution",
        r"\bbase64\b.*\|\s*(ba|z|k)?sh\b",
        "decoding straight into a shell hides what is being run",
    ),
    Rule.of(
        "recursive-root-delete",
        r"\brm\b.*-[a-z]*[rR][a-z]*f?\b.*\s(/|~|\$HOME)\s*$",
        "recursive delete of a filesystem or home root",
    ),
    Rule.of(
        "disk-write",
        r"^\s*(dd|mkfs|fdisk|parted|shred)\b",
        "writing to block devices can destroy the host",
    ),
    Rule.of("fork-bomb", r":\s*\(\s*\)\s*\{.*\|.*&.*\}\s*;?\s*:", "fork bomb"),
    Rule.of(
        "permission-widening", r"\bchmod\b.*\b(777|a\+rwx|o\+w)\b", "making files world-writable"
    ),
    Rule.of(
        "history-rewrite",
        r"\bgit\b.*\bpush\b.*(--force\b|-f\b|\+)",
        "force-pushing rewrites history other people depend on",
    ),
    Rule.of(
        "remote-mutation",
        r"\bgit\b.*\b(push|remote\s+set-url)\b",
        "changing a remote reaches outside the workspace",
    ),
    Rule.of(
        "credential-access",
        r"(\.ssh/|\.aws/|\.kube/config|\.netrc|id_rsa|id_ed25519)",
        "reading or writing credential material",
    ),
    Rule.of(
        "system-configuration",
        r"^\s*\S*\b(systemctl|service|launchctl|crontab)\b",
        "changing system services outlives the run",
    ),
    Rule.of(
        "remote-shell",
        r"^\s*(ssh|scp|rsync|telnet|nc|ncat|netcat)\b",
        "opening a connection to another host",
    ),
    Rule.of(
        "package-install",
        r"\b(apt-get|apt|yum|dnf|brew|apk)\b\s+(install|remove|purge)",
        "installing system packages changes the machine, not the workspace",
    ),
    Rule.of("etc-write", r">\s*/etc/|\btee\b\s+/etc/", "writing to /etc"),
)

#: A conservative starting point for allowlist mode.
DEFAULT_ALLOWED_BINARIES: frozenset[str] = frozenset(
    {
        "cat",
        "head",
        "tail",
        "ls",
        "find",
        "grep",
        "rg",
        "wc",
        "diff",
        "file",
        "stat",
        "python",
        "python3",
        "pytest",
        "ruff",
        "mypy",
        "black",
        "node",
        "npm",
        "npx",
        "pnpm",
        "yarn",
        "go",
        "cargo",
        "rustc",
        "make",
        "git",
        "echo",
        "true",
        "false",
        "pwd",
        "sort",
        "uniq",
        "sed",
        "awk",
        "cut",
        "tr",
    }
)


class CommandPolicy:
    """Decides whether a command may run."""

    def __init__(
        self,
        mode: PolicyMode | str = PolicyMode.DENYLIST,
        rules: Sequence[Rule] = DEFAULT_RULES,
        allowed_binaries: frozenset[str] | set[str] = DEFAULT_ALLOWED_BINARIES,
        allow_shell_operators: bool = False,
        extra_denied_binaries: frozenset[str] | set[str] = frozenset(),
    ) -> None:
        self.mode = PolicyMode(mode)
        self.rules = tuple(rules)
        self.allowed_binaries = frozenset(allowed_binaries)
        self.denied_binaries = frozenset(extra_denied_binaries)
        self.allow_shell_operators = allow_shell_operators

    @classmethod
    def allowlist(cls, binaries: Sequence[str], **kwargs: Any) -> CommandPolicy:
        return cls(mode=PolicyMode.ALLOWLIST, allowed_binaries=frozenset(binaries), **kwargs)

    # -- parsing ---------------------------------------------------------
    @staticmethod
    def parse(command: str | Sequence[str]) -> list[str]:
        """Split a command into argv without ever invoking a shell.

        ``shlex.split`` rather than ``shell=True``: the command comes from a
        language model, and handing model output to a shell is the single
        largest foot-gun available in this codebase.
        """
        if isinstance(command, str):
            try:
                return shlex.split(command)
            except ValueError as exc:
                raise ValueError(f"could not parse the command: {exc}") from exc
        return [str(part) for part in command]

    # -- decision --------------------------------------------------------
    def check(self, command: str | Sequence[str]) -> Decision:
        try:
            argv = self.parse(command)
        except ValueError as exc:
            return Decision.deny("unparseable", str(exc))
        if not argv:
            return Decision.deny("empty", "no command was given")

        if not self.allow_shell_operators:
            operator = next(
                (token for token in argv if token in SHELL_OPERATORS or SUBSTITUTION.search(token)),
                None,
            )
            if operator is not None:
                return Decision.deny(
                    "shell-operators",
                    f"{operator!r} needs a shell, and commands run as argv without one; "
                    "issue one command per call",
                    argv,
                )

        # Rules match the parsed tokens joined by single spaces, not
        # shlex.join: re-quoting would insert quote characters the patterns
        # would then have to know about, so `curl x | sh` would become
        # `curl x '|' sh` and stop matching its own rule.
        text = " ".join(argv)

        binary = argv[0].rsplit("/", 1)[-1]
        if binary in self.denied_binaries:
            return Decision.deny("denied-binary", f"{binary!r} is not permitted", argv)

        if self.mode is PolicyMode.ALLOWLIST and binary not in self.allowed_binaries:
            return Decision.deny(
                "not-allowlisted",
                f"{binary!r} is not in the allowlist for this run",
                argv,
            )

        for rule in self.rules:
            if rule.pattern.search(text):
                return Decision.deny(rule.name, rule.reason, argv)
        return Decision.allow(argv)

    def describe(self) -> dict[str, Any]:
        return {
            "mode": str(self.mode),
            "rules": [{"name": r.name, "reason": r.reason} for r in self.rules],
            "allowed_binaries": sorted(self.allowed_binaries)
            if self.mode is PolicyMode.ALLOWLIST
            else [],
            "allow_shell_operators": self.allow_shell_operators,
            "note": (
                "A denylist is a guardrail against plausible mistakes, not a security "
                "boundary. Containment comes from the path jail, resource limits, the "
                "scrubbed environment and container isolation."
            ),
        }


@dataclass
class PolicyCounters:
    """Denials by rule, so the dashboard can show what agents actually try."""

    denials: dict[str, int] = field(default_factory=dict)

    def record(self, decision: Decision) -> None:
        if not decision.allowed:
            self.denials[decision.rule] = self.denials.get(decision.rule, 0) + 1
