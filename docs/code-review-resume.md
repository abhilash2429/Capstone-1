# Gantry code review — resume prompt

Paste the block under **Prompt to paste** into a fresh session. Everything above it
is the state a previous session established, so the next run does not repeat it.

---

## Where the review got to

**Target:** `gantry` agent harness, `abhilash2429/Capstone-1`, at `main` = `6004219`
(the merge of PR #1). ~13k lines: 8,509 in `src/`, 4,458 in `tests/`.
Review branch: `claude/code-review-orchestration-0n5r99`.

**Baseline, measured, not assumed:**

| Check | Result |
|---|---|
| `pytest -m "not live"` | 387 passed, 2 skipped (the `live` Azure tests), 0 failed |
| `ruff check src tests` | clean |
| `ruff format --check` | clean, 57 files |
| Coverage | 94% (3,743 statements, 243 missed) |

The suite being fully green is the point: every finding below is a defect the
existing tests do not catch.

**Lowest-coverage files, which correlate badly with where the risk is:**

| File | Cover | Uncovered |
|---|---|---|
| `sandbox/_limits.py` | 71% | 30-33, 53, 62-70, 83-93 — the whole rlimit-application path |
| `rpc/transport.py` | 78% | 107-159 — stdio framing edges: partial reads, split messages, EOF |
| `config.py` | 81% | 30-43, 56-63 |
| `telemetry/store.py` | 82% | 196-210 |
| `otlp.py` | 86% | 34-36, 62-64, 109-113 |
| `sandbox/runner.py` | 88% | 345-358, 417-430 |

## Findings confirmed so far

Both were reproduced by execution, not inferred from reading.

### 1. Resource limits are bypassable by the sandboxed command — `sandbox/_limits.py:36-37`

`_set()` clamps the soft limit but passes the *original* `hard` straight back to
`setrlimit(limit, (value, hard))`. The hard ceiling is left untouched, normally
`RLIM_INFINITY`, so the command being sandboxed raises its own soft limit back in
two lines:

```python
soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
resource.setrlimit(resource.RLIMIT_CPU, (hard, hard))   # back to (INFINITY, INFINITY)
```

Reproduced for `RLIMIT_CPU` — `(5, INFINITY)` restored to `(INFINITY, INFINITY)` —
and for `RLIMIT_CORE`, where `_set(CORE, 0)` at line 73 is undone the same way and
core dumps become writable again. `RLIMIT_AS` (line 64), `RLIMIT_FSIZE` (66),
`RLIMIT_NPROC` (68) and `RLIMIT_NOFILE` (70) all share the flaw.

This contradicts the file's own comments. Line 60-61 claims "a hard CPU ceiling
that survives a command ignoring SIGTERM"; it is a soft limit. Line 71-72 claims
core dumps are never written.

Fix: pass `(value, value)` so the hard limit drops too. Lowering a hard limit is
irreversible for an unprivileged process, which is exactly the property a sandbox
wants. Consider whether the CPU limit should still be recoverable for legitimate
long builds before applying it to every limit.

Note this is *not* the same as the deliberate design choice documented at lines
34-35 (a platform rejecting a limit must not block the command). That fallback is
sound. This is the hard value being passed through on the success path.

### 2. `PathJail.open()` destroys data in any `+` mode — `sandbox/jail.py:158,175`

`writing = any(flag in mode for flag in ("w", "a", "x", "+"))` makes `"r+"` a
write, and it then falls into `elif writing:` at 175 and receives
`O_WRONLY | O_CREAT | O_TRUNC`.

Reproduced: a file containing `IMPORTANT EXISTING CONTENT` opened `"r+"` is empty
immediately afterwards, and the returned handle's `read()` raises
`OSError [Errno 9] Bad file descriptor` — the fd is write-only while the
`TextIOWrapper` claims mode `r+`.

Severity is honest-latent: no in-repo caller passes a `+` mode. Callers use `"r"`,
`"w"` (jail.py:197,202) and `"rb"`, `"wb"` (toolkit/files.py:160,170). It is a
landmine in a security-boundary API, not a live data-loss path today.

Fix: map `r+`/`w+`/`a+` to `O_RDWR` and only set `O_TRUNC` for `w`/`w+`, or reject
`+` modes explicitly if read-write is not meant to be supported.

## What has NOT been reviewed yet

- `sandbox/policy.py` — read, not yet analysed. Specific things to chase: the
  `credential-access` rule at line 127 matches on the joined text so it also fires
  on innocent paths containing `id_rsa`; `recursive-root-delete` at 103 anchors on
  `\s(/|~|\$HOME)\s*$` so `rm -rf / --no-preserve-root` and `rm -rf /*` both miss;
  `binary = argv[0].rsplit("/", 1)[-1]` at 257 means `./sudo` or a shim named
  `python` in the workspace is judged by basename alone. Note the module docstring
  already disclaims being a security boundary, so weigh findings against that
  stated intent rather than treating every gap as a bug.
- `sandbox/runner.py` (464 lines) — subprocess lifecycle, timeout and reaping,
  pipe-buffer deadlock, output caps, env scrubbing.
- `loop.py`, `verify.py`, `contract.py`, `budget.py` — termination, budget
  accounting at boundaries, whether a gate can report pass when it should fail.
- `providers/` and `rpc/` — JSON-RPC id correlation, Content-Length framing across
  read boundaries, UTF-8 split mid-character, retry of non-idempotent calls, cache
  key completeness, secrets reaching logs or cache files.
- `toolkit/` — encoding assumptions, unbounded reads, ReDoS from model-supplied
  regex in `search.py`, whether every path routes through the jail.
- `telemetry/` — SQL parameterisation, `check_same_thread`, spans left open on
  exception paths, unknown-model pricing lookup, telemetry failure crashing a run.
- `dispatch.py`, `tools/`, `cli/`, `config.py`, `errors.py` — argument validation
  before handler dispatch, `--grant read-only` enforced at dispatch rather than
  only at listing, config precedence, exit codes.

## Two operational notes for the next session

1. **Codex CLI is not reachable from a Claude Code cloud session.** It is installed
   on the local machine; the cloud session runs in an isolated container where
   `find / -name "codex*" -type f` returns nothing. To use the `codex-worker`
   skill with `gpt-5.6-sol` / `gpt-5.6-luna`, run Claude Code locally. In the cloud,
   the same heavy/light split has to map onto Claude subagents instead.
2. **Do not fan out six subagents at once.** A previous session launched six in
   parallel (three on opus) and all six died instantly on a 429 session limit,
   three before reading a single file. Run two or three at a time, or review on the
   main thread, which is cheaper because it does not re-derive context cold.

## Suggested partition, by risk rather than line count

| Tier | Unit | Lines |
|---|---|---|
| Heavy | `sandbox/` — runner, policy, jail, limits | 1,076 |
| Heavy | `loop.py` + `verify.py` + `contract.py` + `budget.py` | 1,490 |
| Heavy | `providers/` + `rpc/` | 1,663 |
| Light | `toolkit/` | 1,118 |
| Light | `telemetry/` | 1,120 |
| Light | `tools/` + `dispatch.py` + `cli/` + `messages.py` + `config.py` + `errors.py` | 1,542 |

---

## Prompt to paste

> Resume the code review of the `gantry` harness in this repo. Read
> `docs/code-review-resume.md` first — it records the baseline, the two findings
> already confirmed, and exactly which subsystems remain unreviewed. Do not redo
> the finished work.
>
> Work through the unreviewed subsystems in the order listed there, heavy tier
> first, starting with `sandbox/runner.py` and `sandbox/policy.py`. For each one,
> invoke the `code-review` skill on that path (`max` effort for the heavy tier,
> `high` for the light tier), then go beyond it with the adversarial checklist
> given for that subsystem in the doc.
>
> Rules I care about:
> - Reproduce every finding by executing it before reporting it. The suite is
>   green and ruff is clean, so anything real is something the tests miss — prove
>   it with a script under the scratchpad, the way the two existing findings were
>   proved.
> - No style opinions and no unverified speculation. Each finding needs
>   `file:line`, one sentence on the defect, and a concrete failure scenario with
>   specific input and resulting wrong behavior.
> - Weigh findings against stated intent. `policy.py` openly says a denylist is
>   not a security boundary; do not report that as a discovery.
> - Say plainly when a category checks out clean, so I know it was actually looked
>   at rather than skipped.
> - Read-only until I say otherwise. Do not fix, commit, or push without asking.
>
> Append what you find to `docs/code-review-resume.md` under the confirmed
> findings, and move each subsystem you finish out of the "not reviewed" list, so
> the doc stays accurate if this session dies too.
