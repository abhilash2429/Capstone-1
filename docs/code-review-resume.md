# Gantry code review — resume prompt

Paste the block under **Prompt to paste** into a fresh session. Everything above it
is the state a previous session established, so the next run does not repeat it.

---

## Where the review got to

**Target:** `gantry` agent harness, `abhilash2429/Capstone-1`, at `main` = `6004219`
(the merge of PR #1). ~13k lines: 8,509 in `src/`, 4,458 in `tests/`.
Review branches: `claude/code-review-orchestration-0n5r99` (session 1),
`claude/happy-ramanujan-jqn0yn` (session 2).

**Baseline, measured, not assumed. Re-measured at the start of session 2 and
unchanged:**

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

Every finding below was reproduced by execution, not inferred from reading.

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

---

## Session 2 findings — `sandbox/runner.py`

### 3. Writing stdin before `wait()` deadlocks the caller, so `timeout_s` never applies — `sandbox/runner.py:306-311`

`process.stdin.write(...)` runs on the main thread before
`process.wait(timeout=timeout)`, so a payload larger than the pipe buffer blocks
against a child that is not reading, and the timeout clock is never started.

Reproduced with `timeout_s=2.0`, a 1 MiB stdin payload and a child that sleeps 25s
without reading stdin:

```
observed wall clock : 25.08 s
result.duration_ms  : 25076
result.timed_out    : False
result.exit_code    : 0
result.killed_group : False

control: same command, small stdin
observed wall clock : 2.00 s  timed_out=True
```

The 2-second ceiling was ignored and the result claims a clean success. Runtime is
bounded only by the child's own lifetime, so a child that never exits hangs the
harness forever. Reachable through `SandboxRunner.run(..., stdin=...)`;
`toolkit/shell.py` passes no stdin, so it is not reachable from the `bash` tool
today.

Fix: feed stdin from a third thread so the main thread reaches
`wait(timeout=...)` immediately.

### 4. `_drain` publishes captured output only in its `finally`, so output is discarded whenever the reader is still blocked at join-timeout — `sandbox/runner.py:141-143`, with `320-322`

Bytes accumulate in the thread-local `kept` and reach `sink` only after the read
loop ends. If any process inherited the stdout pipe and outlives the direct child,
the reader is still in `stream.read()` at `reader.join(timeout=5)` and
`sink.get("stdout", "")` returns the empty string — not partial output, all of it.

Reproduced with a child that prints, then spawns a background process holding the
pipe:

```
wall            : 10.12 s   (parent exits immediately)
exit_code       : 0
timed_out       : False
stdout captured : ''
truncated       : False

control: same print, no background holder
stdout captured : 'IMPORTANT BUILD OUTPUT\n'
```

The child exits in milliseconds and `wait()` returns at once; the two joins then
burn 10s and the already-buffered output is thrown away with `truncated=False`.
This is the shape of `pytest` leaving a fixture server running, `npm run dev`, or
any `Popen` in a test teardown: the agent is told the command produced no output.

Fix: write into a shared buffer the caller can read rather than only in the
`finally`, and close the parent's copies of the pipe fds.

### 5. No `try/finally` around the child, so a non-`OSError` exception after `Popen` leaks a running process — `sandbox/runner.py:306-311`

`except OSError` catches broken pipes but not encoding errors.
`stdin.encode()` raising `UnicodeEncodeError` (a `ValueError`) propagates out of
`_run_subprocess` with the child alive, never waited on and never group-killed.

Reproduced with `stdin="\ud800"` and `timeout_s=2.0`:

```
run() RAISED UnicodeEncodeError: 'utf-8' codec can't encode character '\ud800'
  in position 0: surrogates not allowed
child processes of the harness after run() aborted:
  PID STAT ELAPSED COMMAND
 3131 S          1 .venv/bin/python -c import time; time.sleep(45)
```

The 45-second child is still running and unowned. `UnicodeEncodeError` is not a
`GantryError`, so `dispatch.py:325` does not classify it either.

Fix: wrap everything after `Popen` in a `try/finally` that calls `_kill_group` and
`wait()` on every exit path.

### 6. Container-mode timeout kills the `docker` client only; the containerised workload survives — `sandbox/runner.py:404-426`

`subprocess.run(..., timeout=timeout)` is used with no `process_group`, no
`--name` and no `docker kill` on the timeout path, so `TimeoutExpired` reaps the
CLI and leaves the container running. This is the exact failure the module
docstring says the subprocess path was given `process_group=0` to avoid.

Executed against the real `_run_container` branch with a stub `docker` first on
`PATH` (`_run_container` calls `subprocess.run` without `env=`, so it resolves
through the parent's PATH). The stub `setsid`s its workload, mirroring the real
topology where the container is a child of dockerd, not of the client:

```
container timeout: wall=2.00s exit=124 timed_out=True killed_group=False
'container' process 3104 after timeout -> ps stat 'Ss'  (S/R = STILL RUNNING)
```

Caveat stated plainly: no dockerd is available in this environment, so the
executed repro proves the `subprocess.run`-timeout semantics that `runner.py`
relies on, not a real container teardown.

Fix: pass `--name gantry-<uuid>` and run `docker kill` on `TimeoutExpired`, or use
`--cidfile` plus `docker rm -f`.

### 7. Container mode applies `max_output_bytes` after buffering the whole stream, and applies it as a character count — `sandbox/runner.py:428-435`

`capture_output=True, text=True` materialises the full output before
`stdout[:cap]`, and `cap` is a byte budget sliced against a `str`.

```
container stub emitted 256 MiB; cap=1024; kept=1024 chars
harness RSS before=20 MiB  peak=789 MiB  growth=769 MiB

stub emits 200k 'é': cap(max_output_bytes)=1024 -> len(stdout)=1024 chars
                     = 2048 bytes, truncated=True
```

769 MiB of harness RSS for a 1 KiB cap, and the "1024-byte" cap yields 2048 bytes
(up to 4096 for 4-byte code points). The subprocess path, checked separately and
clean, keeps peak allocation at 0.1 MB for the same 64 MiB of output. The two
paths do not honour the same contract.

Fix: run the container branch through `Popen` and the same `_drain` readers as the
subprocess branch.

## Session 2 findings — `sandbox/policy.py`

`policy.py`'s module docstring openly disclaims being a security boundary. The
findings below are not "a denylist is bypassable"; each is a rule failing at the
job its own name states.

### 8. `recursive-root-delete` misses every common destructive form — `sandbox/policy.py:101-105`

The pattern anchors on `\s(/|~|\$HOME)\s*$`, so the root target has to be the
final token.

```
DENY  rule=recursive-root-delete  :: 'rm -rf /'
ALLOW rule=default                :: 'rm -rf / --no-preserve-root'
ALLOW rule=default                :: 'rm -rf /*'
ALLOW rule=default                :: 'rm -rf ~/'
ALLOW rule=default                :: 'rm -rf $HOME/'
ALLOW rule=default                :: 'rm -rf / -v'
```

The one case the rule catches, `rm -rf /`, is the only form GNU `rm` already
refuses on its own. Every form that actually deletes passes.

Fix: match the root target as a token —
`(^|\s)(/|~|/\*|~/|\$HOME)(/\*)?(\s|$)` — rather than requiring end-of-string.

### 9. `credential-access` refuses ordinary workspace paths — `sandbox/policy.py:125-129`

`id_rsa` and `id_ed25519` are matched as bare unanchored substrings of the joined
argv, including inside longer identifiers.

```
DENY  rule=credential-access :: 'cat tests/fixtures/id_rsa_parser_test.txt'
DENY  rule=credential-access :: 'python analyze_grid_rsa.py'   <- 'grid_rsa' contains 'id_rsa'
DENY  rule=credential-access :: 'grep -rn id_rsa docs/'
DENY  rule=credential-access :: 'pytest tests/test_id_rsa_fingerprint.py'
DENY  rule=credential-access :: 'ruff check src/valid_rsa.py'  <- 'valid_rsa' contains 'id_rsa'
```

Any repository with an SSH-key parser, a fingerprint fixture, or a variable named
`valid_rsa` has those files permanently unreadable and untestable by the agent,
and the denial message says "reading or writing credential material". The jail is
not a mitigation here: the rule fires on workspace-relative paths.

Fix: anchor to a path component —
`(^|[\s/])(id_rsa|id_ed25519)(\.pub)?($|\s)` — and require `~/` or an absolute
prefix for the `.ssh/` and `.aws/` forms.

### 10. Name-anchored rules are defeated by a path-qualified invocation, and the allowlist judges by basename — `sandbox/policy.py:86-90, 106-110, 135-139, 257`

Three rules match the start of the joined text, which is `argv[0]` verbatim, so a
path prefix bypasses them. `system-configuration` at line 132 uses
`^\s*\S*\b(...)` and does handle it, so the inconsistency is internal to one rule
table.

```
DENY  rule=privilege-escalation :: 'sudo rm -rf /home/user/x'
ALLOW rule=default              :: '/usr/bin/sudo rm -rf /home/user/x'
ALLOW rule=default              :: './sudo rm -rf /home/user/x'
DENY  rule=remote-shell         :: 'ssh host'
ALLOW rule=default              :: '/usr/bin/ssh host'
DENY  rule=disk-write           :: 'dd if=/dev/zero of=/dev/sda'
ALLOW rule=default              :: '/bin/dd if=/dev/zero of=/dev/sda'
DENY  rule=system-configuration :: '/usr/bin/systemctl stop x'   <- \S* prefix, handled
```

The mirror image in allowlist mode, where `binary = argv[0].rsplit("/", 1)[-1]`:

```
ALLOW rule=default          :: './python -V'
ALLOW rule=default          :: '/tmp/evil/python -V'
DENY  rule=not-allowlisted  :: 'sh -c x'
```

A shim the agent wrote itself at `./python` is admitted as the allowlisted
`python`.

Fix: match the name-anchored rules against the already-computed `binary`, and
resolve `argv[0]` against `PATH` before the allowlist comparison.

### 11. Quadratic backtracking in `recursive-root-delete`, evaluated before any timeout applies — `sandbox/policy.py:103`

`\brm\b.*-[a-z]*[rR][a-z]*f?\b.*\s(/|~|\$HOME)\s*$` has two `.*` plus
`[a-z]*[rR][a-z]*`. `policy.check()` runs at `runner.py:231`, outside every span
and timeout.

Input `"rm " + "-arf "*n + "z"`:

```
  len=   2004      10.25 ms
  len=   8004     146.48 ms
  len=  16004     580.18 ms
  len=  32004      2.276 s
  len=  64004      9.084 s
  len= 128004     36.969 s
realistic 'rm -rf <2000 files>' len=56896 -> 0.030 s allowed=True
```

Clean quadratic scaling. A 128 KB model-emitted command burns 37 seconds of CPU on
the dispatcher thread before the command is even admitted. Benign long commands are
unaffected, so this needs a degenerate string — but that string is model-supplied
and length-unbounded.

Fix: cap the length of `text` before rule matching, and match the flag token
directly instead of `[a-z]*[rR][a-z]*f?`.

## Session 2 findings — `loop.py`, `verify.py`, `contract.py`, `budget.py`

### 12. The loop crashes with `AttributeError` on the first elision whenever tracing is off, which is the library default — `loop.py:262`

`self.tracer.current_span().set_attributes(...)` is unguarded. `Tracer.span()`
does not set the `_current_span` contextvar when the tracer is disabled, and the
process-wide tracer is `Tracer(exporter=None, enabled=False)` until `configure`
runs (`tracer.py:365`). Every test constructs the Agent with the enabled `tracer`
fixture, which is why nothing catches it. `_span_cost()` at `loop.py:448` has the
`if span else` guard; line 262 does not.

Reproduced — the same run twice, only the tracer differs:

```
process-wide tracer enabled? False
default tracer (library default): RUN CRASHED AttributeError:
  'NoneType' object has no attribute 'set_attributes'  at loop.py:262
  (after 2 provider calls)
explicit enabled tracer  : stop=no_progress turns=4 elisions=2
```

Input: `Agent(provider=OfflineProvider(...12 turns calling a tool returning
3.2KB...), budget=ObservationBudget(max_context_tokens=2400,
reserve_output_tokens=400))` with no `tracer=` argument. No `AgentResult` is
produced at all; the run and the money spent on it are lost.

Fix: `span = self.tracer.current_span(); if span: span.set_attributes(...)`.

### 13. `_span_cost()` reads cost off the wrong span, so `usage.cost_usd` is permanently 0.00 and `StopReason.COST_BUDGET` is unreachable — `loop.py:441-448`

The provider writes `gantry.cost.usd` on its own `llm_span`, which has already
exited by the time `_complete` returns. `get_tracer().current_span()` is then the
enclosing `agent.run` span, which never carries that attribute. `_span_cost` also
uses the global tracer, not `self.tracer`.

Reproduced with model `gpt-4o` and `BudgetConfig(max_cost_usd=0.01)`:

```
stop_reason: token_budget | consumed 410649 of 400000 tokens
loop usage : {..., 'cost_usd': 0.0, 'provider_calls': 9}
trace cost : 1.6341225000000001
llm span costs: [0.091213, 0.1138, 0.13639, 0.15898, 0.18157, 0.204157,
                 0.226747, 0.249337, 0.271927]
```

$1.63 spent against a $0.01 ceiling, reported as $0.00. The docstring's promise
that "the loop's running total and the trace can never disagree" is inverted.
`tests/test_loop.py:250` only exercises `max_cost_usd=0.0`, which trips on
`0.0 >= 0.0` at the first `before_step` before anything is recorded, so it passes
whether or not cost is ever accumulated. The same zero is written to the run store
at `loop.py:425`.

Fix: return the cost from the completion span the provider just produced (e.g. on
the `Completion` object), not from whatever span happens to be current.

### 14. `_finish_arguments` rewrites every malformed completion claim as `status="completed"`; `FINISH_SCHEMA` is never enforced — `loop.py:390-395`

The loop intercepts `finish` before dispatch, so the registry validator that would
reject these is never asked. A real passing gate is installed in every case below:

```
truncated JSON (mid tool call)  -> stop=completed succeeded=True summary=''
invalid JSON                    -> stop=completed succeeded=True summary=''
typo'd enum value 'refuse'      -> stop=completed succeeded=True summary='I decline'
null status                     -> stop=completed succeeded=True summary='x'
wrong type (int)                -> stop=completed succeeded=True summary=''
missing required fields         -> stop=completed succeeded=True summary=''
extra unexpected field          -> stop=completed succeeded=True summary='s'

schema REJECTS {'status': 'refuse', ...} -> ToolValidationError: 'refuse' is not
  one of ['completed', ...]
schema REJECTS {'status': None, ...}     -> ToolValidationError: None is not of
  type 'string'
schema REJECTS {}                        -> ToolValidationError: 'status' is a
  required property
```

The registry's own validator, given the very schema `loop.py:51` registers,
rejects all four. Worst case: a reply cut off mid-tool-call (`finish_reason=LENGTH`,
which `Completion.truncated` exists to flag and the loop never reads) is recorded
as a successful completion, and a refusal typo'd as `"refuse"` is recorded as
`COMPLETED` / `succeeded=True`.

Fix: run `registry.validate_arguments("finish", args)` on the parsed claim and
feed an unparseable or invalid claim back to the model as a tool error.

### 15. A gate that raises kills the entire run — `verify.py:331`, with `verify.py:221-224`

`GateSet.run` does not guard `gate.check`, and `PythonSyntaxGate` catches only
`SyntaxError`, `OSError` and `UnicodeDecodeError`, while `ast.parse` also raises
`RecursionError` on agent-authored content.

Reproduced from the model's own turn
`write_file(path="gen.py", content="x = a" + ".b"*20000)` followed by `finish`:

```
RUN DIED WITH: RecursionError maximum recursion depth exceeded during
  ast construction
    loop.py 319 run
    loop.py 398 _verify
    verify.py 331 run
    verify.py 220 check
    ast.py 50 parse
```

The file is legal Python; the parse is what dies. `Verification.inconclusive`
exists for exactly this ("a gate that could not run at all", `verify.py:62`) and is
unreachable because the exception escapes first.

Fix: wrap `gate.check(context)` in `try/except Exception` in `GateSet.run` and
record `inconclusive=True`; widen `PythonSyntaxGate` to
`except (SyntaxError, ValueError, RecursionError, MemoryError)`.

### 16. An inconclusive gate (a broken harness) is charged to the agent as `VERIFICATION_FAILED` — `loop.py:280,328`

`GateReport.inconclusive` (`verify.py:275-277`, the uncovered line) has no consumer
in the loop.

```
stop_reason : verification_failed
detail      : checks still failing after 4 attempts: tests
provider calls burned: 4
inconclusive: ['tests'] | failures: ['tests']
as_dict: {'passed': False, 'checks': [{... 'inconclusive': True,
  'detail': 'no sandbox runner is available to execute this gate'}],
  'failed': ['tests']}
```

Same misattribution for a mistyped gate command:
`CommandGate("g", "definitely-not-a-real-command")` returns
`passed=False inconclusive=False detail='exit code 127, expected 0'`, so the model
is told its own work failed and spends four provider calls on a command that can
never exist. `verify.py:62-63` states this conflation must not happen.

Fix: branch on `gate_report.inconclusive` in `Agent.run` and stop with
`INTERNAL_ERROR`; mark exit 127 inconclusive in `CommandGate`.

### 17. `_split_finish` strips every `finish` call but answers only the first, leaving an unanswered tool-call id in the history — `loop.py:388`

```
two finish calls per turn + failing gate : stop=verification_failed requests=4
  UNANSWERED=[(1,'call_offline_0_1','finish'), (2,'call_offline_0_1','finish'),
              (2,'call_offline_1_1','finish'), (3, ...)]
one finish call per turn  + failing gate : stop=verification_failed requests=4
  UNANSWERED=[]
```

Input: an assistant turn emitting two `finish` calls plus a failing gate, so the
loop continues at `loop.py:327` instead of breaking. Requests 1-3 each carry
unanswered `tool_call_id`s — the exact protocol violation this repo names at
`dispatch.py:80-86` ("A provider rejects a turn where any tool-call id went
unanswered"). Against a real provider the run dies on the next request; the offline
provider hides it.

Fix: answer every stripped `finish` call with a tool message per id, or keep the
first and dispatch the rest normally.

### 18. `cap_observation` returns a string longer than the ceiling it was given — `budget.py:131-139`

The result is `limit + len(marker) + 2` characters, the precise failure its own
docstring at lines 133-136 says it must avoid, which stops `_last_resort`
converging.

```
cap_observation(limit=  50 tokens =   200 chars) ->  251 chars ( 63 tokens) +51
cap_observation(limit= 200 tokens =   800 chars) ->  851 chars (213 tokens) +51

available = 1000
fit -> tokens_before 18250 tokens_after 1750 fits False
per-message tokens after last resort: [29]
n shrinkable: 60 -> allowance the code derived: 12
```

Input: `ObservationBudget(max_context_tokens=1200, reserve_output_tokens=200,
keep_recent_turns=0)` with 60 shrinkable messages. The pass derives a 12-token
allowance; each message lands at 29 tokens (12 content + 4 envelope + 13 of
marker), totalling 1,750 against a 1,000-token window. Honouring the limit would
have given 60x16 + 10 = 970 tokens — it would have fit.

Fix: budget the marker inside the limit —
`head+tail = limit - len(marker) - 2` — falling back to the marker alone when that
is non-positive.

### 19. `Elision.fits` is never checked; the loop sends the oversized request anyway — `loop.py:258-262`

`budget.py:70-74` states plainly that "the caller has to know: sending the request
anyway earns a context-length 400". `grep -n fits src/gantry/loop.py` returns no
match.

Fix: stop the run with a typed reason when `not elision.fits`.

### 20. `can_afford`, the pre-spend reservation the module docstring promises, has no production caller — `contract.py:291`

`contract.py:19-20` states "A run terminates *before* exceeding its budget… so the
ceiling is a ceiling". `grep -rn can_afford src/ tests/` finds it defined at
`contract.py:291` and referenced only by `tests/test_contract.py:56-57`. The loop
records tokens and cost only after the call, so one call can blow the whole budget:

```
stop_reason: token_budget | consumed 410649 of 400000 tokens
```

The ceiling was exceeded by 10,649 tokens before the loop noticed.

Fix: call `can_afford`, and a token equivalent, with the projected prompt size in
`before_step` / `_complete`.

### 21. `GateContext.task` is always empty — `loop.py:399`

It is read from `ToolContext.metadata["task"]`, which the loop never populates,
while `task` is a parameter of `run()` in the same scope.

```
run task   : REBUILD THE PARSER AND MAKE TESTS PASS
gate saw   : {'task': '', 'changed': (), 'runner': None}
```

Any task-aware gate silently sees nothing.

Fix: `GateContext(jail=self.jail, runner=self.runner, task=state.task)`.

### 22. `record_provider_call` does not clamp usage, so a provider reporting negative tokens fabricates budget headroom — `contract.py:316-319`

```
after a negative-usage call: {..., 'input_tokens': -5000, 'total_tokens': -5000,
                              'cost_usd': -3.0, 'provider_calls': 1}
before_step now: True
```

Impact is bounded by `max_steps`, but reported usage and `remaining()` are wrong.

Fix: `max(0, ...)` on each addend.

## Session 2 findings — `rpc/`

### 23. `SubprocessTransport` hands the child every credential in the environment — `rpc/transport.py:133`

`env={**os.environ, **(env or {})}` gives an RPC child the full parent
environment, while `sandbox/runner.py:170` deliberately builds the child's
environment from an eight-name allowlist for exactly the reason its own comment at
lines 25-26 gives: "a subprocess inherits every credential in the parent's
environment by default". The same threat model is applied on one subprocess path
and not the other.

Reproduced. With `AZURE_OPENAI_API_KEY=sk-REAL-CREDENTIAL-abc123` and
`GITHUB_TOKEN=ghp-REAL-abc123` set in the parent, a child launched through
`SubprocessTransport` reported back:

```
child saw: {'GH_TOKEN': ..., 'GITHUB_TOKEN': 'ghp-REAL-abc123',
            'AWS_SECRET_ACCESS_KEY': ..., 'AWS_ACCESS_KEY_ID': ...,
            'CLAUDE_CODE_MESSAGING_TOKEN': ...,
            'AZURE_OPENAI_API_KEY': 'sk-REAL-CREDENTIAL-abc123'}
sandbox ENV_ALLOWLIST: ('PATH', 'LANG', 'LC_ALL', 'LC_CTYPE', 'TERM', 'TZ',
                        'PYTHONHASHSEED', 'PYTHONDONTWRITEBYTECODE')
```

The class docstring calls this "the same shape MCP servers use", so the peer is
third-party code by design.

Fix: build the child environment from `ENV_ALLOWLIST` the way `runner.py` does, or
take an explicit `env` with no `os.environ` base.

### 24. An undrained stderr pipe deadlocks any RPC child that logs — `rpc/transport.py:131`

`stderr=subprocess.PIPE` is opened but read only by `stderr_text()`, which nothing
calls during a session. A child that writes more than the pipe buffer (~64 KB) to
stderr blocks in `write()` before it ever answers on stdout.

Reproduced with a child that writes 200 KB to stderr and then serves normally:

```
FAILED after 8.0s: TransportError: timed out waiting for a reply to 'ping'
```

The same child with 1 KB of stderr answers immediately (`result: ok`), so the
volume is the cause, not the script. 64 KB is a low bar for a server that logs per
request. `stderr_text()` itself (line 157) is an unbounded `read()` to EOF, so
calling it on a live child blocks rather than draining.

Fix: drain stderr on a reader thread into a bounded buffer, or use
`stderr=subprocess.DEVNULL`.

### 25. One non-JSON line from the peer permanently kills the client — `rpc/client.py:50,53-56`

`p.decode(line)` runs inside the reader thread's `try`, and any exception exits the
loop and sets `_closed` forever. The loop already skips blank lines at line 48, so
tolerating noise is the established intent; a peer that prints a banner line to
stdout is not tolerated.

Reproduced with a transport serving `{"id":1,...}`, then `INFO: server started`,
then a valid `{"id":2,...}`:

```
call 1 -> first
call 2 FAILED: TransportError: connection closed before a reply arrived
call 3 FAILED: TransportError: connection is closed
valid reply for id 2 was already in the queue: True
```

The well-formed reply to call 2 was sitting in the queue and was never read, and
every future request on the client fails too. This is the failure mode
`transport.py:73-77` names ("a stray print … indistinguishable from a malformed
message") and defends against on the writing side only.

Fix: catch the decode error per line, report it, and continue; reserve termination
for EOF and transport errors.

### 26. A null `error` member reads as a successful null result — `rpc/protocol.py:186`, `rpc/client.py:80`

`parse_response` checks that exactly one of `result` and `error` is *present as a
key*, then `Response.is_error` tests the value. An explicit `"error": null`
satisfies the key check and then reports success.

```
parse_response({'jsonrpc':'2.0','id':7,'error':None})
  -> Response(id=7, result=None, error=None) | is_error = False | result = None
```

`RpcClient.call` at line 80 branches on `is_error`, so the caller receives `None`
as a normal result instead of a raised `JsonRpcError`.

Fix: treat a present-but-null `error` as an error reply, or reject it at parse
time alongside the both/neither case.

## Session 2 findings — `providers/` and `rpc/server.py`

### 27. The adapted-retry path never translates its exception, so a raw SDK error escapes `Provider.complete` — `providers/azure.py:152`

Line 145 wraps its call in `try/except Exception: raise self._translate(exc)`;
the adapted retry at 152 does not. Reproduced with a stub client that rejects
`temperature` with a 400 ("Unsupported value: 'temperature' is not supported with
this model") and then hits a 429 on the adapted retry:

```
LEAKED RAW SDK EXCEPTION: openai.RateLimitError -> rate limited
  is ProviderError? False
client calls: 2   retry sleeps performed: []
```

A retryable rate-limit is not retried (zero sleeps),
`metrics.PROVIDER_CALLS.inc(outcome="error")` and `span.set_status("error")` never
run, and `loop.py:266` (`except ProviderError`) does not catch it — the whole run
aborts instead of finishing with `StopReason.PROVIDER_ERROR`.

Fix: wrap line 152 in the same `try/except Exception: raise self._translate(exc)
from exc` as line 145.

### 28. `usage` is copied from the response with no validation, so a missing block silently charges zero — `providers/azure.py:255-263`

```
azure  usage: {'input_tokens': 0, 'output_tokens': 0, ...}
offline usage: {'input_tokens': 100, 'output_tokens': 8, ...}
after 50 azure calls with usage-less responses:
  total_tokens = 0  cost = 0.0 | token budget decision: True

usage with negative token counts: RAISED ValueError: counters cannot decrease
usage with string token counts:   RAISED TypeError: unsupported operand type(s)
                                  for -: 'str' and 'int'
```

`loop.py:369` charges the budget straight from `completion.usage`, so with
`max_tokens=1000` fifty real, billed calls advance the token and cost ceilings by
nothing; only `max_steps` still bounds the run. `Provider.estimate_usage` exists as
the documented fallback for exactly this and azure never uses it. The negative and
string cases raise untyped exceptions from inside telemetry, after the call was
already billed.

This is the third independent way the cost ceiling fails to hold, alongside
findings 13 and 39.

Fix: coerce to non-negative ints and fall back to `estimate_usage` when the
response reports no usage.

### 29. `choice.message` is read as a hard attribute while everything around it uses `getattr` — `providers/azure.py:253`

Response `Obj(choices=[Obj(finish_reason="stop")])`:

```
choice with no message at all:
  RAISED builtins.AttributeError: 'Obj' object has no attribute 'message'
```

Untyped, so no retry and not caught by `loop.py:266`.

Fix: `Message.from_wire(getattr(choice, "message", None) or {})`, or raise
`ProviderError`.

### 30. A negative `Retry-After` header crashes the retry loop — `providers/base.py:49-50`, with `azure.py:199`

The header value goes through `min()` unchanged into `self._sleep`.

```
delay_for(0, retry_after_s=-5) = -5.0
Retry-After='-5': RAISED builtins.ValueError: sleep length must be non-negative
                  | ProviderError? False
```

Headers `0` and `3600` behave correctly. A single malformed header from a proxy
turns a retryable rate-limit into an untyped crash.

Fix: `return max(0.0, min(retry_after_s, self.max_delay_s))`.

### 31. A handler result that is not JSON-serialisable kills `serve_forever` — `rpc/server.py:128,140,203-206`

`_handle_one` catches handler exceptions, but serialisation happens after that
guard. Handler `lambda params, ctx: {"seen": {"a","b"}}` (a set):

```
handle_line RAISED: TypeError -> Object of type set is not JSON serializable
--- serve_forever dies on it ---
serve_forever RAISED: TypeError -> Object of type set is not JSON serializable
lines still unread: ['{"jsonrpc":"2.0","id":2,"method":"ping"}'] | replies written: []
```

The peer gets no reply at all and the session dies; in a batch the sibling `ping`
reply is discarded too. This is the exact failure the module docstring says cannot
happen.

Fix: encode inside the try, or encode per-response and substitute an
`INTERNAL_ERROR` reply on `TypeError`.

### 32. A notification with malformed `params` is answered, breaking the never-answer-a-notification rule the code states at line 159 — `rpc/server.py:148-151`, with `protocol.py:163-164`

```
notification, good params            -> None
notification, params is a string     -> '{"jsonrpc":"2.0","id":null,"error":{"code":-32602,...}}'
batch of notifications, one bad      -> '[{"jsonrpc":"2.0","id":null,"error":{...}}]'
```

The message has no `id` member, so it is unambiguously a notification, and a batch
of only notifications stops being silent. `-32602` is also a post-dispatch code
being raised at parse time.

Fix: in `_handle_one`, return `None` when the failing payload is a dict with no
`"id"` key.

### 33. `$/cancelRequest` can never fire under `serve_forever`, the only I/O driver the module ships — `rpc/server.py:99-107,170-173,203-206`

The loop is strictly sequential. Peer sends request id 9, then a cancel for id 9:

```
handler saw cancellation: False
_inflight while the handler ran: [9]
_inflight when the cancel was finally read: []
replies written to the peer: '{"jsonrpc":"2.0","id":9,"result":"did all the work"}'
```

The cancel line is not read until the handler has already returned and the
`finally` popped the token, so `_handle_cancel` is a no-op and line 188 is
unreachable in production. `tests/test_rpc.py:186` passes only because it drives
`handle_line` from a second thread by hand; `tests/fixtures/stdio_server.py` uses
`serve_forever` and would not cancel.

Fix: read lines on a reader thread, or dispatch handlers off the read loop, so
control messages can be processed while a request is in flight.

### 34. A corrupt or forward-incompatible cassette raises a raw exception and is counted as a hit — `providers/cache.py:88`

`get()` counts the hit before parsing, and `_from_payload` is tolerant everywhere
(`.get` defaults, `FinishReason.parse` swallowing unknown values) except
`Usage(**...)`, which is strict.

```
extra usage field -> RAISED TypeError: Usage.__init__() got an unexpected keyword
                     argument 'audio_tokens' | ProviderError? False
corrupt cassette  -> RAISED json.decoder.JSONDecodeError: Unterminated string ...
                     | ProviderError? False
cache stats after corrupt read: {'hits': 2, 'misses': 0, 'hit_rate_pct': 100}
```

A cassette recorded by any version that later adds a usage field breaks replay of a
long-lived recording, with an untyped exception `loop.py:266` will not catch and
two bogus "hits" in the stats.

Fix: catch `(json.JSONDecodeError, TypeError, ValueError)` in `get()`, count a
miss, and filter unknown keys in `_from_payload`.

### 35. Cassettes are written world-readable — `providers/cache.py:96`

`mode=0o644 world-readable=True`, directory `0o755`, content
`{"message": {"role": "assistant", "content": "the secret token is sk-live-ABCDEF"} …}`.

The payload holds model output only — no API key, no Authorization header, no
prompt — so this is a lesser leak than finding 23, but any local user can read
every recorded model reply.

Fix: `os.chmod(temp, 0o600)` before `replace`, and create the directory 0700.

### 36. A cache key containing `../` escapes the cassette directory — `providers/cache.py:78`

The key is interpolated into a path with no validation.

```
ResponseCache(dir).put("../../../../tmp/pwned-cassette", …)
  -> wrote: True -> /tmp/pwned-cassette.json
```

`get()` on the same key reads it back. Latent only: every in-repo caller passes a
sha256 hexdigest, but `ResponseCache` is in `providers.__all__`.

Fix: reject keys that do not match `^[0-9a-f]{64}$`.

## Session 2 findings — `toolkit/`, `telemetry/`, `config.py`

### 37. `grep` runs a model-supplied regex with no complexity bound, and the tool timeout cannot stop it — `toolkit/search.py:72,109`

`re.compile(pattern)` accepts anything the model writes, and `dispatch.py:313-321`
enforces `timeout_s` by joining a daemon thread and abandoning it. A catastrophic
pattern therefore pins a core for the remaining life of the process, and each
further call adds another.

Reproduced. Pattern `(a+)+$` against a line of `a`s:

```
  len=20: 0.09s
  len=24: 1.51s
  len=26: 6.03s
  len=28: 24.30s
```

Doubling every character. Against a 40-character line the same call had not
returned after 10s (`alive=True`) and the interpreter had to be killed, against a
declared `timeout_s` of 45.0.

`dispatch.py:296-299` is honest that an abandoned handler runs on and points to
"process isolation, which is what the sandbox provides for the tools that can
genuinely hang" — but `grep` is an in-process `FS_READ` tool and never reaches the
sandbox, so the stated mitigation does not cover it. The input is a plain text
file; nothing about the workspace is hostile.

Fix: reject patterns with nested unbounded quantifiers, cap the line length fed to
`search`, or run `grep` behind the process isolation the docstring relies on.

### 38. `TelemetryStore(":memory:")` is not thread-safe, contradicting the class docstring — `telemetry/store.py:47,52-56`

Connections are thread-local, so with `:memory:` each thread gets its own separate,
empty database. The schema is created once on the constructing thread only. Line 47
special-cases `":memory:"`, so it is a supported mode.

```
write from another thread -> {'e': 'OperationalError: no such table: agent_runs'}
rows visible on main thread: 0
```

The class docstring says "Thread-safe handle onto a Gantry telemetry database".
The file-backed path is unaffected: a second thread opens a new connection to a
file that already carries the schema.

Fix: use a single shared connection guarded by the existing `_write_lock` for
`:memory:`, or use `file::memory:?cache=shared` with `uri=True`.

### 39. An unrecognised deployment name prices at $0.00 and nothing consumes the `priced=False` flag — `telemetry/pricing.py:135-137`

```
'gpt-4o'                -> Cost(usd=12.5, priced=True,  verified=True)
'totally-unknown-model' -> Cost(usd=0.0,  priced=False, verified=False)
```

`get()` falls back to the longest matching known *prefix*, which covers
`gpt-4o-mini-prod-eastus` but not an Azure deployment named `my-gpt-4o` or
`prod-main` — both normal naming. `grep -rn "priced" src/` shows the flag reaching
only a span attribute (`semconv.py:37`); no consumer in `contract.py`, `budget.py`
or `loop.py` reads it, so a cost ceiling silently never fires on an unpriced model.

This survives a fix to finding 13 and needs its own.

Fix: have the contract treat `priced=False` as a hard stop or a loud warning
rather than as $0.00.

### 40. An exporter failure kills a completed run and masks the real exception — `telemetry/tracer.py:291-292`

`self.exporter.export(trace)` is called in the `finally` of `Tracer.trace` with no
guard. `FanOutExporter` contains exporter failures and its comment states the
principle — "Telemetry must never break the agent it is observing" — but that
containment lives only inside the wrapper, and `cli/__main__.py:94-95` wires
`StoreExporter(store)` in directly, unwrapped. `OtlpFileExporter.export`
(`otlp.py:111-113`) is likewise unguarded.

Reproduced with an exporter raising `OSError(28, "No space left on device")`, the
shape of a full disk or an sqlite write that outlives the 30s busy timeout:

```
A. exporter raises on a clean run:
   RUN DIED: OSError: [Errno 28] No space left on device
B. exporter raises while the run is already failing:
   caller sees: OSError: [Errno 28] No space left on device
```

Case B is the worse half: raising from a `finally` replaces the in-flight
exception, so the real error the user needed to see is discarded and they are
handed a telemetry error instead.

Fix: wrap the `export` call at line 292 in `try/except Exception` and log, the way
`FanOutExporter` already does.

### 41. `load_dotenv` silently corrupts values — `config.py:59-60`

`value.strip().strip('"').strip("'")` keeps an inline comment as part of the value
and strips quote characters that are part of the secret.

```
AZURE_OPENAI_API_KEY = 'sk-abc123  # production key'
QUOTED = 'tail-quote'
ODD = 'abc'          # input was ODD=abc"
```

The shipped `.env.example` uses only full-line comments, which the loader handles,
so this bites the first user who adds an inline one — a normal `.env` habit. The
failure surfaces as an opaque 401 from Azure with nothing pointing at the loader.
The docstring's "minimal, dependency-free" intent covers not *supporting* inline
comments; it does not cover corrupting the value silently.

Fix: split on an unquoted `#` before stripping, and strip at most one matched pair
of surrounding quotes.

## What has NOT been reviewed yet

- `toolkit/files.py`, `ledger.py`, `shell.py`, `common.py`, `__init__.py` — a
  review of these was in flight when this doc was written. `toolkit/search.py` is
  done (finding 37).
- `telemetry/metrics.py`, `otlp.py`, `tracer.py`, `semconv.py` — spans left open on
  exception paths, span-attribute cardinality, OTLP payload shape.
  `store.py` and `pricing.py` are done (findings 38-39). "Telemetry failure crashing
  a run" is answered and is finding 40, so start elsewhere on `tracer.py`.
- `tools/registry.py`, `tools/spec.py`, `dispatch.py`, `cli/`, `messages.py`,
  `errors.py` — argument validation before handler dispatch, config precedence,
  exit codes. Partly covered: the `--grant read-only` question is answered under
  "Checked and clean" below.

## Checked and clean

Stated so the next session knows these were looked at, not skipped.

- **`--grant read-only` is enforced at dispatch, not only at listing.**
  `dispatch.py:206` calls `registry.authorize(call.name, self.grant)` before every
  handler. Executed: under `Grant.read_only()` the registry lists
  `['glob', 'grep', 'read_file']`, and `authorize` raises `CapabilityDenied` for
  `bash`, `edit_file` and `write_file` even when named directly.
- **`runner.py` environment scrubbing (163-201).** With `AZURE_OPENAI_API_KEY`,
  `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `AWS_SECRET_ACCESS_KEY`,
  `GANTRY_AZURE_API_KEY`, `GITHUB_TOKEN`, `SSH_AUTH_SOCK`, `LD_PRELOAD`,
  `PYTHONPATH`, `PYTHONSTARTUP` and `BASH_ENV` all set in the parent, the child saw
  only the allowlist. Zero leaks.
- **`runner.py` cwd containment (246-248).** `../..`, `/etc`, `sub/../../..` and a
  workspace symlink pointing at `/etc` all raised `PathEscape`.
- **`runner.py` process-group kill and SIGTERM→SIGKILL escalation.** A child that
  spawns a grandchild and times out: `killed_group=True`, grandchild gone. A child
  with `SIGTERM` set to `SIG_IGN` returned at 4.00s with `exit_code=-9`. Worst-case
  timeout overrun measured at +2s, inside `shell_timeout_margin_s = 15.0`. No
  zombies on any path except finding 5.
- **`runner.py` subprocess-path output cap.** 64 MiB emitted with
  `max_output_bytes=1024`: 1024 bytes kept, `truncated=True`, peak allocation
  0.1 MB. The cap is incremental. UTF-8 split at the cap yields a single U+FFFD,
  no exception.
- **`shell=True` / argv construction.** None anywhere in `sandbox/`. Both `Popen`
  calls take a list; `_limits` ends in `os.execvp` with a list.
- **Pipe-buffer deadlock on stdout/stderr in the subprocess path.** Both drained by
  threads started before `wait()`; 64 MiB completed in 0.3s. (The *stdin* pipe is
  the exception — finding 3.)
- **`policy.py` default-deny on malformed input (230-236).** `""` → `deny/empty`;
  `'"unterminated'` → `deny/unparseable`; `[]` → `deny/empty`.
- **`policy.py` case sensitivity and ReDoS in the other rules.**
  `remote-code-execution`, `decoded-execution` and `fork-bomb` are all linear.
  Only `recursive-root-delete` blows up (finding 11).
- **Loop termination.** Every adversarial input terminates: empty reply every turn
  plus an always-failing gate → `verification_failed` after 4 turns; a tool that
  raises every turn → `no_progress` or `tool_error_budget`; a provider reporting
  zero tokens for 500 scripted turns → still stops. `record_step` is unconditional,
  so no `continue` path escapes the step counter.
- **Iteration cap placement.** `before_step` runs before the provider call;
  `max_steps=N` buys exactly N calls, and the result on the last permitted turn is
  returned, not dropped.
- **Tool-handler exceptions.** A handler raising `RuntimeError` becomes a failure
  `ToolResult`; history stays consistent, one tool message per call id.
- **Message-history consistency under elision.** 12-turn run with 7 elisions: zero
  unanswered `tool_call_id`s across all 12 requests; system and task preserved
  verbatim; receipts not double-wrapped.
- **`CommandGate` verdict semantics (verify.py:115-143).** Exit code is the only
  signal and there is no output parsing, so nothing can fail open: exit 0 printing
  "3 tests FAILED" → pass (by design); no output → pass; stderr-only exit 1 → fail;
  exit 127 → fail; SIGKILL/SIGSEGV → fail; timeout → fail, and `expect_exit=-9`
  cannot launder a timeout into a pass.
- **Contract boundary arithmetic (contract.py:228-303).** Exact equality at every
  limit refuses consistently. Float drift over 1,000,000 charges of $0.000001
  accumulates upward (conservative) and the limit still fires.
- **`action_fingerprint` (contract.py:87-96).** Order-stable across key
  permutations; survives unicode keys, a 5 MB string, a non-serialisable object and
  NaN.
- **Double-charging.** A retried tool call is recorded once; provider-level retries
  report usage only for the attempt that succeeded.
- **`telemetry/store.py` SQL and concurrency (file-backed).** Every caller-supplied
  value travels as a bound parameter; the one interpolated fragment is assembled
  from literals. Thread-local connections, a write lock, `BEGIN IMMEDIATE` with an
  explicit `ROLLBACK`, WAL and a 30s busy timeout. Clean apart from finding 38.
- **Cache key completeness (`providers/cache.py`).** Mutating model, system prompt,
  user message, tool-result content, tool-result id, assistant tool-call arguments,
  tools, tool_choice, max_output_tokens, temperature, top_p, parallel_tool_calls and
  seed each changes the key. No collision found. `role`, `timeout_s` and `metadata`
  correctly do not: `model` is already derived from `role` and hashed, and the other
  two cannot change the answer. The key is a superset of what `_build_kwargs` sends.
- **Cache key stability across processes.** `hash()` is not used; sha256 over
  `json.dumps(sort_keys=True)`. One distinct key across `PYTHONHASHSEED` 0/1/12345.
- **Cached payloads carrying credentials.** `_to_payload` stores only the completion
  (message, finish_reason, model, usage, response_id) — no API key, no
  Authorization header, no prompt.
- **Partially written cassettes.** `put()` writes `.tmp` and `Path.replace()`s it,
  atomic on the same filesystem; a reader never sees a half-written file.
- **Provider retry cap and backoff.** `max_attempts=3`, full-jitter exponential
  capped at `max_delay_s=20`; it cannot spin. No streaming, so no consumed-stream
  re-read.
- **Secrets in provider exception messages and telemetry.** `_translate` embeds only
  the SDK exception string, request id and status code; `describe()` exposes
  endpoint, api_version and deployments but no key; `_map_exception` forwards
  `GantryError.details`, which holds paths and exit codes only.
- **JSON-RPC batch handling, unknown methods and ids.** One malformed element does
  not discard the batch; an empty array is `INVALID_REQUEST`; a batch of only
  notifications is silent; an unknown method gets `-32601` as a request and nothing
  as a notification; `id: null` is answered and not treated as a notification;
  duplicate ids each get their own reply, which the spec permits.
- **RPC handler exceptions.** `_handle_one` converts any handler `Exception` into an
  error response and the `finally` always clears `_inflight`. Only the serialisation
  step afterwards is unguarded (finding 31).
- **`offline.py` divergence from the real provider.** Same `Completion` shape, same
  `ProviderError` types, deterministic exhaustion turn. One divergence, verified as
  harmless: offline tool-call `arguments` is the fixture dict itself while
  production yields a JSON string, but `dispatch.py` fingerprints the *parsed*
  arguments. The other divergence is not harmless and is finding 28: offline always
  populates usage via `estimate_usage`, which is why azure's all-zero usage has no
  test coverage.
- **`toolkit/search.py` glob against a dangling symlink.** `glob` sorts on
  `p.stat().st_mtime` with no `OSError` guard, unlike `grep` at line 102, but
  `jail.iter_files` does not yield broken symlinks, so the hypothesis does not
  reproduce. Left unreported rather than filed as speculation.

## Two operational notes for the next session

1. **Codex CLI is not reachable from a Claude Code cloud session.** Re-confirmed in
   session 2: `codex` is not on `PATH` and `~/.codex` does not exist. Installing the
   binary would not help either — its Azure/LiteLLM provider config and credentials
   live on the local machine, so `codex exec` would have nothing to authenticate
   against. To use the `codex-worker` skill with `gpt-5.6-sol` / `gpt-5.6-luna`, run
   Claude Code locally. In the cloud, the heavy/light split maps onto Claude
   subagents instead.
2. **Do not fan out six subagents at once.** An early session launched six in
   parallel (three on opus) and all six died instantly on a 429 session limit.
   Session 2 ran at most three concurrently with no 429s, and reviewed two
   subsystems on the main thread in parallel with them, which is cheaper because
   the main thread does not re-derive context cold. Three is a workable ceiling.

## Suggested partition, by risk rather than line count

| Tier | Unit | Lines | State |
|---|---|---|---|
| Heavy | `sandbox/` — runner, policy, jail, limits | 1,076 | done |
| Heavy | `loop.py` + `verify.py` + `contract.py` + `budget.py` | 1,490 | done |
| Heavy | `providers/` + `rpc/` | 1,663 | done |
| Light | `toolkit/` | 1,118 | `search.py` done |
| Light | `telemetry/` | 1,120 | `store.py`, `pricing.py` done |
| Light | `tools/` + `dispatch.py` + `cli/` + `messages.py` + `config.py` + `errors.py` | 1,542 | `config.py` partly done |

---

## Prompt to paste

> Resume the code review of the `gantry` harness in this repo. Read
> `docs/code-review-resume.md` first — it records the baseline, the findings
> already confirmed, and exactly which subsystems remain unreviewed. Do not redo
> the finished work, and do not re-report anything in the "Checked and clean"
> section.
>
> Work through the unreviewed subsystems in the order listed there. For each one,
> invoke the `code-review` skill on that path (`max` effort for the heavy tier,
> `high` for the light tier), then go beyond it with the adversarial checklist
> given for that subsystem in the doc.
>
> Rules I care about:
> - Reproduce every finding by executing it before reporting it. The suite is
>   green and ruff is clean, so anything real is something the tests miss — prove
>   it with a script under the scratchpad, the way the existing findings were
>   proved.
> - No style opinions and no unverified speculation. Each finding needs
>   `file:line`, one sentence on the defect, and a concrete failure scenario with
>   specific input and resulting wrong behavior.
> - Weigh findings against stated intent. `policy.py` openly says a denylist is
>   not a security boundary; do not report that as a discovery.
> - Say plainly when a category checks out clean, so I know it was actually looked
>   at rather than skipped.
> - Read-only on `src/` and `tests/` until I say otherwise. Do not fix, commit, or
>   push source changes without asking. Updating this doc is expected.
>
> Append what you find to `docs/code-review-resume.md` under the confirmed
> findings, and move each subsystem you finish out of the "not reviewed" list, so
> the doc stays accurate if this session dies too.
