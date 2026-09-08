"""Tests for the workspace toolkit.

The emphasis is on the failure paths. A read that works is one assertion; a
refused stale edit, a diagnosed whitespace mismatch and a byte-exact CRLF
round trip are the behaviours that decide whether an agent finishes a task.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gantry.config import Config, SandboxConfig
from gantry.errors import CapabilityDenied, PathEscape, ToolExecutionError, ToolNotFound
from gantry.toolkit import ToolkitLimits, build_toolkit
from gantry.toolkit.files import _explain_miss
from gantry.toolkit.ledger import FileLedger
from gantry.tools import Grant, ToolContext


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (tmp_path / "src" / "util.py").write_text("VALUE = 1\n")
    (tmp_path / "README.md").write_text("# Fixture\n\nA calculator.\n")
    return tmp_path


@pytest.fixture
def toolkit(workspace: Path):
    return build_toolkit(workspace, config=Config(sandbox=SandboxConfig(timeout_s=20.0)))


@pytest.fixture
def ctx(workspace: Path) -> ToolContext:
    return ToolContext(run_id="test", workspace=workspace)


def call(toolkit, name: str, ctx: ToolContext, **arguments):
    """Invoke a tool the way the dispatcher does: validate, then run.

    Unset properties are filled with ``None`` because that is what a strict
    function-calling model sends. Strict mode requires every declared property
    to be present, so "optional" means "nullable", never "absent".
    """
    spec = toolkit.registry.get(name)
    payload = {key: arguments.get(key) for key in spec.input_schema["properties"]}
    payload.update(arguments)
    toolkit.registry.validate_arguments(name, payload)
    return spec.handler(payload, ctx)


# --- registration ---------------------------------------------------------
def test_every_tool_registers_with_a_strict_schema(toolkit):
    for spec in toolkit.specs():
        schema = spec.input_schema
        assert schema["additionalProperties"] is False
        assert set(schema["properties"]) == set(schema["required"])


def test_optional_arguments_accept_null(toolkit, ctx):
    # Strict mode sends every declared property, so a handler that treated a
    # missing key as the default would still be handed an explicit null.
    result = call(toolkit, "read_file", ctx, path="README.md", offset=None, limit=None)
    assert result.ok


def test_an_omitted_optional_property_is_still_a_schema_error(toolkit):
    from gantry.errors import ToolValidationError

    with pytest.raises(ToolValidationError, match="'offset' is a required property"):
        toolkit.registry.validate_arguments("read_file", {"path": "README.md"})


def test_the_developer_grant_covers_the_whole_toolkit(toolkit):
    grant = Grant.developer()
    assert [spec.name for spec in toolkit.specs() if not grant.permits(spec)] == []


def test_a_read_only_grant_hides_the_writing_tools(toolkit):
    names = {
        tool["function"]["name"] for tool in toolkit.registry.to_openai_tools(Grant.read_only())
    }
    assert names == {"read_file", "grep", "glob"}


def test_shell_can_be_left_out_entirely(workspace):
    toolkit = build_toolkit(workspace, shell=False)
    assert "bash" not in toolkit.registry
    assert toolkit.runner is None


# --- read -----------------------------------------------------------------
def test_read_numbers_lines_and_reports_size(toolkit, ctx):
    result = call(toolkit, "read_file", ctx, path="src/calc.py")
    assert result.ok
    assert "     1\tdef add(a, b):" in result.content
    assert result.data["total_lines"] == 2
    assert result.data["complete"] is True


def test_read_paginates_and_says_how_to_continue(toolkit, workspace, ctx):
    (workspace / "long.txt").write_text("".join(f"line {n}\n" for n in range(1, 51)))
    result = call(toolkit, "read_file", ctx, offset=None, limit=10, path="long.txt")
    assert "showing lines 1-10" in result.content
    assert "offset=11" in result.content
    assert result.data["complete"] is False


def test_read_of_a_missing_file_suggests_a_near_name(toolkit, ctx):
    with pytest.raises(ToolNotFound) as excinfo:
        call(toolkit, "read_file", ctx, path="src/calk.py")
    assert "calc.py" in str(excinfo.value)


def test_read_of_a_directory_points_at_glob(toolkit, ctx):
    with pytest.raises(ToolExecutionError, match="glob"):
        call(toolkit, "read_file", ctx, path="src")


def test_read_refuses_binary_rather_than_mangling_it(toolkit, workspace, ctx):
    (workspace / "blob.bin").write_bytes(b"\x89PNG\x00\x01\x02")
    with pytest.raises(ToolExecutionError, match="binary"):
        call(toolkit, "read_file", ctx, path="blob.bin")


def test_read_refuses_invalid_utf8(toolkit, workspace, ctx):
    (workspace / "latin.txt").write_bytes(b"caf\xe9 au lait\n")
    with pytest.raises(ToolExecutionError, match="UTF-8"):
        call(toolkit, "read_file", ctx, path="latin.txt")


def test_read_refuses_a_file_over_the_limit(workspace, ctx):
    toolkit = build_toolkit(workspace, limits=ToolkitLimits(max_read_bytes=16))
    with pytest.raises(ToolExecutionError, match="read limit"):
        call(toolkit, "read_file", ctx, path="README.md")


def test_reading_an_empty_file_says_so(toolkit, workspace, ctx):
    (workspace / "empty.txt").touch()
    result = call(toolkit, "read_file", ctx, path="empty.txt")
    assert "is empty" in result.content
    assert result.data["total_lines"] == 0


def test_a_very_long_line_is_clipped(toolkit, workspace, ctx):
    (workspace / "min.js").write_text("x" * 5000 + "\n")
    result = call(toolkit, "read_file", ctx, path="min.js")
    assert "line clipped" in result.content
    assert len(result.content) < 4000


def test_read_cannot_escape_the_workspace(toolkit, ctx):
    with pytest.raises(PathEscape):
        call(toolkit, "read_file", ctx, path="../outside.txt")


# --- write ----------------------------------------------------------------
def test_write_creates_a_file_and_its_parents(toolkit, workspace, ctx):
    result = call(toolkit, "write_file", ctx, path="pkg/deep/new.py", content="X = 1\n")
    assert result.ok and result.data["created"] is True
    assert (workspace / "pkg" / "deep" / "new.py").read_text() == "X = 1\n"


def test_write_refuses_to_clobber_a_file_it_has_not_read(toolkit, ctx):
    result = call(toolkit, "write_file", ctx, path="src/calc.py", content="nope\n")
    assert not result.ok
    assert result.error_code == "tool.stale_write"
    assert "read_file" in result.content


def test_write_is_allowed_once_the_file_has_been_read(toolkit, workspace, ctx):
    call(toolkit, "read_file", ctx, path="src/calc.py")
    result = call(
        toolkit, "write_file", ctx, path="src/calc.py", content="def add(a, b):\n    return a + b\n"
    )
    assert result.ok and result.data["created"] is False
    assert (workspace / "src" / "calc.py").read_text().endswith("a + b\n")


def test_write_refuses_a_protected_directory(toolkit, workspace, ctx):
    (workspace / ".git").mkdir()
    with pytest.raises(CapabilityDenied, match="protected"):
        call(toolkit, "write_file", ctx, path=".git/config", content="[core]\n")


def test_write_refuses_an_oversized_payload(workspace, ctx):
    toolkit = build_toolkit(workspace, limits=ToolkitLimits(max_write_bytes=8))
    with pytest.raises(ToolExecutionError, match="byte limit"):
        call(toolkit, "write_file", ctx, path="big.txt", content="x" * 100)


def test_write_does_not_add_a_trailing_newline(toolkit, workspace, ctx):
    call(toolkit, "write_file", ctx, path="exact.txt", content="no newline")
    assert (workspace / "exact.txt").read_bytes() == b"no newline"


# --- edit -----------------------------------------------------------------
def test_edit_replaces_a_unique_match(toolkit, workspace, ctx):
    call(toolkit, "read_file", ctx, path="src/calc.py")
    result = call(
        toolkit,
        "edit_file",
        ctx,
        path="src/calc.py",
        old_string="return a - b",
        new_string="return a + b",
        replace_all=None,
    )
    assert result.ok and result.data["first_line"] == 2
    assert (workspace / "src" / "calc.py").read_text() == "def add(a, b):\n    return a + b\n"


def test_edit_requires_a_prior_read(toolkit, ctx):
    result = call(
        toolkit,
        "edit_file",
        ctx,
        path="src/calc.py",
        old_string="a - b",
        new_string="a + b",
        replace_all=None,
    )
    assert not result.ok and result.error_code == "tool.stale_write"


def test_edit_refuses_when_the_file_changed_underneath(toolkit, workspace, ctx):
    call(toolkit, "read_file", ctx, path="src/calc.py")
    (workspace / "src" / "calc.py").write_text("def add(a, b):\n    return b - a\n")
    result = call(
        toolkit,
        "edit_file",
        ctx,
        path="src/calc.py",
        old_string="b - a",
        new_string="a + b",
        replace_all=None,
    )
    assert not result.ok
    assert "changed on disk" in result.content


def test_freshness_survives_a_same_second_same_length_edit(toolkit, workspace, ctx):
    """The failure mode that a timestamp check would miss.

    Same byte length, same second: mtime and size are both unchanged, which is
    exactly the pair CPython's bytecode cache trusts. Hashing the content is
    what makes this detectable at all.
    """
    target = workspace / "src" / "calc.py"
    call(toolkit, "read_file", ctx, path="src/calc.py")
    before = target.stat()
    target.write_text("def add(a, b):\n    return a + b\n")  # identical length
    import os

    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert target.stat().st_size == before.st_size
    assert target.stat().st_mtime_ns == before.st_mtime_ns

    result = call(
        toolkit,
        "edit_file",
        ctx,
        path="src/calc.py",
        old_string="a + b",
        new_string="a * b",
        replace_all=None,
    )
    assert not result.ok and "changed on disk" in result.content


def test_edit_rejects_an_ambiguous_match_with_line_numbers(toolkit, workspace, ctx):
    (workspace / "dup.py").write_text("x = 1\ny = 2\nx = 1\n")
    call(toolkit, "read_file", ctx, path="dup.py")
    result = call(
        toolkit,
        "edit_file",
        ctx,
        path="dup.py",
        old_string="x = 1",
        new_string="x = 9",
        replace_all=None,
    )
    assert not result.ok and result.error_code == "tool.ambiguous_match"
    assert "line(s) 1, 3" in result.content


def test_replace_all_takes_every_occurrence(toolkit, workspace, ctx):
    (workspace / "dup.py").write_text("x = 1\ny = 2\nx = 1\n")
    call(toolkit, "read_file", ctx, path="dup.py")
    result = call(
        toolkit,
        "edit_file",
        ctx,
        path="dup.py",
        old_string="x = 1",
        new_string="x = 9",
        replace_all=True,
    )
    assert result.ok and result.data["replacements"] == 2
    assert (workspace / "dup.py").read_text() == "x = 9\ny = 2\nx = 9\n"


def test_a_no_op_edit_is_refused(toolkit, ctx):
    call(toolkit, "read_file", ctx, path="src/calc.py")
    result = call(
        toolkit,
        "edit_file",
        ctx,
        path="src/calc.py",
        old_string="same",
        new_string="same",
        replace_all=None,
    )
    assert not result.ok and "identical" in result.content


def test_an_empty_old_string_is_refused(toolkit, ctx):
    call(toolkit, "read_file", ctx, path="src/calc.py")
    result = call(
        toolkit,
        "edit_file",
        ctx,
        path="src/calc.py",
        old_string="",
        new_string="x",
        replace_all=None,
    )
    assert not result.ok and "write_file" in result.content


def test_edit_preserves_crlf_line_endings(toolkit, workspace, ctx):
    (workspace / "dos.txt").write_bytes(b"alpha\r\nbeta\r\n")
    call(toolkit, "read_file", ctx, path="dos.txt")
    result = call(
        toolkit,
        "edit_file",
        ctx,
        path="dos.txt",
        old_string="beta",
        new_string="gamma",
        replace_all=None,
    )
    assert result.ok
    assert (workspace / "dos.txt").read_bytes() == b"alpha\r\ngamma\r\n"


# --- miss diagnostics -----------------------------------------------------
def test_miss_diagnosis_spots_copied_line_numbers():
    assert "line numbers" in _explain_miss("def add():\n    pass\n", "     1\tdef add():")


def test_miss_diagnosis_spots_crlf():
    assert "CRLF" in _explain_miss("a\r\nb\r\n", "a\nb\n")


def test_miss_diagnosis_spots_surrounding_whitespace():
    assert "whitespace" in _explain_miss("value = 1\n", "\nvalue = 1\n\n")


def test_miss_diagnosis_spots_indentation():
    assert "indentation" in _explain_miss("    if x:\n        go()\n", "if x:\n  go()")


def test_miss_diagnosis_points_at_the_first_line():
    current = "def handler(request):\n    return 1\n"
    assert "line(s) 1" in _explain_miss(current, "def handler(request):\n    return 2\n")


def test_miss_diagnosis_admits_when_nothing_matches():
    assert "No part of it" in _explain_miss("a = 1\n", "completely unrelated text")


# --- grep -----------------------------------------------------------------
def grep(toolkit, ctx, **kwargs):
    arguments = {
        "pattern": "",
        "path": None,
        "glob": None,
        "case_sensitive": None,
        "max_results": None,
    }
    arguments.update(kwargs)
    return call(toolkit, "grep", ctx, **arguments)


def test_grep_reports_path_line_and_text(toolkit, ctx):
    result = grep(toolkit, ctx, pattern=r"def add")
    assert result.ok
    assert "src/calc.py:1: def add(a, b):" in result.content
    assert result.data["files_matched"] == ["src/calc.py"]


def test_grep_is_case_insensitive_by_default(toolkit, ctx):
    assert grep(toolkit, ctx, pattern="FIXTURE").data["matches"] == 1
    assert grep(toolkit, ctx, pattern="FIXTURE", case_sensitive=True).data["matches"] == 0


def test_grep_honours_a_glob_and_a_path(toolkit, ctx):
    assert grep(toolkit, ctx, pattern="=", glob="**/*.md").data["matches"] == 0
    assert grep(toolkit, ctx, pattern="VALUE", path="src").data["matches"] == 1


def test_grep_returns_a_usable_message_for_a_bad_regex(toolkit, ctx):
    result = grep(toolkit, ctx, pattern="(unclosed")
    assert not result.ok and "Invalid regular expression" in result.content


def test_grep_skips_binary_and_ignored_directories(toolkit, workspace, ctx):
    (workspace / "blob.bin").write_bytes(b"needle\x00needle")
    vendored = workspace / "node_modules" / "pkg"
    vendored.mkdir(parents=True)
    (vendored / "index.js").write_text("needle\n")
    assert grep(toolkit, ctx, pattern="needle").data["matches"] == 0


def test_grep_caps_its_own_output(workspace, ctx):
    (workspace / "many.txt").write_text("hit\n" * 50)
    toolkit = build_toolkit(workspace, limits=ToolkitLimits(max_grep_matches=5))
    result = grep(toolkit, ctx, pattern="hit")
    assert result.data["matches"] == 5
    assert result.data["capped"] is True


def test_grep_rejects_a_path_that_is_not_a_directory(toolkit, ctx):
    with pytest.raises(ToolExecutionError, match="not a directory"):
        grep(toolkit, ctx, pattern="x", path="README.md")


# --- glob -----------------------------------------------------------------
def test_glob_lists_matching_files_newest_first(toolkit, workspace, ctx):
    import os
    import time

    old = workspace / "src" / "util.py"
    os.utime(old, (time.time() - 500, time.time() - 500))
    result = call(toolkit, "glob", ctx, pattern="src/**/*.py", path=None, max_results=None)
    assert result.data["paths"] == ["src/calc.py", "src/util.py"]


def test_glob_reports_no_match_plainly(toolkit, ctx):
    result = call(toolkit, "glob", ctx, pattern="**/*.rs", path=None, max_results=None)
    assert result.ok and result.data["count"] == 0


def test_glob_caps_and_says_so(workspace, ctx):
    for n in range(10):
        (workspace / f"f{n}.txt").write_text("x")
    toolkit = build_toolkit(workspace, limits=ToolkitLimits(max_glob_results=3))
    result = call(toolkit, "glob", ctx, pattern="*.txt", path=None, max_results=None)
    assert result.data["count"] == 3 and result.data["capped"] is True


# --- bash -----------------------------------------------------------------
def bash(toolkit, ctx, command: str, **kwargs):
    arguments = {"command": command, "cwd": None, "timeout_s": None}
    arguments.update(kwargs)
    return call(toolkit, "bash", ctx, **arguments)


def test_bash_runs_inside_the_workspace(toolkit, ctx):
    result = bash(toolkit, ctx, 'python -c "import os; print(os.getcwd())"')
    assert result.ok
    assert str(toolkit.jail.root) in result.content


def test_bash_reports_a_failing_command_with_its_output(toolkit, ctx):
    result = bash(toolkit, ctx, "python -c \"import sys; sys.stderr.write('boom'); sys.exit(3)\"")
    assert not result.ok
    assert result.error_code == "sandbox.nonzero_exit"
    assert result.data["exit_code"] == 3
    assert "boom" in result.content


def test_bash_surfaces_a_policy_refusal_as_a_refusal(toolkit, ctx):
    result = bash(toolkit, ctx, "sudo rm -rf /")
    assert not result.ok
    assert result.error_code == "sandbox.command_denied"
    assert "will not run" in result.content


def test_bash_rejects_an_empty_command(toolkit, ctx):
    with pytest.raises(ToolExecutionError, match="empty"):
        bash(toolkit, ctx, "   ")


def test_the_bash_dispatch_timeout_exceeds_the_sandbox_timeout(toolkit):
    """Otherwise the dispatcher abandons the wait while the child runs on."""
    spec = toolkit.registry.get("bash")
    assert spec.timeout_s > toolkit.runner.config.timeout_s


def test_a_requested_timeout_cannot_exceed_the_sandbox_ceiling(workspace, ctx):
    toolkit = build_toolkit(workspace, config=Config(sandbox=SandboxConfig(timeout_s=1.0)))
    result = bash(toolkit, ctx, 'python -c "import time; time.sleep(5)"', timeout_s=60)
    assert not result.ok and result.data["timed_out"] is True


def test_bash_is_not_retryable(toolkit):
    assert toolkit.registry.get("bash").is_retryable is False
    assert toolkit.registry.get("edit_file").is_retryable is False
    assert toolkit.registry.get("read_file").is_retryable is True


# --- the ledger -----------------------------------------------------------
def test_a_partial_read_does_not_license_an_edit(workspace, ctx):
    (workspace / "long.py").write_text("".join(f"x{n} = {n}\n" for n in range(40)))
    toolkit = build_toolkit(workspace)
    call(toolkit, "read_file", ctx, path="long.py", offset=None, limit=5)
    result = call(
        toolkit,
        "edit_file",
        ctx,
        path="long.py",
        old_string="x0 = 0",
        new_string="x0 = 9",
        replace_all=None,
    )
    assert not result.ok and "only read part" in result.content


def test_writing_records_the_file_so_a_follow_up_edit_works(toolkit, ctx):
    call(toolkit, "write_file", ctx, path="fresh.py", content="a = 1\n")
    result = call(
        toolkit,
        "edit_file",
        ctx,
        path="fresh.py",
        old_string="a = 1",
        new_string="a = 2",
        replace_all=None,
    )
    assert result.ok


def test_the_ledger_forgets_on_request(workspace):
    ledger = FileLedger()
    path = workspace / "README.md"
    ledger.record(path, "content")
    assert ledger.observed(path) is not None
    ledger.forget(path)
    assert ledger.observed(path) is None


def test_bash_can_invalidate_a_read_and_the_edit_notices(toolkit, ctx):
    """No bookkeeping between the two tools: the digest catches it."""
    call(toolkit, "read_file", ctx, path="src/util.py")
    bash(toolkit, ctx, "python -c \"open('src/util.py','w').write('VALUE = 2\\n')\"")
    result = call(
        toolkit,
        "edit_file",
        ctx,
        path="src/util.py",
        old_string="VALUE",
        new_string="TOTAL",
        replace_all=None,
    )
    assert not result.ok and "changed on disk" in result.content
