"""Tests for the ``gantry`` command line.

Two of these matter more than the rest: that ``doctor`` never prints a
credential, and that ``run`` exits non-zero when the agent did not finish.
The second is what makes the CLI usable as a CI step.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gantry.cli.__main__ import (
    EXIT_INCOMPLETE,
    EXIT_OK,
    build_gates,
    build_parser,
    build_provider,
    main,
)
from gantry.config import AzureConfig, Config


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (tmp_path / "test_calc.py").write_text(
        "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    return tmp_path


def test_the_parser_requires_a_subcommand():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_run_defaults_to_the_offline_provider_and_developer_grant():
    args = build_parser().parse_args(["run", "do a thing"])
    assert args.provider == "offline"
    assert args.grant == "developer"


def test_tools_lists_what_the_grant_refuses(project: Path, capsys):
    assert main(["tools", "-w", str(project), "-g", "read-only"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "[x] bash" in out
    assert "[ ] read_file" in out
    assert "refused: needs fs:write,proc:exec" in out


def test_tools_json_is_machine_readable(project: Path, capsys):
    assert main(["tools", "-w", str(project), "--json"]) == EXIT_OK
    described = json.loads(capsys.readouterr().out)
    assert {entry["name"] for entry in described} >= {"read_file", "bash"}


def test_doctor_never_prints_the_key_itself(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)  # no .env to load
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "sk-do-not-print-me")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    assert main(["doctor"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "sk-do-not-print-me" not in out
    assert "AZURE_OPENAI_API_KEY is set" in out
    assert "azure endpoint" in out


def test_doctor_reports_missing_configuration_without_failing(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    assert main(["doctor"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "MISS" in out
    assert "Offline runs work regardless." in out


def test_an_unfinished_run_exits_non_zero(project: Path, capsys):
    """The offline provider does no work, so the gate must refuse the claim."""
    code = main(["run", "fix the test", "-w", str(project), "--gate", "tests=python -m pytest -q"])
    assert code == EXIT_INCOMPLETE
    out = capsys.readouterr().out
    assert "verification_failed" in out
    assert "gate     FAIL  tests" in out


def test_run_json_output_is_parseable(project: Path, capsys):
    main(["run", "fix it", "-w", str(project), "--json", "--max-steps", "2"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"]
    assert payload["gates"] is not None
    # With no --gate given, the only check is that the workspace still parses,
    # which it does. A run is only as demanding as its acceptance criteria,
    # which is exactly why the eval fixtures define real ones.
    assert payload["succeeded"] is True


def test_a_run_can_be_recorded_and_listed_again(project: Path, capsys):
    db = project / "runs.db"
    main(["run", "fix it", "-w", str(project), "--db", str(db), "--max-steps", "1"])
    capsys.readouterr()
    assert main(["trace", "--db", str(db), "--json"]) == EXIT_OK
    runs = json.loads(capsys.readouterr().out)
    assert len(runs) == 1
    assert runs[0]["task"] == "fix it"


def test_trace_on_an_empty_database_says_so(tmp_path: Path, capsys):
    assert main(["trace", "--db", str(tmp_path / "empty.db")]) == EXIT_OK
    assert "No runs recorded" in capsys.readouterr().out


def test_azure_without_configuration_explains_what_is_missing():
    with pytest.raises(SystemExit, match="AZURE_OPENAI_ENDPOINT"):
        build_provider("azure", Config(azure=AzureConfig()))


def test_an_unknown_provider_is_refused():
    with pytest.raises(SystemExit, match="unknown provider"):
        build_provider("mystery", Config())


def test_gate_arguments_parse_with_and_without_a_name():
    gates = build_gates(["tests=python -m pytest -q", "ruff check ."], syntax=True)
    names = [gate.name for gate in gates.gates]
    assert names == ["python-syntax", "tests", "check"]
