"""Registration-time validation is the point of the registry: a malformed tool
should fail on line 12 of a test run, not three tool calls into a paid agent
run. These tests assert that each class of mistake is caught early."""

from __future__ import annotations

import pytest

from conftest import echo_schema, make_spec
from gantry.errors import (
    CapabilityDenied,
    ToolNotFound,
    ToolRegistrationError,
    ToolValidationError,
)
from gantry.tools import Capability, Grant, ToolRegistry, ToolResult, ToolSpec


def test_register_and_resolve(registry):
    registry.register(make_spec("echo"))
    assert registry.get("echo").version == "1.0.0"
    assert "echo" in registry
    assert len(registry) == 1


@pytest.mark.parametrize("name", ["Echo", "9lives", "with space", "with-dash", "", "x" * 65])
def test_invalid_names_are_rejected(registry, name):
    with pytest.raises(ToolRegistrationError):
        registry.register(make_spec(name))


def test_blank_description_is_rejected(registry):
    with pytest.raises(ToolRegistrationError, match="description is required"):
        registry.register(make_spec("echo", description="   "))


def test_oversized_description_is_rejected(registry):
    """Descriptions are prompt tokens on every turn, so their size is enforced."""
    with pytest.raises(ToolRegistrationError, match="over the"):
        registry.register(make_spec("echo", description="x" * 2000))


def test_malformed_json_schema_is_rejected(registry):
    with pytest.raises(ToolRegistrationError, match="not a valid JSON Schema"):
        registry.register(make_spec("echo", input_schema={"type": "not-a-type"}))


def test_strict_mode_requires_every_property_in_required(registry):
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
        "required": ["a"],
        "additionalProperties": False,
    }
    with pytest.raises(ToolRegistrationError) as exc:
        registry.register(make_spec("echo", input_schema=schema))
    assert exc.value.details["missing_from_required"] == ["b"]


def test_strict_mode_requires_additional_properties_false(registry):
    schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
    with pytest.raises(ToolRegistrationError, match="additionalProperties"):
        registry.register(make_spec("echo", input_schema=schema))


def test_non_strict_tools_skip_the_strict_checks(registry):
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    registry.register(make_spec("echo", input_schema=schema, strict=False))
    assert "strict" not in registry.to_openai_tools()[0]["function"]


def test_duplicate_registration_is_refused_unless_replacing(registry):
    registry.register(make_spec("echo"))
    with pytest.raises(ToolRegistrationError, match="already registered"):
        registry.register(make_spec("echo"))
    registry.register(make_spec("echo", description="Replaced."), replace=True)
    assert registry.get("echo").description == "Replaced."


def test_versions_resolve_to_the_highest_by_default(registry):
    registry.register(make_spec("echo", version="1.0.0"))
    registry.register(make_spec("echo", version="1.10.0"))
    registry.register(make_spec("echo", version="1.9.0"))
    # Highest by semver ordering, not lexicographic: 1.10.0 beats 1.9.0.
    assert registry.get("echo").version == "1.10.0"
    assert registry.get("echo", "1.0.0").version == "1.0.0"
    assert registry.get("echo@1.9.0").version == "1.9.0"


def test_unknown_tool_and_unknown_version_both_raise(registry):
    registry.register(make_spec("echo"))
    with pytest.raises(ToolNotFound):
        registry.get("nope")
    with pytest.raises(ToolNotFound, match="no version"):
        registry.get("echo", "9.9.9")


def test_only_the_latest_version_is_listed(registry):
    registry.register(make_spec("echo", version="1.0.0"))
    registry.register(make_spec("echo", version="2.0.0"))
    assert [s.version for s in registry.list()] == ["2.0.0"]


# --- argument validation ---------------------------------------------------
def test_validation_reports_every_error_at_once(registry):
    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "count": {"type": "integer"}},
        "required": ["path", "count"],
        "additionalProperties": False,
    }
    registry.register(make_spec("read", input_schema=schema))
    with pytest.raises(ToolValidationError) as exc:
        registry.validate_arguments("read", {"path": 7})
    errors = exc.value.details["errors"]
    # One turn per error is one turn too many, so both problems are reported.
    assert len(errors) == 2
    assert any("count" in e for e in errors)
    assert any("path" in e for e in errors)


def test_non_object_arguments_are_rejected(registry):
    registry.register(make_spec("echo"))
    with pytest.raises(ToolValidationError, match="must be a JSON object"):
        registry.validate_arguments("echo", ["text"])


def test_valid_arguments_pass_through(registry):
    registry.register(make_spec("echo"))
    assert registry.validate_arguments("echo", {"text": "hi"}) == {"text": "hi"}


# --- capability scoping ----------------------------------------------------
@pytest.fixture
def scoped_registry(registry) -> ToolRegistry:
    registry.register(make_spec("read_file", capabilities=frozenset({Capability.FS_READ})))
    registry.register(
        make_spec("write_file", capabilities=frozenset({Capability.FS_WRITE}), destructive=True)
    )
    registry.register(make_spec("run_shell", capabilities=frozenset({Capability.PROC_EXEC})))
    return registry


def test_a_grant_filters_what_the_model_is_told_about(scoped_registry):
    """Unpermitted tools are never described. Refusing a call the model was
    invited to make wastes a turn; not offering it costs nothing."""
    names = [t["function"]["name"] for t in scoped_registry.to_openai_tools(Grant.read_only())]
    assert names == ["read_file"]
    dev = [t["function"]["name"] for t in scoped_registry.to_openai_tools(Grant.developer())]
    assert dev == ["read_file", "run_shell", "write_file"]


def test_authorize_reports_the_missing_capabilities(scoped_registry):
    with pytest.raises(CapabilityDenied) as exc:
        scoped_registry.authorize("write_file", Grant.read_only())
    assert exc.value.details["missing"] == ["fs:write"]


def test_grants_can_narrow_to_named_tools(scoped_registry):
    grant = Grant(capabilities=frozenset(Capability), tools=frozenset({"read_file"}))
    assert [s.name for s in scoped_registry.list(grant)] == ["read_file"]
    with pytest.raises(CapabilityDenied):
        scoped_registry.authorize("run_shell", grant)


def test_grants_fail_closed_for_a_newly_added_capability(registry):
    """A tool needing a capability nobody granted is invisible, not permitted."""
    registry.register(make_spec("exfiltrate", capabilities=frozenset({Capability.NET})))
    assert registry.list(Grant.developer()) == []


# --- provider rendering ----------------------------------------------------
def test_openai_tool_json_shape(registry):
    registry.register(make_spec("echo"))
    tool = registry.to_openai_tools()[0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "echo"
    assert tool["function"]["strict"] is True
    assert tool["function"]["parameters"]["additionalProperties"] is False


def test_describe_exposes_execution_properties(registry):
    registry.register(
        make_spec(
            "write_file", idempotent=False, destructive=True, concurrency_safe=False, timeout_s=5.0
        )
    )
    described = registry.describe()[0]
    assert described["destructive"] is True
    assert described["concurrency_safe"] is False
    assert described["timeout_s"] == 5.0


def test_only_idempotent_tools_are_retryable():
    """Retrying a write after a timeout can apply it twice: a timeout means the
    answer was lost, not that the work was."""
    assert make_spec("read", idempotent=True).is_retryable
    assert not make_spec("write", idempotent=False).is_retryable


# --- results ---------------------------------------------------------------
def test_truncation_is_recorded_not_silent():
    result = ToolResult.success("x" * 100).truncate(20)
    assert result.truncated is True
    assert result.original_bytes == 100
    assert "80 more characters" in result.content
    assert result.content.startswith("x" * 20)


def test_short_output_is_untouched():
    result = ToolResult.success("short").truncate(100)
    assert result.content == "short"
    assert result.truncated is False


def test_bad_handler_and_timeout_are_caught_at_registration(registry):
    with pytest.raises(ToolRegistrationError, match="not callable"):
        registry.register(ToolSpec("echo", "1.0.0", "d", echo_schema("t"), handler="nope"))
    with pytest.raises(ToolRegistrationError, match="timeout_s"):
        registry.register(make_spec("echo", timeout_s=0))


def test_bad_version_string_is_rejected(registry):
    with pytest.raises(ToolRegistrationError, match=r"major\.minor\.patch"):
        registry.register(make_spec("echo", version="1.0"))
