"""The tool registry: validation, versioning and capability scoping.

Everything here happens at *registration* time wherever it possibly can. A
malformed schema, a description that will not survive strict mode, a name the
provider will reject - all of those are programming errors, and a programming
error that surfaces on line 12 of your test run is worth ten of the same error
surfacing three tool calls into a paid agent run.

The registry is also the single place that decides what a model is even told
about. Capability scoping is applied when rendering the tool list, so a
read-only run does not merely refuse to write, it never learns that a write
tool exists.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Iterator
from typing import Any

from jsonschema import Draft202012Validator, ValidationError
from jsonschema.exceptions import SchemaError, best_match

from gantry.errors import (
    CapabilityDenied,
    ToolNotFound,
    ToolRegistrationError,
    ToolValidationError,
)
from gantry.tools.spec import Capability, Grant, ToolHandler, ToolSpec

#: Providers restrict function names. This is the intersection of what OpenAI
#: accepts and what stays readable in a trace.
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")

#: Tool descriptions are prompt tokens on every single turn. A 2,000-character
#: description is not thorough, it is a recurring tax.
MAX_DESCRIPTION_CHARS = 1024


def _version_key(version: str) -> tuple[int, int, int]:
    major, minor, patch = version.split(".")
    return int(major), int(minor), int(patch)


def check_strict_schema(name: str, schema: dict[str, Any]) -> None:
    """Verify a schema satisfies OpenAI strict function-calling rules.

    Strict mode guarantees the model returns arguments that validate, which
    removes a whole class of retry loop. It has three requirements that are
    easy to miss and that only fail once you are already making paid calls:
    the root must be an object, it must set ``additionalProperties: false``,
    and *every* property must be listed in ``required`` - optionality is
    expressed by allowing ``null`` in the type, not by omitting the key.
    """
    if schema.get("type") != "object":
        raise ToolRegistrationError(
            f"tool {name!r}: strict mode requires the root schema to be an object",
            found=schema.get("type"),
        )
    if schema.get("additionalProperties") is not False:
        raise ToolRegistrationError(
            f'tool {name!r}: strict mode requires "additionalProperties": false',
        )
    properties = set(schema.get("properties", {}))
    required = set(schema.get("required", []))
    if properties != required:
        missing = sorted(properties - required)
        extra = sorted(required - properties)
        raise ToolRegistrationError(
            f"tool {name!r}: strict mode requires every property to be listed in "
            "'required' (express optionality with a nullable type instead)",
            missing_from_required=missing,
            required_but_undefined=extra,
        )


def format_validation_error(error: ValidationError) -> str:
    """Render a schema error the way a model can act on.

    The message goes straight back into the conversation as a tool result, so
    it is written for the reader that has to fix it: which field, what was
    wrong, and what was expected.
    """
    path = "/".join(str(part) for part in error.absolute_path)
    location = f"argument '{path}'" if path else "arguments"
    return f"{location}: {error.message}"


class ToolRegistry:
    """A thread-safe, versioned collection of tools."""

    def __init__(self) -> None:
        self._tools: dict[str, dict[str, ToolSpec]] = {}
        self._validators: dict[str, Draft202012Validator] = {}
        self._lock = threading.RLock()

    # -- registration ---------------------------------------------------
    def register(self, spec: ToolSpec, *, replace: bool = False) -> ToolSpec:
        """Validate and add a tool. Raises rather than silently accepting."""
        if not NAME_PATTERN.match(spec.name):
            raise ToolRegistrationError(
                f"invalid tool name {spec.name!r}: expected lowercase snake_case, "
                "1-64 characters, starting with a letter",
                pattern=NAME_PATTERN.pattern,
            )
        if not VERSION_PATTERN.match(spec.version):
            raise ToolRegistrationError(
                f"tool {spec.name!r}: version {spec.version!r} is not major.minor.patch"
            )
        if not spec.description.strip():
            raise ToolRegistrationError(f"tool {spec.name!r}: description is required")
        if len(spec.description) > MAX_DESCRIPTION_CHARS:
            raise ToolRegistrationError(
                f"tool {spec.name!r}: description is {len(spec.description)} characters, "
                f"over the {MAX_DESCRIPTION_CHARS} limit",
            )
        if not callable(spec.handler):
            raise ToolRegistrationError(f"tool {spec.name!r}: handler is not callable")
        if spec.timeout_s <= 0:
            raise ToolRegistrationError(f"tool {spec.name!r}: timeout_s must be positive")

        try:
            # check_schema raises SchemaError (the schema itself is wrong), not
            # ValidationError (an instance is wrong). They are siblings, not
            # parent and child, so catching the wrong one lets a malformed
            # schema through to the first live call.
            Draft202012Validator.check_schema(spec.input_schema)
        except SchemaError as exc:
            raise ToolRegistrationError(
                f"tool {spec.name!r}: input_schema is not a valid JSON Schema",
                reason=exc.message,
            ) from exc

        if spec.strict:
            check_strict_schema(spec.name, spec.input_schema)

        with self._lock:
            versions = self._tools.setdefault(spec.name, {})
            if spec.version in versions and not replace:
                raise ToolRegistrationError(
                    f"tool {spec.qualified_name} is already registered",
                    hint="bump the version, or pass replace=True",
                )
            versions[spec.version] = spec
            self._validators[spec.qualified_name] = Draft202012Validator(spec.input_schema)
        return spec

    def tool(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        *,
        version: str = "1.0.0",
        capabilities: set[Capability] | frozenset[Capability] = frozenset(),
        **options: Any,
    ):
        """Decorator form of :meth:`register`."""

        def decorate(handler: ToolHandler) -> ToolHandler:
            self.register(
                ToolSpec(
                    name=name,
                    version=version,
                    description=description,
                    input_schema=input_schema,
                    handler=handler,
                    capabilities=frozenset(capabilities),
                    **options,
                )
            )
            return handler

        return decorate

    # -- lookup ---------------------------------------------------------
    def get(self, name: str, version: str | None = None) -> ToolSpec:
        """Resolve a tool, defaulting to the highest registered version."""
        with self._lock:
            if "@" in name and version is None:
                name, _, version = name.partition("@")
            versions = self._tools.get(name)
            if not versions:
                raise ToolNotFound(f"no tool named {name!r}", available=sorted(self._tools))
            if version is None:
                version = max(versions, key=_version_key)
            spec = versions.get(version)
            if spec is None:
                raise ToolNotFound(
                    f"tool {name!r} has no version {version!r}",
                    available=sorted(versions),
                )
            return spec

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name.partition("@")[0] in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[ToolSpec]:
        return iter(self.list())

    def list(self, grant: Grant | None = None) -> list[ToolSpec]:
        """Latest version of every tool, optionally filtered by a grant."""
        with self._lock:
            specs = [
                versions[max(versions, key=_version_key)]
                for versions in self._tools.values()
                if versions
            ]
        if grant is not None:
            specs = [s for s in specs if grant.permits(s)]
        return sorted(specs, key=lambda s: s.name)

    # -- authorisation --------------------------------------------------
    def authorize(self, name: str, grant: Grant, version: str | None = None) -> ToolSpec:
        """Resolve a tool and confirm the grant permits it."""
        spec = self.get(name, version)
        if not grant.permits(spec):
            raise CapabilityDenied(
                f"tool {spec.name!r} requires capabilities the current grant does not include",
                tool=spec.name,
                required=sorted(spec.capabilities),
                missing=sorted(grant.missing_for(spec)),
            )
        return spec

    # -- argument validation --------------------------------------------
    def validate_arguments(
        self, name: str, arguments: Any, version: str | None = None
    ) -> dict[str, Any]:
        """Validate call arguments against the tool's schema.

        Returns the arguments on success. On failure raises
        :class:`ToolValidationError` carrying *every* problem, not just the
        first: a model that gets one error per turn needs one turn per error.
        """
        spec = self.get(name, version)
        if not isinstance(arguments, dict):
            raise ToolValidationError(
                f"tool {spec.name!r}: arguments must be a JSON object, "
                f"got {type(arguments).__name__}",
                tool=spec.name,
            )
        validator = self._validators[spec.qualified_name]
        errors = sorted(validator.iter_errors(arguments), key=lambda e: list(e.absolute_path))
        if errors:
            messages = [format_validation_error(e) for e in errors]
            primary = best_match(errors)
            raise ToolValidationError(
                f"tool {spec.name!r}: " + "; ".join(messages),
                tool=spec.name,
                errors=messages,
                primary_path="/".join(str(p) for p in primary.absolute_path) if primary else "",
            )
        return arguments

    # -- provider rendering ---------------------------------------------
    def to_openai_tools(self, grant: Grant | None = None) -> list[dict[str, Any]]:
        """Render the granted tools as an OpenAI ``tools`` array.

        Scoping happens here rather than at call time so an unpermitted tool is
        never described to the model. Refusing a call the model was invited to
        make wastes a turn; not offering it costs nothing.
        """
        return [spec.to_openai_tool() for spec in self.list(grant)]

    def describe(self, grant: Grant | None = None) -> list[dict[str, Any]]:
        """Human- and dashboard-facing summary of the registered tools."""
        return [
            {
                "name": s.name,
                "version": s.version,
                "description": s.description,
                "capabilities": sorted(s.capabilities),
                "timeout_s": s.timeout_s,
                "idempotent": s.idempotent,
                "destructive": s.destructive,
                "concurrency_safe": s.concurrency_safe,
            }
            for s in self.list(grant)
        ]
