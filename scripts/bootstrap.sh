#!/usr/bin/env bash
# Prepare a working environment for Gantry.
#
# Idempotent and quiet on the happy path: creates .venv if missing, installs the
# package in editable mode with its dev extras, and does nothing expensive when
# the environment is already good. Safe to run on every session start.
set -euo pipefail

cd "$(dirname "$0")/.."

VENV="${GANTRY_VENV:-.venv}"
PYTHON="${PYTHON:-python3}"

if [ ! -x "$VENV/bin/python" ]; then
  echo "gantry: creating $VENV" >&2
  "$PYTHON" -m venv "$VENV"
fi

# Already installed and importable? Then there is nothing to do.
if "$VENV/bin/python" -c "import gantry, jsonschema, pytest, openai" >/dev/null 2>&1; then
  exit 0
fi

echo "gantry: installing dependencies" >&2
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet -e ".[dev,api,azure]"
echo "gantry: ready. run tests with: $VENV/bin/python -m pytest" >&2
