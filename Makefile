# Gantry developer tasks.
#
# Every target works from a clean clone: `make test` bootstraps the environment
# first, so there is no separate setup step to forget.

VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.PHONY: help install test test-live lint fmt cov clean demo

help: ## Show the available targets
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-12s %s\n", $$1, $$2}'

install: ## Create the virtualenv and install the package with dev extras
	@./scripts/bootstrap.sh

test: install ## Run the test suite (live tests deselected)
	@$(PY) -m pytest -m "not live"

test-live: install ## Run the tests that need real Azure credentials
	@$(PY) -m pytest -m live -rs

cov: install ## Run the suite with a coverage report
	@$(PY) -m pytest -m "not live" --cov=gantry --cov-report=term-missing

lint: install ## Check formatting and lint rules
	@$(VENV)/bin/ruff check src tests
	@$(VENV)/bin/ruff format --check src tests

fmt: install ## Apply formatting and safe lint fixes
	@$(VENV)/bin/ruff check --fix src tests
	@$(VENV)/bin/ruff format src tests

clean: ## Remove caches and build artefacts
	@rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist .demo src/*.egg-info
	@find . -name __pycache__ -type d -prune -exec rm -rf {} +

demo: install ## Run the CLI end to end against a throwaway fixture
	@rm -rf .demo && mkdir -p .demo
	@printf 'def add(a, b):\n    return a - b\n' > .demo/calc.py
	@printf 'from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n' > .demo/test_calc.py
	$(VENV)/bin/gantry tools --grant read-only
	-$(VENV)/bin/gantry run "fix the failing test" -w .demo \
		--gate 'tests=python -m pytest -q' --db .demo/gantry.db
	$(VENV)/bin/gantry trace --db .demo/gantry.db
