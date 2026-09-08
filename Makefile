# Gantry developer tasks.
#
# Every target works from a clean clone: `make test` bootstraps the environment
# first, so there is no separate setup step to forget.

VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.PHONY: help install test test-live lint fmt cov clean

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
	@rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist src/*.egg-info
	@find . -name __pycache__ -type d -prune -exec rm -rf {} +
