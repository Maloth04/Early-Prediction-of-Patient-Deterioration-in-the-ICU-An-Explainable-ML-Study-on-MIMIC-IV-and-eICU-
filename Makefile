.DEFAULT_GOAL := help
PY ?= python

.PHONY: help install dev test lint format check no-data-free clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package and its core dependencies
	$(PY) -m pip install -e .

dev:  ## Install with the test and lint tools
	$(PY) -m pip install -e ".[dev]"

test:  ## Run the test suite (no data, no deep-learning stack, ~2s)
	$(PY) -m pytest -q

lint:  ## Lint, and check formatting
	$(PY) -m ruff check .
	$(PY) -m ruff format --check src tests

format:  ## Apply formatting
	$(PY) -m ruff format src tests
	$(PY) -m ruff check --fix .

check: lint test  ## Everything CI runs, locally

data-free:  ## Run the analyses that need no data at all
	$(PY) analysis/external_validation/precision.py
	$(PY) analysis/external_validation/verify_hm.py

clean:  ## Remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
