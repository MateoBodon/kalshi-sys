PYTHON ?= python

define run_with_uv
	@if command -v uv >/dev/null 2>&1; then \
		uv run $(1); \
	else \
		$(PYTHON) -m $(1); \
	fi
endef

.PHONY: fmt lint typecheck test scan

fmt:
	@if command -v uv >/dev/null 2>&1; then \
		uv run ruff format .; \
	else \
		$(PYTHON) -m ruff format .; \
	fi
	@if command -v uv >/dev/null 2>&1; then \
		uv run ruff check --select I --fix .; \
	else \
		$(PYTHON) -m ruff check --select I --fix .; \
	fi

lint:
	@if command -v uv >/dev/null 2>&1; then \
		uv run ruff check .; \
	else \
		$(PYTHON) -m ruff check .; \
	fi

typecheck:
	@if command -v uv >/dev/null 2>&1; then \
		uv run mypy src tests; \
	else \
		$(PYTHON) -m mypy src tests; \
	fi

test:
	@if command -v uv >/dev/null 2>&1; then \
		uv run pytest; \
	else \
		$(PYTHON) -m pytest; \
	fi

scan:
	$(PYTHON) -m kalshi_alpha.exec.runners.scan_ladders --series CPI --dry-run

.PHONY: paper_live_offline paper_live_online

paper_live_offline:
	$(PYTHON) -m kalshi_alpha.exec.pipelines.week --preset paper_live --offline --report --paper-ledger --broker dry

paper_live_online:
	$(PYTHON) -m kalshi_alpha.exec.pipelines.week --preset paper_live --online --report --paper-ledger --broker dry
