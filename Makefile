PYTHON ?= python

define run_with_uv
	@if command -v uv >/dev/null 2>&1; then \
		uv run $(1); \
	else \
		$(PYTHON) -m $(1); \
	fi
endef

.PHONY: fmt lint typecheck test scan telemetry-smoke report live-smoke monitors pilot-readiness pilot-bundle freshness-smoke

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
		uv run pytest && uv run python -m kalshi_alpha.dev.sanity_check; \
	else \
		$(PYTHON) -m pytest && $(PYTHON) -m kalshi_alpha.dev.sanity_check; \
	fi

scan:
	$(PYTHON) -m kalshi_alpha.exec.runners.scan_ladders --series CPI --dry-run

telemetry-smoke:
	$(PYTHON) -c 'from datetime import UTC, datetime; from kalshi_alpha.exec.telemetry.sink import TelemetrySink; sink = TelemetrySink(); now = datetime.now(tz=UTC); sink.emit("sent", source="make.telemetry", data={"order_id": "SIM-001", "side": "YES", "contracts": 10, "timestamp": now.isoformat()}); sink.emit("fill", source="make.telemetry", data={"order_id": "SIM-001", "filled": 10, "price": 0.45, "latency_ms": 180}); sink.emit("heartbeat", source="make.telemetry", data={"ws_state": "open", "seq": 1})'

report:
	$(PYTHON) -m kalshi_alpha.exec.scoreboard

monitors:
	@if command -v uv >/dev/null 2>&1; then \
		uv run python -m kalshi_alpha.exec.monitors.cli; \
	else \
		$(PYTHON) -m kalshi_alpha.exec.monitors.cli; \
	fi

pilot-readiness:
	@if command -v uv >/dev/null 2>&1; then \
		uv run python -m kalshi_alpha.exec.reports.ramp; \
	else \
		$(PYTHON) -m kalshi_alpha.exec.reports.ramp; \
	fi

pilot-bundle:
	@if command -v uv >/dev/null 2>&1; then \
		uv run python -m kalshi_alpha.exec.pilot_bundle; \
	else \
		$(PYTHON) -m kalshi_alpha.exec.pilot_bundle; \
	fi

freshness-smoke:
	@if command -v uv >/dev/null 2>&1; then \
		uv run python -m kalshi_alpha.exec.monitors.freshness --print; \
	else \
		$(PYTHON) -m kalshi_alpha.exec.monitors.freshness --print; \
	fi

live-smoke:
	PYTHONPATH=src $(PYTHON) -m kalshi_alpha.dev.sanity_check --live-smoke --env $${KALSHI_ENV:-prod}

.PHONY: paper_live_offline paper_live_online

paper_live_offline:
	$(PYTHON) -m kalshi_alpha.exec.pipelines.week --preset paper_live --offline --report --paper-ledger --broker dry

paper_live_online:
	$(PYTHON) -m kalshi_alpha.exec.pipelines.week --preset paper_live --online --report --paper-ledger --broker dry
