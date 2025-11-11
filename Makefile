PYTHON ?= python

define run_with_uv
	@if command -v uv >/dev/null 2>&1; then \
		uv run $(1); \
	else \
		$(PYTHON) -m $(1); \
	fi
endef

.PHONY: fmt lint typecheck test scan telemetry-smoke report live-smoke monitors pilot-readiness pilot-bundle freshness-smoke ingest-index calibrate-index scan-index-noon scan-index-close micro-index fees-parse collect-polygon-ws backtest-build backtest-hourly backtest-close replay-yesterday aws-calib aws-replay aws-deploy-dashboards parity-ci

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
		uv run pytest && PYTHONPATH=src uv run python -m kalshi_alpha.dev.sanity_check; \
	else \
		$(PYTHON) -m pytest && PYTHONPATH=src $(PYTHON) -m kalshi_alpha.dev.sanity_check; \
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

ingest-index:
	@if [ -z "$(START)" ] || [ -z "$(END)" ]; then \
		echo "Usage: make ingest-index START=YYYY-MM-DD END=YYYY-MM-DD"; exit 1; \
	fi
	$(PYTHON) -m kalshi_alpha.exec.ingest.polygon_index --start $(START) --end $(END) --symbols I:SPX I:NDX

calibrate-index:
	$(PYTHON) -m kalshi_alpha.jobs.calibrate_hourly --series INXU NASDAQ100U
	$(PYTHON) -m kalshi_alpha.jobs.calibrate_close --series INX NASDAQ100

scan-index-noon:
	$(PYTHON) -m kalshi_alpha.exec.scanners.scan_index_noon --series INXU NASDAQ100U --offline --fixtures-root tests/data_fixtures

scan-index-close:
	$(PYTHON) -m kalshi_alpha.exec.scanners.scan_index_close --series INX NASDAQ100 --offline --fixtures-root tests/data_fixtures

micro-index:
	$(PYTHON) -m kalshi_alpha.exec.runners.micro_index --series INXU --offline --fixtures-root tests/data_fixtures --min-ev 0.05 --contracts 1 --regenerate-scoreboard

fees-parse:
	PYTHONPATH=src $(PYTHON) -m kalshi_alpha.dev.parse_fees --pdf docs/kalshi-fee-schedule.pdf --output data/proc/state/fees.json

collect-polygon-ws:
	PYTHONPATH=src $(PYTHON) -m kalshi_alpha.exec.collectors.polygon_ws

backtest-build:
	@if [ -z "$(START)" ] || [ -z "$(END)" ]; then \
		echo "Usage: make backtest-build START=YYYY-MM-DD END=YYYY-MM-DD"; exit 1; \
	fi
	PYTHONPATH=src $(PYTHON) -m kalshi_alpha.backtest.generate_dataset --start $(START) --end $(END)

backtest-hourly:
	PYTHONPATH=src $(PYTHON) -m kalshi_alpha.backtest.score_hourly

backtest-close:
	PYTHONPATH=src $(PYTHON) -m kalshi_alpha.backtest.score_close

replay-yesterday:
	@if command -v $(PYTHON) >/dev/null 2>&1; then \
		REPLAY_DATE=$$($(PYTHON) -c 'from datetime import datetime, timedelta; from zoneinfo import ZoneInfo; print((datetime.now(tz=ZoneInfo("America/New_York")) - timedelta(days=1)).strftime("%Y-%m-%d"))'); \
	else \
		echo "Python interpreter not found; cannot compute replay date"; exit 1; \
	fi; \
	REPLAY_FILE="data/replay/$${REPLAY_DATE}_spx_ndx.json"; \
	if [ ! -f "$${REPLAY_FILE}" ]; then \
		echo "Replay file missing: $${REPLAY_FILE}"; exit 1; \
	fi; \
	PYTHONPATH=src $(PYTHON) -m kalshi_alpha.replay.polygon_index_replay --file "$${REPLAY_FILE}" --speed 10 --start "11:40" --end "12:05" --summary "reports/_artifacts/replay/polygon_index_replay_noon.json" --proc-parquet "reports/_artifacts/replay/polygon_replay_noon.parquet"; \
	PYTHONPATH=src $(PYTHON) -m kalshi_alpha.replay.polygon_index_replay --file "$${REPLAY_FILE}" --speed 10 --start "15:45" --end "16:05" --summary "reports/_artifacts/replay/polygon_index_replay_close.json" --proc-parquet "reports/_artifacts/replay/polygon_replay_close.parquet"

aws-calib:
	$(PYTHON) scripts/aws_job.py --job calib_hourly --command "$(PYTHON) -m jobs.calib_hourly --series INXU NASDAQ100U" --artifact data/proc/calib/index

aws-replay:
	@REPLAY_DATE=$$($(PYTHON) -c 'from datetime import datetime, timedelta; from zoneinfo import ZoneInfo; print((datetime.now(tz=ZoneInfo("America/New_York")) - timedelta(days=1)).strftime("%Y-%m-%d"))'); \
	REPLAY_FILE=$${FILE:-data/replay/$${REPLAY_DATE}_spx_ndx.json}; \
	if [ ! -f "$$REPLAY_FILE" ]; then echo "Replay file missing: $$REPLAY_FILE"; exit 1; fi; \
	$(PYTHON) scripts/aws_job.py --job replay --command "PYTHONPATH=src $(PYTHON) -m kalshi_alpha.replay.polygon_index_replay --file $$REPLAY_FILE --speed 20 --start 11:40 --end 12:05" --artifact reports/_artifacts/replay_ev.parquet

aws-deploy-dashboards:
	@mkdir -p reports/_artifacts/monitors
	@echo "Bundling monitor artifacts for dashboard deploy..."
	@tar -czf reports/_artifacts/monitors/dashboard_bundle.tgz reports/_artifacts/monitors >/dev/null 2>&1 || true
	@ls -lh reports/_artifacts/monitors/dashboard_bundle.tgz

parity-ci:
	$(PYTHON) scripts/parity_gate.py --threshold 0.15

.PHONY: paper_live_offline paper_live_online

paper_live_offline:
	$(PYTHON) -m kalshi_alpha.exec.pipelines.week --preset paper_live --offline --report --paper-ledger --broker dry

paper_live_online:
	$(PYTHON) -m kalshi_alpha.exec.pipelines.week --preset paper_live --online --report --paper-ledger --broker dry
