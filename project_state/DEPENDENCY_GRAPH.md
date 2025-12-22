# Dependency Graph

## Metadata
- Generated: 2025-12-22T19:42:20Z
- Git SHA: a907a2eed87531d8178c3dc183d6f070182f9ebe
- Branch: codex/TICKET-000_project_state_rebuild
- Commands: `python tools/project_state_build.py`, `rg --files`, `sed -n '1,200p' README.md`, `sed -n '1,200p' docs/PROGRESS.md`, `sed -n '1,200p' CHANGELOG.md`, `sed -n '1,200p' pyproject.toml`, `sed -n '1,200p' Makefile`

## Source
- Internal import adjacency list in `project_state/_generated/import_graph.json`.
- Only src/tools/experiments Python files are analyzed.

## High-degree modules (top 15 by internal imports)
- `src/kalshi_alpha/exec/runners/scan_ladders.py` — 63 internal deps
- `src/kalshi_alpha/exec/pipelines/daily.py` — 26 internal deps
- `src/kalshi_alpha/core/archive/replay.py` — 15 internal deps
- `src/kalshi_alpha/exec/supervisor.py` — 11 internal deps
- `src/kalshi_alpha/datastore/ingest.py` — 11 internal deps
- `src/kalshi_alpha/brokers/kalshi/live.py` — 11 internal deps
- `src/kalshi_alpha/exec/scanners/index_scan_common.py` — 10 internal deps
- `src/kalshi_alpha/exec/runners/micro_index.py` — 10 internal deps
- `src/kalshi_alpha/exec/scanners/scan_index_hourly.py` — 9 internal deps
- `src/kalshi_alpha/exec/scanners/scan_index_close.py` — 9 internal deps
- `src/kalshi_alpha/replay/polygon_index_replay.py` — 8 internal deps
- `src/kalshi_alpha/exec/ledger/__init__.py` — 8 internal deps
- `src/kalshi_alpha/strategies/teny/__init__.py` — 7 internal deps
- `src/kalshi_alpha/strategies/index/backtest_polygon.py` — 7 internal deps
- `src/kalshi_alpha/strategies/cpi/__init__.py` — 7 internal deps

## Core entrypoints and their internal dependencies
### src/kalshi_alpha/exec/runners/scan_ladders.py
- `src/kalshi_alpha/brokers/__init__.py`
- `src/kalshi_alpha/brokers/kalshi/base.py`
- `src/kalshi_alpha/config/__init__.py`
- `src/kalshi_alpha/core/__init__.py`
- `src/kalshi_alpha/core/archive/__init__.py`
- `src/kalshi_alpha/core/execution/fillprob.py`
- `src/kalshi_alpha/core/execution/fillratio.py`
- `src/kalshi_alpha/core/execution/index_models.py`
- `src/kalshi_alpha/core/execution/slippage.py`
- `src/kalshi_alpha/core/fees/__init__.py`
- `src/kalshi_alpha/core/fees/index_series.py`
- `src/kalshi_alpha/core/gates/__init__.py`
- `src/kalshi_alpha/core/kalshi_api/__init__.py`
- `src/kalshi_alpha/core/kalshi_ws.py`
- `src/kalshi_alpha/core/pricing/__init__.py`
- `src/kalshi_alpha/core/pricing/align.py`
- `src/kalshi_alpha/core/risk/__init__.py`
- `src/kalshi_alpha/core/risk/drawdown.py`
- `src/kalshi_alpha/core/sizing/__init__.py`
- `src/kalshi_alpha/data/__init__.py`
- `src/kalshi_alpha/datastore/paths.py`
- `src/kalshi_alpha/drivers/__init__.py`
- `src/kalshi_alpha/drivers/aaa_gas/__init__.py`
- `src/kalshi_alpha/drivers/aaa_gas/fetch.py`
- `src/kalshi_alpha/drivers/aaa_gas/ingest.py`
- `src/kalshi_alpha/drivers/calendar/loader.py`
- `src/kalshi_alpha/drivers/macro_calendar/__init__.py`
- `src/kalshi_alpha/drivers/polygon_index/client.py`
- `src/kalshi_alpha/drivers/polygon_index/symbols.py`
- `src/kalshi_alpha/exec/__init__.py`
- `src/kalshi_alpha/exec/collectors/tob_logger.py`
- `src/kalshi_alpha/exec/fees.py`
- `src/kalshi_alpha/exec/gate_utils.py`
- `src/kalshi_alpha/exec/heartbeat.py`
- `src/kalshi_alpha/exec/index_paper_ledger.py`
- `src/kalshi_alpha/exec/ledger/__init__.py`
- `src/kalshi_alpha/exec/limits.py`
- `src/kalshi_alpha/exec/monitors/__init__.py`
- `src/kalshi_alpha/exec/monitors/fee_rules.py`
- `src/kalshi_alpha/exec/monitors/freshness.py`
- `src/kalshi_alpha/exec/monitors/sigma_drift.py`
- `src/kalshi_alpha/exec/monitors/summary.py`
- `src/kalshi_alpha/exec/pilot/__init__.py`
- `src/kalshi_alpha/exec/quote_microprice.py`
- `src/kalshi_alpha/exec/quote_optim.py`
- `src/kalshi_alpha/exec/reports/__init__.py`
- `src/kalshi_alpha/exec/scanners/__init__.py`
- `src/kalshi_alpha/exec/scanners/cpi.py`
- `src/kalshi_alpha/exec/scanners/utils.py`
- `src/kalshi_alpha/exec/state/orders.py`
- `src/kalshi_alpha/markets/discovery.py`
- `src/kalshi_alpha/risk/__init__.py`
- `src/kalshi_alpha/risk/var_index.py`
- `src/kalshi_alpha/sched/__init__.py`
- `src/kalshi_alpha/sched/regimes.py`
- `src/kalshi_alpha/strategies/__init__.py`
- `src/kalshi_alpha/strategies/claims/__init__.py`
- `src/kalshi_alpha/strategies/cpi/__init__.py`
- `src/kalshi_alpha/strategies/index/__init__.py`
- `src/kalshi_alpha/strategies/teny/__init__.py`
- `src/kalshi_alpha/strategies/weather/__init__.py`
- `src/kalshi_alpha/structures/__init__.py`
- `src/kalshi_alpha/utils/env.py`

### src/kalshi_alpha/exec/supervisor_index.py
- `src/kalshi_alpha/data/__init__.py`
- `src/kalshi_alpha/drivers/polygon_index_ws.py`
- `src/kalshi_alpha/exec/collectors/tob_logger.py`
- `src/kalshi_alpha/exec/preflight_index.py`
- `src/kalshi_alpha/exec/runners/__init__.py`
- `src/kalshi_alpha/exec/runners/micro_index.py`
- `src/kalshi_alpha/sched/__init__.py`

### src/kalshi_alpha/exec/preflight_index.py
- `src/kalshi_alpha/datastore/paths.py`
- `src/kalshi_alpha/exec/heartbeat.py`
- `src/kalshi_alpha/strategies/index/model_polygon.py`
- `src/kalshi_alpha/utils/env.py`
- `src/kalshi_alpha/utils/keys.py`

### src/kalshi_alpha/exec/pipelines/daily.py
- `src/kalshi_alpha/core/archive/scorecards.py`
- `src/kalshi_alpha/core/execution/fillratio.py`
- `src/kalshi_alpha/core/execution/slippage.py`
- `src/kalshi_alpha/core/gates/__init__.py`
- `src/kalshi_alpha/core/kalshi_api/__init__.py`
- `src/kalshi_alpha/core/pricing/align.py`
- `src/kalshi_alpha/core/risk/__init__.py`
- `src/kalshi_alpha/core/risk/drawdown.py`
- `src/kalshi_alpha/datastore/ingest.py`
- `src/kalshi_alpha/datastore/paths.py`
- `src/kalshi_alpha/drivers/__init__.py`
- `src/kalshi_alpha/drivers/macro_calendar/__init__.py`
- `src/kalshi_alpha/exec/gate_utils.py`
- `src/kalshi_alpha/exec/heartbeat.py`
- `src/kalshi_alpha/exec/ledger/__init__.py`
- `src/kalshi_alpha/exec/pipelines/calendar.py`
- `src/kalshi_alpha/exec/pipelines/today.py`
- `src/kalshi_alpha/exec/reports/__init__.py`
- `src/kalshi_alpha/exec/runners/scan_ladders.py`
- `src/kalshi_alpha/exec/state/orders.py`
- `src/kalshi_alpha/strategies/claims/__init__.py`
- `src/kalshi_alpha/strategies/cpi/__init__.py`
- `src/kalshi_alpha/strategies/teny/__init__.py`
- `src/kalshi_alpha/strategies/weather/__init__.py`
- `src/kalshi_alpha/utils/family.py`
- `src/kalshi_alpha/utils/secrets.py`

### src/kalshi_alpha/exec/pipelines/today.py
- `src/kalshi_alpha/exec/heartbeat.py`
- `src/kalshi_alpha/exec/pipelines/calendar.py`
- `src/kalshi_alpha/exec/pipelines/daily.py`
- `src/kalshi_alpha/exec/state/orders.py`
- `src/kalshi_alpha/utils/family.py`

### src/kalshi_alpha/exec/pipelines/week.py
- `src/kalshi_alpha/datastore/paths.py`
- `src/kalshi_alpha/exec/heartbeat.py`
- `src/kalshi_alpha/exec/pipelines/calendar.py`
- `src/kalshi_alpha/exec/pipelines/daily.py`
- `src/kalshi_alpha/exec/state/orders.py`
- `src/kalshi_alpha/utils/family.py`

### src/kalshi_alpha/exec/scoreboard.py
- `src/kalshi_alpha/core/execution/defaults.py`
- `src/kalshi_alpha/core/execution/index_models.py`
- `src/kalshi_alpha/exec/__init__.py`
- `src/kalshi_alpha/exec/monitors/freshness.py`
- `src/kalshi_alpha/exec/pilot_readiness.py`
- `src/kalshi_alpha/exec/slo.py`
- `src/kalshi_alpha/utils/family.py`

### src/kalshi_alpha/exec/monitors/cli.py
- `src/kalshi_alpha/exec/heartbeat.py`
- `src/kalshi_alpha/exec/monitors/runtime.py`

### src/kalshi_alpha/exec/runners/pilot.py
- `src/kalshi_alpha/exec/runners/__init__.py`
- `src/kalshi_alpha/exec/runners/scan_ladders.py`
