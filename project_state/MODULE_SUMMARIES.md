# Module Summaries

## Metadata
- Generated: 2025-12-22T19:42:20Z
- Git SHA: a907a2eed87531d8178c3dc183d6f070182f9ebe
- Branch: codex/TICKET-000_project_state_rebuild
- Commands: `python tools/project_state_build.py`, `rg --files`, `sed -n '1,200p' README.md`, `sed -n '1,200p' docs/PROGRESS.md`, `sed -n '1,200p' CHANGELOG.md`, `sed -n '1,200p' pyproject.toml`, `sed -n '1,200p' Makefile`

## Index
- Summaries are grouped by top-level package.
- Counts include top-level classes/functions only.
- Source of truth: `project_state/_generated/symbol_index.json`.

## __init__.py
- `src/kalshi_alpha/__init__.py` — funcs=0, classes=0; doc: Kalshi Alpha namespace package.

## backtest
- `src/kalshi_alpha/backtest/__init__.py` — funcs=0, classes=0; doc: Backtesting utilities for index ladder strategies.
- `src/kalshi_alpha/backtest/generate_dataset.py` — funcs=9, classes=1; doc: Build minute-level backtest dataset for index ladders.
- `src/kalshi_alpha/backtest/index_calendar.py` — funcs=8, classes=2; doc: Trading calendar helpers for index backtests.
- `src/kalshi_alpha/backtest/score_close.py` — funcs=2, classes=0; doc: Evaluate close index calibrations against historical dataset.
- `src/kalshi_alpha/backtest/score_hourly.py` — funcs=2, classes=0; doc: Evaluate hourly index calibrations against historical dataset.
- `src/kalshi_alpha/backtest/scoring.py` — funcs=12, classes=2; doc: Shared scoring utilities for index backtests.

## brokers
- `src/kalshi_alpha/brokers/__init__.py` — funcs=1, classes=0; doc: Factory helpers for broker adapters.
- `src/kalshi_alpha/brokers/kalshi/base.py` — funcs=1, classes=2; doc: Shared interfaces for Kalshi broker adapters.
- `src/kalshi_alpha/brokers/kalshi/dry.py` — funcs=0, classes=2; doc: Dry-run broker adapter that records intended Kalshi orders.
- `src/kalshi_alpha/brokers/kalshi/endpoints.py` — funcs=1, classes=1; doc: Kalshi environment endpoint helpers.
- `src/kalshi_alpha/brokers/kalshi/http_client.py` — funcs=2, classes=5; doc: Shared Kalshi HTTP client implementing RSA-PSS auth, retries, and structured logging.
- `src/kalshi_alpha/brokers/kalshi/live.py` — funcs=7, classes=2; doc: Live Kalshi broker adapter with rate limiting, backoff, and auditing.
- `src/kalshi_alpha/brokers/kalshi/ws_client.py` — funcs=3, classes=2; doc: Authenticated Kalshi WebSocket client with reconnect/backoff.

## config
- `src/kalshi_alpha/config/__init__.py` — funcs=0, classes=0; doc: Configuration loaders for strategy- and scanner-level metadata.
- `src/kalshi_alpha/config/index_ops.py` — funcs=4, classes=2; doc: Load shared operational window configuration for index ladder strategies.
- `src/kalshi_alpha/config/index_rules.py` — funcs=4, classes=2; doc: Load index ladder rule semantics from the markdown rulebook.
- `src/kalshi_alpha/config/size_ladder.py` — funcs=1, classes=3; doc: Size ladder configuration for staged lot/bin limits.

## core
- `src/kalshi_alpha/core/__init__.py` — funcs=0, classes=0; doc: Core primitives shared across Kalshi Alpha components.
- `src/kalshi_alpha/core/archive/__init__.py` — funcs=0, classes=0; doc: Archiving and replay utilities for Kalshi public market data.
- `src/kalshi_alpha/core/archive/archiver.py` — funcs=3, classes=0; doc: Archive Kalshi public data snapshots for later replay.
- `src/kalshi_alpha/core/archive/replay.py` — funcs=15, classes=1; doc: Replay archived Kalshi data to recompute proposal EVs.
- `src/kalshi_alpha/core/archive/scorecards.py` — funcs=5, classes=1; doc: Replay scorecard computation from archived manifests and proposals.
- `src/kalshi_alpha/core/backtest/__init__.py` — funcs=7, classes=2; doc: Backtesting utilities: scoring rules, event replay, and calibration diagnostics.
- `src/kalshi_alpha/core/datastore/__init__.py` — funcs=2, classes=3; doc: Datastore helpers for raw snapshots, processed tables, and DuckDB cataloging.
- `src/kalshi_alpha/core/execution/defaults.py` — funcs=4, classes=0; doc: Execution defaults for index ladder maker behaviour (alpha & slippage).
- `src/kalshi_alpha/core/execution/fillprob.py` — funcs=3, classes=0; doc: Load conservative fill probability curves derived from TOB snapshots.
- `src/kalshi_alpha/core/execution/fillratio.py` — funcs=7, classes=1; doc: Estimate expected fills using simple visible-depth heuristics.
- `src/kalshi_alpha/core/execution/index_models.py` — funcs=4, classes=2; doc: Execution curve loaders for index fills and slippage.
- `src/kalshi_alpha/core/execution/order_queue.py` — funcs=0, classes=2; doc: FIFO order queue for coordinating cancel/replace operations.
- `src/kalshi_alpha/core/execution/series_utils.py` — funcs=1, classes=0; doc: Helpers for canonicalizing execution series names.
- `src/kalshi_alpha/core/execution/slippage.py` — funcs=12, classes=2; doc: Slippage modelling utilities for paper execution.
- `src/kalshi_alpha/core/fees/__init__.py` — funcs=9, classes=1; doc: Kalshi fee schedule utilities loaded from the canonical JSON configuration.
- `src/kalshi_alpha/core/fees/index_series.py` — funcs=4, classes=1; doc: Index series fee curve loader.
- `src/kalshi_alpha/core/gates/__init__.py` — funcs=0, classes=0; doc: Quality gate evaluation utilities.
- `src/kalshi_alpha/core/gates/quality_gates.py` — funcs=7, classes=6; doc: Production quality gates for daily orchestration.
- `src/kalshi_alpha/core/kalshi_api/__init__.py` — funcs=2, classes=5; doc: Read-only Kalshi public market-data client.
- `src/kalshi_alpha/core/kalshi_ws.py` — funcs=8, classes=3; doc: Utilities for consuming Kalshi orderbook websockets and deriving imbalance metrics.
- `src/kalshi_alpha/core/pricing/__init__.py` — funcs=11, classes=4; doc: Ladder pricing utilities: survival curve projection, PMF extraction, and EV analysis.
- `src/kalshi_alpha/core/pricing/align.py` — funcs=6, classes=1; doc: Utilities for aligning model PMFs to ladder strike grids.
- `src/kalshi_alpha/core/pricing/mispricing.py` — funcs=3, classes=1; doc: Ladder mispricing analytics and spread detection.
- `src/kalshi_alpha/core/risk/__init__.py` — funcs=1, classes=5; doc: Risk utilities including PAL (Position and Loss) policy enforcement.
- `src/kalshi_alpha/core/risk/drawdown.py` — funcs=6, classes=1; doc: Persistent drawdown guard for paper PnL caps.
- `src/kalshi_alpha/core/sizing/__init__.py` — funcs=0, classes=0; doc: Sizing utilities for portfolio-aware ladder execution.
- `src/kalshi_alpha/core/sizing/kelly.py` — funcs=4, classes=0; doc: Kelly-based sizing helpers with safety caps.
- `src/kalshi_alpha/core/ws.py` — funcs=0, classes=1; doc: Lightweight Kalshi websocket client with RSA-PSS authentication.

## data
- `src/kalshi_alpha/data/__init__.py` — funcs=0, classes=0; doc: Data-plane helpers (websocket sentries, etc.).
- `src/kalshi_alpha/data/failover.py` — funcs=1, classes=2; doc: SLO-enforced dual-feed failover controller.
- `src/kalshi_alpha/data/ws_sentry.py` — funcs=1, classes=1; doc: Lightweight websocket freshness sentry with strict (final-minute) gating.

## datastore
- `src/kalshi_alpha/datastore/ingest.py` — funcs=9, classes=1; doc: CLI to ingest data snapshots across drivers.
- `src/kalshi_alpha/datastore/paths.py` — funcs=0, classes=0; doc: Shared datastore paths.
- `src/kalshi_alpha/datastore/snapshots.py` — funcs=5, classes=0; doc: Helpers for writing raw datastore snapshots.

## dev
- `src/kalshi_alpha/dev/__init__.py` — funcs=0, classes=0; doc: Developer utilities such as repository hygiene checks.
- `src/kalshi_alpha/dev/imbalance_snap.py` — funcs=7, classes=0; doc: Capture Kalshi orderbook imbalance metrics for the TENY close window.
- `src/kalshi_alpha/dev/parse_fees.py` — funcs=7, classes=0; doc: Parse Kalshi fee schedule PDF and emit normalized JSON configuration.
- `src/kalshi_alpha/dev/sanity_check.py` — funcs=6, classes=0; doc: Repository hygiene check with optional live smoke test.
- `src/kalshi_alpha/dev/ws_smoke.py` — funcs=5, classes=0; doc: Developer CLI to smoke-test Kalshi websocket imbalance streaming.

## drivers
- `src/kalshi_alpha/drivers/__init__.py` — funcs=0, classes=0; doc: Data drivers ingesting macro and settlement data sources.
- `src/kalshi_alpha/drivers/aaa_gas/__init__.py` — funcs=1, classes=1; doc: AAA gasoline price driver stub (fixture backed).
- `src/kalshi_alpha/drivers/aaa_gas/fetch.py` — funcs=6, classes=1; doc: AAA gasoline national average fetcher.
- `src/kalshi_alpha/drivers/aaa_gas/ingest.py` — funcs=3, classes=0; doc: AAA gasoline bootstrap ingestion.
- `src/kalshi_alpha/drivers/bls_cpi/__init__.py` — funcs=4, classes=1; doc: BLS CPI driver with online/offline support.
- `src/kalshi_alpha/drivers/calendar/__init__.py` — funcs=0, classes=0; doc: Minimal event calendar helpers for index ladders.
- `src/kalshi_alpha/drivers/calendar/loader.py` — funcs=5, classes=1; doc: Load and query minimal macro event calendar metadata.
- `src/kalshi_alpha/drivers/cleveland_nowcast/__init__.py` — funcs=7, classes=1; doc: Cleveland Fed inflation nowcast driver.
- `src/kalshi_alpha/drivers/dol_claims/__init__.py` — funcs=3, classes=1; doc: DOL ETA-539 weekly claims driver.
- `src/kalshi_alpha/drivers/index_polygon.py` — funcs=7, classes=0; doc: Polygon index history loader for offline modelling/backtests.
- `src/kalshi_alpha/drivers/kalshi_index_history.py` — funcs=6, classes=1; doc: Load historical Kalshi index ladder quotes from disk for offline backtests.
- `src/kalshi_alpha/drivers/macro_calendar/__init__.py` — funcs=10, classes=0; doc: Macro calendar driver producing release-day dummy variables.
- `src/kalshi_alpha/drivers/macro_calendar/cli.py` — funcs=7, classes=0; doc: Command-line utility to refresh macro calendar day dummies.
- `src/kalshi_alpha/drivers/nws_cli/__init__.py` — funcs=9, classes=2; doc: NOAA/NWS Daily Climate Report (DCR) driver.
- `src/kalshi_alpha/drivers/polygon_index/__init__.py` — funcs=0, classes=0; doc: Polygon index data integration.
- `src/kalshi_alpha/drivers/polygon_index/client.py` — funcs=1, classes=4; doc: Massive indices client supporting REST ingestion and optional Massive websocket.
- `src/kalshi_alpha/drivers/polygon_index/snapshots.py` — funcs=6, classes=0; doc: Snapshot helpers for Polygon index data.
- `src/kalshi_alpha/drivers/polygon_index/symbols.py` — funcs=3, classes=1; doc: Shared symbol metadata for Polygon-powered index ladders.
- `src/kalshi_alpha/drivers/polygon_index/windowed.py` — funcs=10, classes=4; doc: Windowed Polygon index websocket collector for hourly and close ladders.
- `src/kalshi_alpha/drivers/polygon_index_ws.py` — funcs=9, classes=2; doc: Shared Polygon index websocket helper with singleton lifecycle and metrics.
- `src/kalshi_alpha/drivers/treasury_yields/__init__.py` — funcs=12, classes=1; doc: Treasury par yield driver with offline support.

## exec
- `src/kalshi_alpha/exec/__init__.py` — funcs=0, classes=0; doc: Execution scaffolding for dry-run ladder scanning.
- `src/kalshi_alpha/exec/backtest_index_polygon.py` — funcs=3, classes=0; doc: CLI entrypoint for Polygon-only index ladder backtests.
- `src/kalshi_alpha/exec/brokers/__init__.py` — funcs=0, classes=0; doc: Broker integration placeholders (dry-run only).
- `src/kalshi_alpha/exec/brokers/kalshi/__init__.py` — funcs=0, classes=0; doc: Placeholder for future authenticated Kalshi broker integration.
- `src/kalshi_alpha/exec/collectors/__init__.py` — funcs=0, classes=0; doc: Collectors for external data feeds.
- `src/kalshi_alpha/exec/collectors/kalshi_tob.py` — funcs=6, classes=0; doc: Capture Kalshi top-of-book snapshots for fill modeling.
- `src/kalshi_alpha/exec/collectors/polygon_ws.py` — funcs=10, classes=3; doc: Massive (Polygon) websocket collector for index ladders.
- `src/kalshi_alpha/exec/collectors/tob_logger.py` — funcs=9, classes=1; doc: Bounded top-of-book snapshot + quote-intent logger for index ladders.
- `src/kalshi_alpha/exec/fees.py` — funcs=5, classes=2; doc: Execution-time fee helpers backed by configs/fees.json.
- `src/kalshi_alpha/exec/gate_utils.py` — funcs=2, classes=0; doc: Shared helpers for quality gate configuration and artifacts.
- `src/kalshi_alpha/exec/heartbeat.py` — funcs=8, classes=0; doc: Heartbeat and kill-switch utilities for execution pipelines.
- `src/kalshi_alpha/exec/housekeep.py` — funcs=7, classes=1; doc: Artifact retention housekeeping utility.
- `src/kalshi_alpha/exec/index_paper_ledger.py` — funcs=5, classes=0; doc: Lightweight JSONL ledger for index paper (dry) trades.
- `src/kalshi_alpha/exec/ingest/polygon_index.py` — funcs=3, classes=0; doc: Download Polygon index history into the raw datastore.
- `src/kalshi_alpha/exec/ledger/__init__.py` — funcs=9, classes=3; doc: Paper trading ledger utilities.
- `src/kalshi_alpha/exec/ledger/aggregate.py` — funcs=4, classes=0; doc: Aggregate paper ledger CSV outputs into a single Parquet dataset.
- `src/kalshi_alpha/exec/ledger/schema.py` — funcs=0, classes=1; doc: Typed ledger schema definitions.
- `src/kalshi_alpha/exec/limits.py` — funcs=0, classes=3; doc: Limit enforcement helpers for proposal generation and broker validation.
- `src/kalshi_alpha/exec/live_smoke.py` — funcs=8, classes=1; doc: Read-only smoke test for Kalshi index ladders.
- `src/kalshi_alpha/exec/monitors/__init__.py` — funcs=0, classes=0; doc: Runtime monitoring entry points.
- `src/kalshi_alpha/exec/monitors/cli.py` — funcs=4, classes=0; doc: Command-line entry point for runtime monitors.
- `src/kalshi_alpha/exec/monitors/fee_rules.py` — funcs=2, classes=0; doc: Helper utilities for fee/rule watcher artifacts.
- `src/kalshi_alpha/exec/monitors/freshness.py` — funcs=30, classes=2; doc: Data feed freshness monitor for ramp readiness.
- `src/kalshi_alpha/exec/monitors/runtime.py` — funcs=16, classes=2; doc: Compute runtime execution monitors and persist artifacts.
- `src/kalshi_alpha/exec/monitors/sequential.py` — funcs=1, classes=2; doc: Sequential change-detection guardrails for EV deltas.
- `src/kalshi_alpha/exec/monitors/sigma_drift.py` — funcs=2, classes=0; doc: Helpers for sigma drift monitor artifacts.
- `src/kalshi_alpha/exec/monitors/summary.py` — funcs=3, classes=1; doc: Helpers for aggregating persisted monitor artifacts.
- `src/kalshi_alpha/exec/pilot/__init__.py` — funcs=0, classes=0; doc: Pilot execution helpers.
- `src/kalshi_alpha/exec/pilot/config.py` — funcs=3, classes=1; doc: Utilities for loading pilot mode configuration.
- `src/kalshi_alpha/exec/pilot/runtime.py` — funcs=6, classes=1; doc: Runtime helpers for configuring pilot sessions.
- `src/kalshi_alpha/exec/pilot_bundle.py` — funcs=11, classes=1; doc: Assemble a single tarball with key pilot readiness artifacts.
- `src/kalshi_alpha/exec/pilot_readiness.py` — funcs=11, classes=1; doc: Compute pilot readiness for index ladders based on recent paper fills.
- `src/kalshi_alpha/exec/pipelines/calendar.py` — funcs=11, classes=1; doc: Calendar-aware run window resolution for daily pipelines.
- `src/kalshi_alpha/exec/pipelines/daily.py` — funcs=19, classes=0; doc: Daily orchestration pipeline for ladder strategies.
- `src/kalshi_alpha/exec/pipelines/preflight.py` — funcs=2, classes=0; doc: Lightweight preflight helper for daily pipeline windows.
- `src/kalshi_alpha/exec/pipelines/today.py` — funcs=11, classes=1; doc: Autonomous "today" orchestration that selects daily modes based on calendars.
- `src/kalshi_alpha/exec/pipelines/week.py` — funcs=13, classes=1; doc: Weekly orchestration wrapper running daily pipeline modes in sequence.
- `src/kalshi_alpha/exec/policy/freeze.py` — funcs=3, classes=1; doc: Utilities for evaluating pre-event freeze windows per series family.
- `src/kalshi_alpha/exec/preflight_index.py` — funcs=12, classes=1; doc: GO/NO-GO checks for SPX/NDX index ladder windows.
- `src/kalshi_alpha/exec/quote_microprice.py` — funcs=1, classes=2; doc: Microprice calculator and replacement throttle for ladder quoting.
- `src/kalshi_alpha/exec/quote_optim.py` — funcs=0, classes=2; doc: Quote optimization utilities: PMF skew gating, microprice bias, freshness widening.
- `src/kalshi_alpha/exec/reports/__init__.py` — funcs=10, classes=0; doc: Generate markdown reports for ladder scans.
- `src/kalshi_alpha/exec/reports/ramp.py` — funcs=19, classes=1; doc: Compute pilot ramp readiness reports.
- `src/kalshi_alpha/exec/runners/__init__.py` — funcs=0, classes=0; doc: Command-line runners for ladder scanning workflows.
- `src/kalshi_alpha/exec/runners/micro_index.py` — funcs=6, classes=0; doc: Microlive runner for index ladders: one window, 1-lot maker quotes.
- `src/kalshi_alpha/exec/runners/orders_doctor.py` — funcs=3, classes=0; doc: Command-line helper for cleaning up outstanding DRY orders.
- `src/kalshi_alpha/exec/runners/pilot.py` — funcs=3, classes=0; doc: Single-entry CLI wrapper for pilot ladder sessions.
- `src/kalshi_alpha/exec/runners/pilot_close.py` — funcs=3, classes=0; doc: Entry point for maker-only close index pilot sessions.
- `src/kalshi_alpha/exec/runners/pilot_hourly.py` — funcs=3, classes=0; doc: Entry point for maker-only hourly index pilot sessions.
- `src/kalshi_alpha/exec/runners/risk_preview.py` — funcs=11, classes=0; doc: CLI to preview risk posture before running a ladder scan.
- `src/kalshi_alpha/exec/runners/scan_ladders.py` — funcs=76, classes=4; doc: CLI scanner that produces dry-run order proposals for Kalshi ladders.
- `src/kalshi_alpha/exec/scanners/__init__.py` — funcs=0, classes=0; doc: Utilities that reconcile ladder prices with strategy distributions.
- `src/kalshi_alpha/exec/scanners/cpi.py` — funcs=1, classes=0; doc: CPI ladder scanner utilities.
- `src/kalshi_alpha/exec/scanners/fast_index.py` — funcs=8, classes=1; doc: Fast offline index scan helpers used by --fast-fixtures.
- `src/kalshi_alpha/exec/scanners/index_scan_common.py` — funcs=11, classes=2; doc: Shared helpers for index ladder scanner CLIs.
- `src/kalshi_alpha/exec/scanners/scan_index_close.py` — funcs=3, classes=0; doc: Scanner helpers for daily close index ladders.
- `src/kalshi_alpha/exec/scanners/scan_index_hourly.py` — funcs=3, classes=2; doc: Scanner helpers and CLI for intraday hourly index ladders.
- `src/kalshi_alpha/exec/scanners/scan_index_noon.py` — funcs=1, classes=0; doc: Deprecated alias for hourly index ladder scanner CLI.
- `src/kalshi_alpha/exec/scanners/utils.py` — funcs=2, classes=0; doc: Utilities shared by ladder scanners.
- `src/kalshi_alpha/exec/scoreboard.py` — funcs=18, classes=0; doc: Generate rolling performance scoreboards.
- `src/kalshi_alpha/exec/scoreboard_index_paper.py` — funcs=9, classes=1; doc: Scoreboard for index paper (dry) ledger entries.
- `src/kalshi_alpha/exec/slo.py` — funcs=19, classes=1; doc: Service level objective (SLO) aggregations for scoreboard + telemetry exports.
- `src/kalshi_alpha/exec/state/orders.py` — funcs=1, classes=2; doc: Persistence utilities for tracking outstanding broker orders.
- `src/kalshi_alpha/exec/supervisor.py` — funcs=3, classes=2; doc: 24/7 supervisor daemon that orchestrates live index ladder scans.
- `src/kalshi_alpha/exec/supervisor_index.py` — funcs=14, classes=2; doc: Supervisor for index ladder windows with preflight and WS freshness gating.
- `src/kalshi_alpha/exec/telemetry/__init__.py` — funcs=0, classes=0; doc: Telemetry utilities for execution flows.
- `src/kalshi_alpha/exec/telemetry/shipper.py` — funcs=4, classes=0; doc: Utility to bundle telemetry JSONL into artifacts for shipping.
- `src/kalshi_alpha/exec/telemetry/sink.py` — funcs=8, classes=2; doc: Append-only telemetry sink for live execution events.
- `src/kalshi_alpha/exec/window_guard.py` — funcs=2, classes=0; doc: Shared helpers for gating index runners to ET trading windows.

## markets
- `src/kalshi_alpha/markets/__init__.py` — funcs=0, classes=0; doc: Market utilities for Kalshi index ladders.
- `src/kalshi_alpha/markets/discovery.py` — funcs=8, classes=2; doc: Market discovery utilities for INX/NDX ladders.

## models
- `src/kalshi_alpha/models/__init__.py` — funcs=0, classes=0; doc: Model primitives for kalshi-alpha.
- `src/kalshi_alpha/models/pmf_index.py` — funcs=7, classes=3; doc: Index PMF utilities with σ_tod curves and optional EOD variance bumps.

## replay
- `src/kalshi_alpha/replay/fill_model.py` — funcs=5, classes=0; doc: Convert TOB snapshots into conservative fill probability curves.
- `src/kalshi_alpha/replay/polygon_index_replay.py` — funcs=12, classes=2; doc: Replay recorded Polygon websocket aggregates into the index pipeline artifacts.

## risk
- `src/kalshi_alpha/risk/__init__.py` — funcs=0, classes=0; doc: Risk helpers for kalshi-alpha.
- `src/kalshi_alpha/risk/correlation.py` — funcs=2, classes=4; doc: Correlation-aware VaR limiter with inventory tilt support for index ladders.
- `src/kalshi_alpha/risk/var_index.py` — funcs=1, classes=1; doc: Simple per-family VaR limiter for index ladders.

## sched
- `src/kalshi_alpha/sched/__init__.py` — funcs=0, classes=0; doc: Scheduler helpers for hourly/EOD ladders plus regime-aware flags.
- `src/kalshi_alpha/sched/hotrestart.py` — funcs=2, classes=2; doc: Hot-restart snapshot utilities for maker ops.
- `src/kalshi_alpha/sched/regimes.py` — funcs=2, classes=1; doc: Trading-day regime flags for macro events (FOMC, CPI) with SLO overrides.
- `src/kalshi_alpha/sched/windows.py` — funcs=8, classes=1; doc: US/Eastern-aware scheduler for hourly and close index ladder windows.

## strategies
- `src/kalshi_alpha/strategies/__init__.py` — funcs=0, classes=0; doc: Strategy modules producing ladder probability distributions.
- `src/kalshi_alpha/strategies/base.py` — funcs=6, classes=0; doc: Shared helpers for strategy distributions.
- `src/kalshi_alpha/strategies/claims/__init__.py` — funcs=17, classes=2; doc: Initial jobless claims strategy with calibration utilities.
- `src/kalshi_alpha/strategies/cpi/__init__.py` — funcs=15, classes=2; doc: CPI strategy producing monthly and YoY distributions.
- `src/kalshi_alpha/strategies/cpi/components.py` — funcs=6, classes=1; doc: Component signals used in CPI v1.5 nowcasting.
- `src/kalshi_alpha/strategies/gas/__init__.py` — funcs=0, classes=0; doc: Placeholder for future gasoline ladder strategies.
- `src/kalshi_alpha/strategies/index/__init__.py` — funcs=0, classes=0; doc: Index ladder strategies powered by Polygon data.
- `src/kalshi_alpha/strategies/index/backtest_polygon.py` — funcs=10, classes=2; doc: Minimal Polygon-only backtest harness for index ladders with Kalshi quotes.
- `src/kalshi_alpha/strategies/index/cdf.py` — funcs=16, classes=3; doc: CDF helpers shared by index strategies.
- `src/kalshi_alpha/strategies/index/close_range.py` — funcs=5, classes=1; doc: End-of-day close range strategy for index ladders.
- `src/kalshi_alpha/strategies/index/fill_model.py` — funcs=1, classes=0; doc: Lightweight maker fill probability heuristic for index ladders.
- `src/kalshi_alpha/strategies/index/hourly_above_below.py` — funcs=6, classes=1; doc: Intraday hourly above/below strategy for index ladders.
- `src/kalshi_alpha/strategies/index/model_polygon.py` — funcs=8, classes=1; doc: Simple Polygon-only distribution model for SPX/NDX ladders.
- `src/kalshi_alpha/strategies/index/noon_above_below.py` — funcs=0, classes=0; doc: Backward-compatible shim for hourly index above/below strategy.
- `src/kalshi_alpha/strategies/teny/__init__.py` — funcs=13, classes=1; doc: 10-year Treasury yield strategy with factor calibration.
- `src/kalshi_alpha/strategies/weather/__init__.py` — funcs=5, classes=2; doc: Weather strategy stubs enforcing NOAA/NWS DCR settlement requirements.

## structures
- `src/kalshi_alpha/structures/__init__.py` — funcs=0, classes=0; doc: Structure-level utilities (allocators, range builders, hedges).
- `src/kalshi_alpha/structures/allocator.py` — funcs=4, classes=8; doc: Capital allocator for INX/NDX structures with correlation-aware VaR guardrails.
- `src/kalshi_alpha/structures/range_ab.py` — funcs=2, classes=2; doc: Construct hedged Range↔AB structures from adjacent strikes.

## tools
- `tools/__init__.py` — funcs=0, classes=0; doc: Utility scripts for kalshi-sys tooling.
- `tools/build_fillcalib_dataset.py` — funcs=0, classes=0
- `tools/failover_smoke.py` — funcs=4, classes=0; doc: CLI smoke test for DualFeedFailover (synthetic timeline).
- `tools/project_state_build.py` — funcs=16, classes=1; doc: Generate project_state/_generated artifacts (inventory, symbols, imports, make targets).
- `tools/replay.py` — funcs=13, classes=1; doc: Replay recorded Kalshi sessions to validate EV parity.
- `tools/settlement_basis_audit.py` — funcs=21, classes=2; doc: Settlement basis audit for index ladder windows (Polygon vs Kalshi expiration value).
- `tools/verify_gpt_bundle.py` — funcs=8, classes=1; doc: Verify GPT bundle completeness and diff hygiene.

## utils
- `src/kalshi_alpha/utils/env.py` — funcs=1, classes=0; doc: Environment loading utilities.
- `src/kalshi_alpha/utils/family.py` — funcs=3, classes=0; doc: Family helpers for focusing execution on index ladders.
- `src/kalshi_alpha/utils/http.py` — funcs=1, classes=1; doc: HTTP utilities with caching support.
- `src/kalshi_alpha/utils/keys.py` — funcs=5, classes=0; doc: Secure secret loaders with macOS Keychain support.
- `src/kalshi_alpha/utils/secrets.py` — funcs=3, classes=0; doc: Utilities for detecting and redacting sensitive strings.
