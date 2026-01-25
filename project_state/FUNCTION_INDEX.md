# Function Index

## Metadata
- Generated: 2026-01-10T11:43:04Z
- Git SHA: 31316e59451269689f2da173d8a9c6d9049d3d5e
- Branch: codex/TICKET-111_project_state_refresh
- Commands: `python tools/project_state_build.py`, `python3 tools/agentic/project_state_refresh.py --zip`, `rg --files`, `sed -n '1,200p' README.md`, `sed -n '1,200p' docs/PROGRESS.md`, `sed -n '1,200p' CHANGELOG.md`, `sed -n '1,200p' pyproject.toml`, `sed -n '1,200p' Makefile`

## Legend
- Paths are repo-relative.
- Signatures are best-effort from AST parsing.
- Docstrings are first-line only.
- Source of truth: `project_state/_generated/symbol_index.json`.

## src/kalshi_alpha/__init__.py
Module doc: Kalshi Alpha namespace package.
No top-level functions or classes.

## src/kalshi_alpha/backtest/__init__.py
Module doc: Backtesting utilities for index ladder strategies.
No top-level functions or classes.

## src/kalshi_alpha/backtest/generate_dataset.py
Module doc: Build minute-level backtest dataset for index ladders.
Classes:
- `DatasetRow()`
Functions:
- `parse_args(argv=None) -> argparse.Namespace`
- `build_dataset(*, start, end, symbols, client=None) -> pl.DataFrame`
- `_fetch_day_bars(client, symbol, trading_day) -> list[MinuteBar]`
- `_rows_for_symbol(trading_day, symbol, bars, targets) -> list[DatasetRow]`
- `_in_session(timestamp_utc, trading_day) -> bool`
- `_ewma_sigma_series(bars, *, span, min_samples) -> list[float]`
- `_micro_drift_series(bars, *, window) -> list[float]`
- `_on_before_map(bars, targets, symbol) -> dict[tuple[str, datetime], float | None]`
- `main(argv=None) -> None`

## src/kalshi_alpha/backtest/index_calendar.py
Module doc: Trading calendar helpers for index backtests.
Classes:
- `TargetType(str, Enum)` — Label for target horizons.
- `TargetSpec()` — Target timestamp metadata for a given trading day.
Functions:
- `trading_days(start, end) -> list[date]` — Return all NYSE trading days in the inclusive [start, end] window.
- `enumerate_targets(start, end, *, target_hours=None) -> Iterator[TargetSpec]` — Yield target specifications between the provided bounds.
- `targets_for_day(trading_day, *, target_hours=None) -> list[TargetSpec]` — Resolve all backtest targets for the provided trading day.
- `is_trading_day(day) -> bool` — Return whether the supplied date is a regular NYSE trading session.
- `_us_equity_holidays(year) -> set[date]` — Approximate NYSE holiday calendar for the provided year.
- `_nth_weekday(year, month, weekday, n) -> date`
- `_last_weekday(year, month, weekday) -> date`
- `_easter_date(year) -> date` — Compute Easter Sunday using Anonymous Gregorian algorithm.

## src/kalshi_alpha/backtest/score_close.py
Module doc: Evaluate close index calibrations against historical dataset.
Functions:
- `parse_args(argv=None) -> argparse.Namespace`
- `main(argv=None) -> None`

## src/kalshi_alpha/backtest/score_hourly.py
Module doc: Evaluate hourly index calibrations against historical dataset.
Functions:
- `parse_args(argv=None) -> argparse.Namespace`
- `main(argv=None) -> None`

## src/kalshi_alpha/backtest/scoring.py
Module doc: Shared scoring utilities for index backtests.
Classes:
- `ScoreSample()`
- `ScoreReport()`
Functions:
- `evaluate_backtest(*, dataset_path=DEFAULT_DATASET_PATH, output_dir, horizon, polygon_to_series, contracts=DEFAULT_CONTRACTS, calibration_loader=None) -> ScoreReport`
- `_load_dataset(path) -> pl.DataFrame`
- `_select_terminal_rows(frame, horizon) -> list[dict[str, object]]`
- `_default_calibration_loader(symbol, horizon) -> index_cdf.SigmaCalibration`
- `_resolve_mean_std(calibration, *, horizon, minutes, current_price) -> tuple[float, float]`
- `_normal_cdf(value, mean, std) -> float`
- `_normal_pdf(value, mean, std) -> float`
- `_normal_crps(value, mean, std) -> float`
- `_index_taker_fee(contracts, price) -> float`
- `_summaries(samples) -> dict[str, dict[str, float]]`
- `_pit_histogram(samples, *, bins) -> list[dict[str, float]]`
- `_write_outputs(report, output_dir) -> None`

## src/kalshi_alpha/brokers/__init__.py
Module doc: Factory helpers for broker adapters.
Functions:
- `create_broker(mode, *, artifacts_dir, audit_dir, acknowledge_risks=False, live_kwargs=None) -> Broker`

## src/kalshi_alpha/brokers/kalshi/base.py
Module doc: Shared interfaces for Kalshi broker adapters.
Classes:
- `BrokerOrder()` — Canonical representation of an order submission.
- `Broker(Protocol)` — Protocol describing the expected broker adapter surface area.
Functions:
- `ensure_directory(path) -> Path`

## src/kalshi_alpha/brokers/kalshi/dry.py
Module doc: Dry-run broker adapter that records intended Kalshi orders.
Classes:
- `_SerializedOrder()`
- `DryBroker(Broker)` — Broker implementation that serializes orders without hitting the network.

## src/kalshi_alpha/brokers/kalshi/endpoints.py
Module doc: Kalshi environment endpoint helpers.
Classes:
- `KalshiEndpoints()`
Functions:
- `resolve(env) -> KalshiEndpoints`

## src/kalshi_alpha/brokers/kalshi/http_client.py
Module doc: Shared Kalshi HTTP client implementing RSA-PSS auth, retries, and structured logging.
Classes:
- `Clock()` — Small helper to allow deterministic clocks in tests.
- `Sleep()` — Wrapper so tests can bypass actual sleeping.
- `KalshiHttpError(RuntimeError)` — Raised when the Kalshi API request exhausts retries.
- `KalshiClockSkewError(RuntimeError)` — Raised when the local clock drifts too far from UTC.
- `KalshiHttpClient()` — Resilient Kalshi HTTP client with header-based RSA-PSS signing.
Functions:
- `_mask(value) -> str`
- `_ensure_utc(dt) -> datetime`

## src/kalshi_alpha/brokers/kalshi/live.py
Module doc: Live Kalshi broker adapter with rate limiting, backoff, and auditing.
Classes:
- `_RateLimiter()` — Token bucket rate limiter used to throttle Trading API calls.
- `LiveBroker(Broker)` — Networked broker adapter for the Kalshi trading API.
Functions:
- `_safe_float(value) -> float | None`
- `_price_from_entry(entry) -> float | None`
- `_best_price(levels, *, side) -> float | None`
- `_extract_best_prices(snapshot) -> tuple[float | None, float | None]`
- `_parse_timestamp(value) -> datetime | None`
- `_extract_snapshot_timestamp(metadata, snapshot) -> datetime | None`
- `_validate_live_environment() -> None`

## src/kalshi_alpha/brokers/kalshi/ws_client.py
Module doc: Authenticated Kalshi WebSocket client with reconnect/backoff.
Classes:
- `KalshiWebsocketError(RuntimeError)` — Raised when the Kalshi websocket client exhausts reconnect attempts.
- `KalshiWebsocketClient()` — Small helper wrapping an authenticated Kalshi websocket connection.
Functions:
- `_utc_now() -> datetime`
- `_safe_float(value) -> float | None`
- `_safe_int(value) -> int | None`

## src/kalshi_alpha/config/__init__.py
Module doc: Configuration loaders for strategy- and scanner-level metadata.
No top-level functions or classes.

## src/kalshi_alpha/config/index_ops.py
Module doc: Load shared operational window configuration for index ladder strategies.
Classes:
- `IndexOpsWindow()` — Operations window definition with cancel buffers and optional offsets.
- `IndexOpsConfig()` — Aggregated operations configuration for index ladder scanners and microlive.
Functions:
- `load_index_ops_config(path=None) -> IndexOpsConfig` — Read the index operations configuration from disk.
- `_load_index_ops_config_cached(resolved_path) -> IndexOpsConfig`
- `_parse_window(payload, *, label) -> IndexOpsWindow`
- `_parse_time(value) -> time`

## src/kalshi_alpha/config/index_rules.py
Module doc: Load index ladder rule semantics from the markdown rulebook.
Classes:
- `IndexRule()` — Structured view of a Kalshi index ladder rule entry.
- `IndexRuleBook()` — Container for all series rule entries plus shared metadata.
Functions:
- `load_index_rulebook(path=None) -> IndexRuleBook` — Parse the markdown rule summary and return a structured rulebook.
- `_load_index_rulebook_cached(resolved_path) -> IndexRuleBook`
- `lookup_index_rule(series, *, path=None) -> IndexRule` — Return the rule metadata for a specific Kalshi series.
- `_extract_front_matter(path) -> dict[str, Any]`

## src/kalshi_alpha/config/size_ladder.py
Module doc: Size ladder configuration for staged lot/bin limits.
Classes:
- `SeriesLimits()`
- `SizeStage()`
- `SizeLadderConfig()`
Functions:
- `load_size_ladder(path=None) -> SizeLadderConfig`

## src/kalshi_alpha/core/__init__.py
Module doc: Core primitives shared across Kalshi Alpha components.
No top-level functions or classes.

## src/kalshi_alpha/core/archive/__init__.py
Module doc: Archiving and replay utilities for Kalshi public market data.
No top-level functions or classes.

## src/kalshi_alpha/core/archive/archiver.py
Module doc: Archive Kalshi public data snapshots for later replay.
Functions:
- `archive_scan(series, client, events, markets, orderbooks, out_dir=None) -> Path` — Persist a snapshot of series/events/markets/orderbooks for later replay.
- `_dump_json(obj) -> str`
- `_to_jsonable(obj) -> object`

## src/kalshi_alpha/core/archive/replay.py
Module doc: Replay archived Kalshi data to recompute proposal EVs.
Classes:
- `_ReplayContext()`
Functions:
- `replay_manifest(manifest_path, *, model_version='v15', orderbooks_override=None) -> Path` — Recompute expected values for archived proposals and write parquet output.
- `_load_context(manifest_path) -> _ReplayContext`
- `_load_markets(ctx) -> dict[str, Market]`
- `_load_orderbooks(ctx) -> dict[str, Orderbook]`
- `_load_proposals(ctx) -> list[dict[str, Any]]`
- `_proposal_liquidity(proposal) -> str`
- `_resolve_driver_fixtures(ctx) -> Path`
- `_strategy_pmf_for_series(*, series, strikes, fixtures_dir, override, offline, model_version='v15', orderbook_imbalance=None, event_timestamp=None) -> list[LadderBinProbability]`
- `_load_history(fixtures_dir, namespace) -> list[dict[str, Any]]`
- `_macro_calendar_lookup(history, *, fixtures_dir) -> dict[str, dict[str, float]]`
- `_normalize_macro_date(value) -> str | None`
- `_strip_dummy_prefix(column) -> str`
- `_build_rungs(market) -> list[LadderRung]`
- `_resolve_rung_index(strikes, strike) -> int | None`
- `_write_output(records, artifacts_dir) -> Path`

## src/kalshi_alpha/core/archive/scorecards.py
Module doc: Replay scorecard computation from archived manifests and proposals.
Classes:
- `ReplayScorecard()` — Container for replay scorecard metrics.
Functions:
- `build_replay_scorecard(*, manifest_path, model_version='v15', driver_fixtures=None) -> ReplayScorecard` — Compute replay scorecard metrics for archived manifest and proposals.
- `_compute_deltas(strategy_survival, market_survival) -> list[float]`
- `_orderbook_depth(orderbook) -> int`
- `_empty_summary_df(model_version) -> pl.DataFrame`
- `_empty_delta_df() -> pl.DataFrame`

## src/kalshi_alpha/core/backtest/__init__.py
Module doc: Backtesting utilities: scoring rules, event replay, and calibration diagnostics.
Classes:
- `Snapshot()`
- `EventReplayer()` — Simple chronological iterator over event snapshots.
Functions:
- `brier_score(probabilities, outcomes) -> float`
- `log_loss(probabilities, outcomes, *, epsilon=EPS) -> float`
- `crps_from_pmf(pmf, observation) -> float` — Continuous Ranked Probability Score for discrete ladder PMFs.
- `_bin_contains(bin_prob, observation) -> bool`
- `_bin_width(bin_prob) -> float`
- `reliability_table(probabilities, outcomes, *, buckets=10) -> list[dict]` — Bucketed calibration table (reliability curve).
- `probability_integral_transform(pmf, observation) -> float` — Discrete PIT using ladder PMFs.

## src/kalshi_alpha/core/datastore/__init__.py
Module doc: Datastore helpers for raw snapshots, processed tables, and DuckDB cataloging.
Classes:
- `SnapshotWriter()`
- `ProcessedWriter()`
- `DuckDBCatalog()` — Lightweight registry of processed snapshots inside DuckDB.
Functions:
- `_timestamp_slug(ts=None) -> str`
- `_next_available(path) -> Path`

## src/kalshi_alpha/core/execution/defaults.py
Module doc: Execution defaults for index ladder maker behaviour (alpha & slippage).
Functions:
- `_resolve_path(path=None) -> Path`
- `_load_defaults(resolved_path) -> dict[str, Any]`
- `default_alpha(series, *, path=None) -> float | None`
- `slippage_config(series, *, path=None) -> dict[str, Any] | None`

## src/kalshi_alpha/core/execution/fillprob.py
Module doc: Load conservative fill probability curves derived from TOB snapshots.
Classes:
- `FillCurveStatus()`
Functions:
- `resolve_curve_path(path=None) -> Path | None`
- `_load_payload(path_str) -> tuple[Mapping[str, object] | None, str | None]`
- `curve_status(path=None) -> FillCurveStatus`
- `probability(series, *, seconds_to_event=None, window_id=None, side=None, quote_distance_to_touch_bin=None, time_to_expiry_bin=None, path=None) -> float`
- `adjust_alpha(series, base_alpha, *, seconds_to_event=None, window_id=None, side=None, quote_distance_to_touch_bin=None, time_to_expiry_bin=None, path=None) -> float`
- `_latest_curve_path(root) -> Path | None`
- `_parse_asof_date(name) -> date | None`
- `_safe_float(value) -> float | None`
- `_safe_int(value) -> int | None`
- `_safe_str(value) -> str | None`
- `_clamp_probability(value) -> float`

## src/kalshi_alpha/core/execution/fillratio.py
Module doc: Estimate expected fills using simple visible-depth heuristics.
Classes:
- `FillRatioEstimator()`
Functions:
- `alpha_row(depth, size, alpha_base) -> float` — Row-level fill alpha adjusted by visible depth.
- `expected_fills(size, visible_depth, alpha) -> tuple[int, float]` — Return expected filled contracts and fill ratio for requested size.
- `tune_alpha(series, archives_dir=None, *, lookback_days=14, min_observations=30, persist=True) -> float | None` — Estimate a fill-allocation alpha from aggregated ledger outcomes.
- `load_alpha(series) -> float | None` — Return the persisted alpha value for the given series family.
- `_visible_depth(side, price, orderbook) -> float`
- `_load_ledger_frame() -> pl.DataFrame`
- `_persist_alpha(series, alpha, sample_size=None) -> None`

## src/kalshi_alpha/core/execution/index_models.py
Module doc: Execution curve loaders for index fills and slippage.
Classes:
- `AlphaCurve()`
- `SlippageCurve()`
Functions:
- `_curve_path(series, name) -> Path`
- `load_alpha_curve(series) -> AlphaCurve | None`
- `load_slippage_curve(series) -> SlippageCurve | None`
- `slippage_ticks_to_price(ticks) -> float`

## src/kalshi_alpha/core/execution/order_queue.py
Module doc: FIFO order queue for coordinating cancel/replace operations.
Classes:
- `_QueueItem()`
- `OrderQueue()` — Synchronous queue that retries cancel/replace operations with backoff.

## src/kalshi_alpha/core/execution/series_utils.py
Module doc: Helpers for canonicalizing execution series names.
Functions:
- `canonical_series_family(label) -> str` — Return the canonical family identifier for a series label.

## src/kalshi_alpha/core/execution/slippage.py
Module doc: Slippage modelling utilities for paper execution.
Classes:
- `SlippageModel()` — Represents a piecewise linear slippage curve.
- `SlippageCalibration()` — Persisted calibration describing a fitted depth-impact curve.
Functions:
- `price_with_slippage(*, side, contracts, proposal_price, orderbook, model) -> tuple[float, float]` — Return adjusted fill price and slippage (signed) given an orderbook.
- `_top_of_book(side, proposal_price, orderbook) -> float`
- `_levels_for_side(side, orderbook) -> Sequence[dict[str, float]]`
- `fit_slippage(series, *, lookback_days=14, min_observations=30, persist=True) -> SlippageCalibration | None` — Fit a simple depth-impact curve from realized ledger slippage.
- `load_slippage_calibration(series) -> SlippageCalibration | None` — Load a persisted slippage calibration for the given series family.
- `load_slippage_model(series, *, mode='depth') -> SlippageModel | None` — Instantiate a `SlippageModel` from persisted calibration parameters.
- `_load_ledger_frame() -> pl.DataFrame`
- `_persist_slippage(calibration) -> None`
- `_load_calibration_entry(family) -> dict[str, object] | None`
- `_quantile(frame, quantile, *, fallback) -> float`
- `_ensure_monotonic(curve) -> tuple[tuple[float, float], ...]`
- `_parse_timestamp(value) -> datetime`

## src/kalshi_alpha/core/fees/__init__.py
Module doc: Kalshi fee schedule utilities loaded from the canonical JSON configuration.
Classes:
- `FeeSchedule()` — Encapsulates Kalshi maker/taker fee logic with JSON-backed configuration.
Functions:
- `_load_fee_config(config_path=None) -> dict[str, Any]`
- `_to_decimal(value) -> Decimal`
- `_contains_keyword(name, keywords) -> bool`
- `round_up_to_cent(amount) -> Decimal` — Round *amount* up to the next cent, matching Kalshi settlement semantics.
- `_is_index_series(series_key) -> bool`
- `_ensure_index_maker_fee_guard() -> None`
- `fee_index_taker(*, price, contracts) -> Decimal` — Return the taker fee for an index ladder, rounded up to the nearest cent.
- `taker_fee(contracts, price, *, series=None, market_name=None, half_rate=None, schedule=DEFAULT_FEE_SCHEDULE) -> Decimal` — Convenience wrapper using the default fee schedule.
- `maker_fee(contracts, price, *, series=None, market_name=None, half_rate=None, schedule=DEFAULT_FEE_SCHEDULE) -> Decimal` — Convenience wrapper using the default fee schedule.

## src/kalshi_alpha/core/fees/index_series.py
Module doc: Index series fee curve loader.
Classes:
- `IndexFeeCurve()` — Parametric fee curve for a specific index ladder series.
Functions:
- `load_index_fee_curves(path=None) -> Mapping[str, IndexFeeCurve]` — Load all index fee curves from the reference JSON file.
- `get_index_fee_curve(series, path=None) -> IndexFeeCurve | None` — Return the fee curve for *series*, or ``None`` if not configured.
- `_load_index_fee_curves_cached(resolved_path) -> dict[str, IndexFeeCurve]`
- `_load_with_extends(path, *, _visited=None) -> dict[str, object]`

## src/kalshi_alpha/core/gates/__init__.py
Module doc: Quality gate evaluation utilities.
No top-level functions or classes.

## src/kalshi_alpha/core/gates/quality_gates.py
Module doc: Production quality gates for daily orchestration.
Classes:
- `MetricThreshold()`
- `DataFreshnessThreshold()`
- `ReconciliationThreshold()`
- `QualityGateConfig()`
- `QualityGateResult()`
- `_QualityGateEvaluator()`
Functions:
- `load_quality_gate_config(path=None) -> QualityGateConfig` — Load quality gate thresholds from YAML.
- `run_quality_gates(*, config=None, config_path=None, monitors=None, scope=None, now=None, proc_root=None, raw_root=None) -> QualityGateResult` — Evaluate all gates and return go/no-go verdict.
- `_maybe_float(value, *, default=None) -> float | None`
- `_filter_quality_gate_config(config, scope) -> QualityGateConfig`
- `_normalize_scope(scope) -> str | None`
- `_namespace_prefix(namespace) -> str`
- `_latest_parquet(directory) -> Path | None`

## src/kalshi_alpha/core/kalshi_api/__init__.py
Module doc: Read-only Kalshi public market-data client.
Classes:
- `Series()`
- `Event()`
- `Market()`
- `Orderbook()`
- `KalshiPublicClient()` — Simple public API client with offline fixture support.
Functions:
- `_matches_filter(market, series_ticker, status, event_ticker) -> bool`
- `_parse_timestamp(value) -> datetime | None`

## src/kalshi_alpha/core/kalshi_ws.py
Module doc: Utilities for consuming Kalshi orderbook websockets and deriving imbalance metrics.
Classes:
- `OrderbookLevel()`
- `OrderbookImbalanceTracker()` — Track rolling orderbook imbalance for a single market.
- `OrderbookSnapshotWriter()` — Append orderbook snapshots to newline-delimited JSON files.
Functions:
- `compute_imbalance(bids, asks, *, depth=3) -> float` — Return the normalized bid/ask imbalance for the top-of-book.
- `persist_orderbook_snapshot(ticker, snapshot, *, timestamp=None) -> Path`
- `persist_imbalance_metric(ticker, imbalance, *, timestamp=None) -> Path`
- `load_latest_imbalance(ticker) -> tuple[float, datetime] | None`
- `stream_orderbook_imbalance(tickers, *, depth=3, window_seconds=30, ws_url=WS_ENDPOINT, client=None, run_seconds=None, now_fn=None, writer_root=None, reader_timeout=5.0, auth_token=None) -> dict[str, float]` — Consume orderbook deltas for ``tickers`` and persist rolling imbalance metrics.
- `replay_snapshots(paths, *, depth=3, window_seconds=30) -> dict[str, float]` — Utility to recompute imbalance metrics from existing snapshots.
- `_extract_ticker(payload) -> str`
- `_extract_snapshot(payload) -> dict[str, Any]`

## src/kalshi_alpha/core/pricing/__init__.py
Module doc: Ladder pricing utilities: survival curve projection, PMF extraction, and EV analysis.
Classes:
- `Liquidity(Enum)` — Maker/taker liquidity flag.
- `OrderSide(Enum)` — YES or NO order direction.
- `LadderRung()` — Represents a single ladder strike with YES price quoted in probability space.
- `LadderBinProbability()` — Probability mass associated with a bin.
Functions:
- `_index_fee(contracts, price, series_key) -> float`
- `_clamp_probability(value) -> float`
- `project_survival(values) -> list[float]` — Project values onto the monotone non-increasing cone via PAV.
- `survival_from_quotes(rungs, *, enforce_monotonic=True) -> list[float]` — Extract the implied survival curve from ladder quotes.
- `pmf_from_survival(strikes, survival, *, tolerance=1e-08) -> list[LadderBinProbability]` — Convert a survival curve into discrete bin probabilities.
- `pmf_from_quotes(rungs, *, tolerance=1e-08) -> list[LadderBinProbability]` — Convenience wrapper that goes from ladder quotes → survival → PMF.
- `has_probability_arbitrage(pmf, *, tolerance=1e-06) -> bool` — Return True when probabilities violate basic add-up constraints.
- `_validate_inputs(probability, yes_price, *, tolerance=1e-09) -> tuple[float, float]`
- `expected_value_after_fees(*, contracts, yes_price, event_probability, side, liquidity, schedule=DEFAULT_FEE_SCHEDULE, series=None, market_name=None) -> float` — Compute the expected value of a trade after trading fees.
- `yes_no_expected_values(*, contracts, yes_price, event_probability, liquidity, schedule=DEFAULT_FEE_SCHEDULE, series=None, market_name=None) -> dict[OrderSide, float]` — Return both YES and NO EVs after fees.
- `project_simplex(values) -> list[float]` — Project raw scores onto the probability simplex.

## src/kalshi_alpha/core/pricing/align.py
Module doc: Utilities for aligning model PMFs to ladder strike grids.
Classes:
- `SkipScan(Exception)` — Indicates the scan should be skipped and surfaced as a NO-GO.
Functions:
- `cdf_from_pmf(grid_breaks, pmf) -> Callable[[float], float]` — Build a piecewise-linear CDF from arbitrary ladder bin probabilities.
- `pmf_on_strikes(cdf_fn, strikes) -> list[float]` — Project a CDF onto ladder strikes, returning simplex-projected probabilities.
- `align_pmf_to_strikes(pmf, strikes) -> list[LadderBinProbability]` — Align an arbitrary PMF to the ladder grid defined by strikes.
- `_ladder_bins(strikes) -> list[tuple[float | None, float | None]]`
- `_cdf_at(x, pmf) -> float`
- `_clamp(value) -> float`

## src/kalshi_alpha/core/pricing/mispricing.py
Module doc: Ladder mispricing analytics and spread detection.
Classes:
- `KinkMetrics()`
Functions:
- `implied_cdf_kinks(survival) -> KinkMetrics` — Quantify curvature and monotonicity violations for an implied survival curve.
- `prob_sum_gap(pmf) -> float` — Return the absolute probability mass deficit/excess for a PMF.
- `kink_spreads(pmf, market_pmf, *, max_legs=4, min_abs_delta=1e-06) -> list[dict[str, object]]` — Identify adjacent-bin spread candidates based on probability deltas.

## src/kalshi_alpha/core/risk/__init__.py
Module doc: Risk utilities including PAL (Position and Loss) policy enforcement.
Classes:
- `PALPolicy()` — Position and loss guardrails for a single Kalshi series.
- `OrderProposal()` — Dry-run order proposal evaluated by the PAL guard.
- `PALGuard()` — Tracks rolling exposure versus PAL limits.
- `PortfolioConfig()`
- `PortfolioRiskManager()`
Functions:
- `max_loss_for_order(order, *, schedule=DEFAULT_FEE_SCHEDULE) -> float` — Return the maximum loss in USD for the order, including trading fees.

## src/kalshi_alpha/core/risk/drawdown.py
Module doc: Persistent drawdown guard for paper PnL caps.
Classes:
- `DrawdownStatus()`
Functions:
- `record_pnl(pnl, *, timestamp=None, state_dir=None) -> None` — Persist paper ledger PnL into daily/weekly aggregates.
- `check_limits(daily_cap, weekly_cap, *, now=None, state_dir=None) -> DrawdownStatus` — Evaluate whether drawdown caps are respected.
- `_state_directory(override=None) -> Path`
- `_state_path(override=None) -> Path`
- `_load_state(override=None) -> dict[str, Any]`
- `_save_state(state, override=None) -> None`

## src/kalshi_alpha/core/sizing/__init__.py
Module doc: Sizing utilities for portfolio-aware ladder execution.
No top-level functions or classes.

## src/kalshi_alpha/core/sizing/kelly.py
Module doc: Kelly-based sizing helpers with safety caps.
Functions:
- `kelly_yes_no(p_true, p_mkt) -> float` — Return the raw Kelly fraction for taking the YES side.
- `truncate_kelly(kelly_fraction, cap) -> float` — Truncate a Kelly fraction to a symmetric +/- cap.
- `scale_kelly(kelly_fraction, uncertainty, ob_imbalance, cap) -> float` — Scale a Kelly fraction by uncertainty and orderbook imbalance penalties.
- `apply_caps(size, pal, max_loss_per_strike, max_var) -> float` — Apply PAL, per-strike, and VaR caps to a proposed risk size.

## src/kalshi_alpha/core/ws.py
Module doc: Lightweight Kalshi websocket client with RSA-PSS authentication.
Classes:
- `KalshiWebsocketClient()` — Minimal Kalshi websocket client that signs the handshake with RSA-PSS.

## src/kalshi_alpha/data/__init__.py
Module doc: Data-plane helpers (websocket sentries, etc.).
No top-level functions or classes.

## src/kalshi_alpha/data/failover.py
Module doc: SLO-enforced dual-feed failover controller.
Classes:
- `FeedSample(Generic[T])` — Normalized feed sample.
- `DualFeedFailover(Generic[T])` — Track two feeds (Polygon primary, Massive secondary) and decide which to trust.
Functions:
- `_ensure_utc(moment) -> datetime`

## src/kalshi_alpha/data/ws_sentry.py
Module doc: Lightweight websocket freshness sentry with strict (final-minute) gating.
Classes:
- `WSFreshnessSentry()` — Track websocket update latency and expose strict freshness checks.
Functions:
- `_ensure_utc(moment) -> datetime`

## src/kalshi_alpha/datastore/ingest.py
Module doc: CLI to ingest data snapshots across drivers.
Classes:
- `IngestContext()`
Functions:
- `ingest_bls_cpi(ctx) -> None`
- `ingest_dol_claims(ctx) -> None`
- `ingest_treasury(ctx) -> None`
- `ingest_cleveland(ctx) -> None`
- `ingest_nws(ctx) -> None`
- `ingest_aaa(ctx) -> None`
- `parse_args(argv) -> argparse.Namespace`
- `_resolve_sources(args) -> list[str]`
- `main(argv=None) -> None`

## src/kalshi_alpha/datastore/paths.py
Module doc: Shared datastore paths.
No top-level functions or classes.

## src/kalshi_alpha/datastore/snapshots.py
Module doc: Helpers for writing raw datastore snapshots.
Functions:
- `_timestamp() -> datetime`
- `_snapshot_dir(namespace, timestamp=None) -> Path`
- `_unique_path(path) -> Path` — Return a unique path by appending a numeric suffix if necessary.
- `write_json_snapshot(namespace, name, payload, *, timestamp=None) -> Path`
- `write_text_snapshot(namespace, name, content, *, timestamp=None) -> Path`

## src/kalshi_alpha/dev/__init__.py
Module doc: Developer utilities such as repository hygiene checks.
No top-level functions or classes.

## src/kalshi_alpha/dev/imbalance_snap.py
Module doc: Capture Kalshi orderbook imbalance metrics for the TENY close window.
Functions:
- `_parse_time(value) -> time`
- `_parse_date(value) -> date`
- `_build_parser() -> argparse.ArgumentParser`
- `_resolve_window(args, now_utc) -> tuple[datetime, datetime]`
- `_apply_root_overrides(args) -> None`
- `_capture(args) -> dict[str, float]`
- `main(argv=None) -> dict[str, float]`

## src/kalshi_alpha/dev/parse_fees.py
Module doc: Parse Kalshi fee schedule PDF and emit normalized JSON configuration.
Functions:
- `_normalize_text(pdf_bytes) -> str` — Decode PDF bytes into a best-effort string suitable for regex parsing.
- `_extract_rate(pattern, text, fallback) -> float`
- `_extract_list(pattern, text) -> list[str]`
- `_to_amount(value) -> int | None`
- `_parse_brackets(text) -> list[dict[str, Any]]`
- `parse_fee_schedule(pdf_path, *, base_config=None) -> dict[str, Any]`
- `main(argv=None) -> Path`

## src/kalshi_alpha/dev/sanity_check.py
Module doc: Repository hygiene check with optional live smoke test.
Functions:
- `main(argv=None) -> int`
- `_parse_args(argv) -> argparse.Namespace`
- `_run_repo_checks(root) -> int`
- `_run_live_smoke(env_override) -> int`
- `_safe_summary(payload, *, keys) -> list[str]`
- `_is_excluded(path, root) -> bool`

## src/kalshi_alpha/dev/ws_smoke.py
Module doc: Developer CLI to smoke-test Kalshi websocket imbalance streaming.
Functions:
- `_build_parser() -> argparse.ArgumentParser`
- `_normalize_tickers(raw) -> list[str]`
- `_latest_snapshot_path(ticker) -> Path | None`
- `_execute(args) -> dict[str, float]`
- `main(argv=None) -> dict[str, float]`

## src/kalshi_alpha/drivers/__init__.py
Module doc: Data drivers ingesting macro and settlement data sources.
No top-level functions or classes.

## src/kalshi_alpha/drivers/aaa_gas/__init__.py
Module doc: AAA gasoline price driver stub (fixture backed).
Classes:
- `GasPriceSummary()`
Functions:
- `load_summary(*, offline_path=None) -> GasPriceSummary`

## src/kalshi_alpha/drivers/aaa_gas/fetch.py
Module doc: AAA gasoline national average fetcher.
Classes:
- `AAANationalPrice()`
Functions:
- `fetch_latest(*, offline=False, fixtures_dir=None, force_refresh=False, session=None) -> AAANationalPrice`
- `_parse_price_html(html) -> tuple[date, float]`
- `_update_daily_parquet(as_of, price) -> None`
- `_update_monthly(frame) -> None`
- `mtd_average(reference=None) -> float | None`
- `main(argv=None) -> None`

## src/kalshi_alpha/drivers/aaa_gas/ingest.py
Module doc: AAA gasoline bootstrap ingestion.
Functions:
- `bootstrap_from_csv(path) -> dict[str, Path]` — Create daily and monthly Parquet files from historical AAA CSV.
- `_ensure_bootstrap_dir(csv_path) -> Path`
- `main(argv=None) -> None`

## src/kalshi_alpha/drivers/bls_cpi/__init__.py
Module doc: BLS CPI driver with online/offline support.
Classes:
- `CPIRelease()`
Functions:
- `fetch_release_calendar(*, offline=False, fixtures_dir=None, force_refresh=False, session=None) -> list[datetime]` — Return upcoming CPI release datetimes in ET.
- `fetch_latest_release(*, offline=False, fixtures_dir=None, force_refresh=False, session=None) -> CPIRelease` — Fetch the most recent CPI release from BLS.
- `_parse_calendar_html(html) -> dict[str, list[str]]`
- `_fetch_latest_release_online(*, force_refresh, session) -> dict[str, object]`

## src/kalshi_alpha/drivers/calendar/__init__.py
Module doc: Minimal event calendar helpers for index ladders.
No top-level functions or classes.

## src/kalshi_alpha/drivers/calendar/loader.py
Module doc: Load and query minimal macro event calendar metadata.
Classes:
- `EventCalendar()` — In-memory lookup helper for macro event tags keyed by ET date.
Functions:
- `_normalize_tags(tags) -> tuple[str, ...]`
- `_to_date(moment) -> date`
- `load_calendar(path=None) -> EventCalendar`
- `_load_calendar_cached(path_str) -> EventCalendar`
- `calendar_tags_for(moment, *, path=None) -> tuple[str, ...]`

## src/kalshi_alpha/drivers/cleveland_nowcast/__init__.py
Module doc: Cleveland Fed inflation nowcast driver.
Classes:
- `NowcastSeries()`
Functions:
- `fetch_nowcast(*, offline=False, fixtures_dir=None, force_refresh=False, session=None) -> dict[str, NowcastSeries]` — Return headline/core nowcasts.
- `_to_series(payload) -> NowcastSeries`
- `_parse_monthly_nowcast(json_text) -> tuple[dict[str, object], dict[str, object]]` — Parse the Cleveland Fed chart JSON into headline and core payloads.
- `_select_latest_entry(payload) -> dict[str, object]`
- `_extract_series(dataset, target, as_of) -> tuple[float, datetime]`
- `_parse_tooltext_date(tooltext, fallback) -> datetime`
- `_extract_monthly_json_url(page_text) -> str | None`

## src/kalshi_alpha/drivers/dol_claims/__init__.py
Module doc: DOL ETA-539 weekly claims driver.
Classes:
- `ClaimsReport()`
Functions:
- `fetch_latest_report(*, offline=False, fixtures_dir=None, force_refresh=False, session=None) -> ClaimsReport` — Fetch the latest ETA-539 report.
- `_parse_eta_539_csv(csv_text) -> ClaimsReport`
- `latest_claims_dataframe(report) -> pl.DataFrame` — Return a Polars dataframe for downstream use.

## src/kalshi_alpha/drivers/index_polygon.py
Module doc: Polygon index history loader for offline modelling/backtests.
Functions:
- `_sanitize_symbol(symbol) -> str` — Convert a Polygon ticker into a filesystem-friendly slug.
- `_normalize_symbol_label(symbol) -> str` — Return a colon-delimited Polygon ticker for downstream use.
- `_parse_date(value) -> date | None`
- `_resolve_symbol_dir(symbol, base_root=None) -> Path` — Pick the first existing directory for the provided symbol.
- `_iter_symbol_files(symbol_dir, *, start_date, end_date) -> list[Path]`
- `load_symbol_minutes(symbol, *, start_date=None, end_date=None, base_root=None, as_pandas=False) -> pl.DataFrame` — Load minute-level Polygon aggregates for a single index symbol.
- `load_minutes(symbols=None, *, start_date=None, end_date=None, base_root=None, as_pandas=False) -> pl.DataFrame` — Load and concatenate Polygon minute bars for multiple symbols.

## src/kalshi_alpha/drivers/kalshi_index_history.py
Module doc: Load historical Kalshi index ladder quotes from disk for offline backtests.
Classes:
- `QuoteSnapshot()` — Latest ladder snapshot for a given day/horizon.
Functions:
- `_candidate_paths(series, trading_day, root) -> Iterable[Path]`
- `_load_frame(path) -> pl.DataFrame`
- `_ensure_et_timestamp(value) -> datetime`
- `_normalize_columns(frame, series, trading_day) -> pl.DataFrame`
- `load_quotes_for_day(series, trading_day, *, root=None) -> pl.DataFrame` — Load all quotes for a trading day from disk, normalizing columns.
- `latest_snapshot(series, trading_day, horizon, *, as_of=None, root=None) -> QuoteSnapshot | None` — Return the last snapshot at or before `as_of` (or the day’s last) for the horizon.

## src/kalshi_alpha/drivers/macro_calendar/__init__.py
Module doc: Macro calendar driver producing release-day dummy variables.
Functions:
- `emit_day_dummies(start, end, out_path=None, *, offline=False, fixtures_dir=None) -> Path`
- `_coerce_date(value) -> date`
- `_date_range(start, end) -> Iterable[date]`
- `_cpi_release_dates(start, end, *, offline, fixtures_dir) -> set[date]`
- `_fomc_release_dates(start, end, fixtures_dir) -> set[date]`
- `_jobs_release_dates(start, end) -> set[date]`
- `_claims_release_dates(start, end) -> set[date]`
- `_parse_date(value) -> date | None`
- `_first_weekday(year, month, *, weekday) -> date`
- `_first_of_next_month(moment) -> date`

## src/kalshi_alpha/drivers/macro_calendar/cli.py
Module doc: Command-line utility to refresh macro calendar day dummies.
Functions:
- `_parse_date(value) -> date`
- `_build_parser() -> argparse.ArgumentParser`
- `_date_range(start, end) -> list[date]`
- `_resolve_fixture_records(payload) -> list[dict[str, object]]`
- `_load_fixture(path) -> dict[str, dict[str, bool]]`
- `_frame_from_history(start, end, mapping) -> pl.DataFrame`
- `main(argv=None) -> Path`

## src/kalshi_alpha/drivers/nws_cli/__init__.py
Module doc: NOAA/NWS Daily Climate Report (DCR) driver.
Classes:
- `StationConfig()`
- `DailyClimateRecord()`
Functions:
- `load_station_config(path) -> dict[str, StationConfig]`
- `parse_daily_climate_report(path) -> DailyClimateRecord` — Parse a minimal DCR text fixture.
- `parse_multi_station_report(path) -> list[DailyClimateRecord]`
- `settlement_assertion() -> str`
- `_to_date(value) -> date`
- `fetch_station_metadata(*, offline=False, fixtures_dir=None, force_refresh=False, session=None) -> dict[str, StationConfig]`
- `fetch_daily_climate_report(station_id, *, offline=False, fixtures_dir=None, force_refresh=False, session=None) -> DailyClimateRecord`
- `_parse_report_text(text) -> DailyClimateRecord`
- `_write_temp_station(payload) -> Path` — Persist payload temporarily for load_station_config when online fetch unsupported.

## src/kalshi_alpha/drivers/polygon_index/__init__.py
Module doc: Polygon index data integration.
No top-level functions or classes.

## src/kalshi_alpha/drivers/polygon_index/client.py
Module doc: Massive indices client supporting REST ingestion and optional Massive websocket.
Classes:
- `PolygonAPIError(RuntimeError)` — Raised when the Polygon API returns an error.
- `MinuteBar()`
- `IndexSnapshot()`
- `PolygonIndicesClient()` — REST + websocket client for Massive (Polygon) index data.
Functions:
- `_safe_float(value) -> float | None`
- `_parse_snapshot_timestamp(value) -> datetime | None`

## src/kalshi_alpha/drivers/polygon_index/snapshots.py
Module doc: Snapshot helpers for Polygon index data.
Functions:
- `write_minute_bars(symbol, bars) -> None`
- `write_snapshot(snapshot) -> None`
- `distance_to_strikes(price, strikes) -> dict[float, float]`
- `ewma_sigma_now(bars, *, span=30, min_samples=5) -> float`
- `micro_drift(bars, *, window=5) -> float`
- `build_snapshot_metrics(*, price, strikes, bars, ewma_span=30, drift_window=5) -> dict[str, object]`

## src/kalshi_alpha/drivers/polygon_index/symbols.py
Module doc: Shared symbol metadata for Polygon-powered index ladders.
Classes:
- `IndexSymbol()`
Functions:
- `supported_series() -> list[IndexSymbol]`
- `resolve_series(series) -> IndexSymbol`
- `polygon_tickers(series) -> list[str]`

## src/kalshi_alpha/drivers/polygon_index/windowed.py
Module doc: Windowed Polygon index websocket collector for hourly and close ladders.
Classes:
- `SubscriptionWindow()` — Inclusive subscription window for a specific aggregate timespan.
- `WindowLatency()`
- `WindowResult()`
- `PolygonWindowedCollector()` — Collect Polygon aggregates for hourly/close windows and persist parquet files.
Functions:
- `_target_datetime(trading_day, hour, minute) -> datetime`
- `hourly_windows(trading_day) -> list[SubscriptionWindow]` — Return per-hour subscription windows for the trading day.
- `close_window(trading_day) -> SubscriptionWindow` — Return the close subscription window for the trading day.
- `close_second_window(trading_day) -> SubscriptionWindow` — Return the second-level close window.
- `_truncate_latencies(latencies) -> WindowLatency`
- `_message_timestamp(message) -> datetime | None`
- `_safe_float(value) -> float | None`
- `_aggregated_value(entry, key) -> Any`
- `_aggregate_price(entry, key) -> float | None`
- `_aggregate_trades(entry) -> int | None`

## src/kalshi_alpha/drivers/polygon_index_ws.py
Module doc: Shared Polygon index websocket helper with singleton lifecycle and metrics.
Classes:
- `PolygonIndexWSConfig()` — Configuration for the shared Polygon index websocket connection.
- `PolygonWSClient()` — Lightweight wrapper around PolygonIndicesClient.stream_aggregates.
Functions:
- `active_connection_count() -> int` — Return the number of active Polygon index websocket connections.
- `last_message_at() -> datetime | None` — Return the timestamp of the most recent websocket message, if any.
- `last_message_age_seconds(now=None) -> float | None` — Return the age in seconds of the most recent websocket message.
- `get_shared_connection(config=None) -> PolygonWSClient` — Return the singleton Polygon websocket client for this process.
- `close_shared_connection() -> None` — Close and reset the shared websocket connection, if active.
- `polygon_index_ws(config=None) -> AsyncGenerator[PolygonWSClient, None]` — Context manager that yields the shared connection and cleans up on exit.
- `_record_message_timestamp(timestamp=None) -> None`
- `_increment_active_connections() -> None`
- `_decrement_active_connections() -> None`

## src/kalshi_alpha/drivers/treasury_yields/__init__.py
Module doc: Treasury par yield driver with offline support.
Classes:
- `ParYield()`
Functions:
- `_daily_dir() -> Path`
- `_latest_parquet_path() -> Path`
- `fetch_daily_yields(*, offline=False, fixtures_dir=None, force_refresh=False, session=None) -> list[ParYield]` — Fetch daily par yields.
- `_parse_treasury_csv(csv_text) -> list[ParYield]`
- `dgs10_latest_rate(yields) -> float | None`
- `yields_to_frame(yields, *, persist=True) -> pl.DataFrame`
- `today_close() -> pl.DataFrame` — Return the most recent closing par yields as a DataFrame.
- `yesterday_close() -> pl.DataFrame` — Return the prior trading day's closing par yields.
- `_write_processed_parquet(frame) -> None`
- `_list_daily_snapshots() -> list[tuple[date, Path]]`
- `_coerce_date(value) -> date`
- `_normalize_latest_row(csv_text) -> str` — Rewrite the first data row to use today's date so freshness gates pass.

## src/kalshi_alpha/exec/__init__.py
Module doc: Execution scaffolding for dry-run ladder scanning.
No top-level functions or classes.

## src/kalshi_alpha/exec/backtest_index_polygon.py
Module doc: CLI entrypoint for Polygon-only index ladder backtests.
Functions:
- `_parse_date(value) -> date | None`
- `_parse_args(argv=None) -> argparse.Namespace`
- `main(argv=None) -> None`

## src/kalshi_alpha/exec/brokers/__init__.py
Module doc: Broker integration placeholders (dry-run only).
No top-level functions or classes.

## src/kalshi_alpha/exec/brokers/kalshi/__init__.py
Module doc: Placeholder for future authenticated Kalshi broker integration.
No top-level functions or classes.

## src/kalshi_alpha/exec/calibration_ages.py
Module doc: Calibration age inspection and reporting for index ladders.
Classes:
- `CalibrationAgeResult()`
- `CalibrationSeriesSummary()`
Functions:
- `_ensure_utc(moment) -> datetime`
- `_format_age(age_hours) -> str`
- `_format_timestamp(value) -> str`
- `_hourly_candidates(root, slug, hour) -> list[Path]`
- `_close_candidates(root, slug) -> list[Path]`
- `_age_from_mtime(path, now_utc) -> tuple[float | None, str | None]`
- `_evaluate_candidates(*, series, horizon, candidates, now_utc, max_age_hours) -> CalibrationAgeResult`
- `inspect_calibration_ages(*, now=None, root=CALIBRATION_ROOT, max_age_days=DEFAULT_MAX_AGE_DAYS, series=None, hourly_hours=DEFAULT_HOURLY_HOURS) -> list[CalibrationAgeResult]`
- `summarize_by_series(results) -> dict[str, CalibrationSeriesSummary]`
- `render_markdown(results, *, asof_date, generated_at) -> str`
- `write_report(results, *, asof_date, output_path, generated_at) -> Path`
- `_parse_date(value, *, now_et) -> date`
- `_build_parser() -> argparse.ArgumentParser`
- `main(argv=None) -> int`

## src/kalshi_alpha/exec/collectors/__init__.py
Module doc: Collectors for external data feeds.
No top-level functions or classes.

## src/kalshi_alpha/exec/collectors/kalshi_tob.py
Module doc: Capture Kalshi top-of-book snapshots for fill modeling.
Functions:
- `parse_args(argv=None) -> argparse.Namespace`
- `main(argv=None) -> None`
- `_build_client(fixtures_root, offline) -> KalshiPublicClient`
- `_resolve_markets(client, series_list) -> list[Market]`
- `_snapshot_from_orderbook(market, orderbook, captured_at) -> dict[str, object]`
- `_best_entry(entries, side=None) -> dict[str, float]`

## src/kalshi_alpha/exec/collectors/polygon_ws.py
Module doc: Massive (Polygon) websocket collector for index ladders.
Classes:
- `CollectorConfig()`
- `CadenceTracker()`
- `TooManyConnectionsError(RuntimeError)` — Raised when Massive websocket rejects the connection due to max_connections.
Functions:
- `_normalize_status(value) -> str | None`
- `_format_text(value) -> str | None`
- `_market_status_for_symbol(payload, symbol) -> str | None`
- `_inactive_symbols(payload, symbols) -> tuple[list[str], dict[str, str]]`
- `_resolved_aliases(raw) -> AliasMap`
- `_parse_args(argv=None) -> CollectorConfig`
- `_normalize_entries(payload) -> Iterable[dict[str, Any]]`
- `_process_entries(*, entries, alias_map, channel_prefix, now, proc_parquet, freshness_config, freshness_output, tracker) -> None`
- `_iter_status(payload) -> Iterable[dict[str, Any]]`
- `_await_status(websocket, *, expected) -> list[object]` — Drain websocket until one of the expected status codes (case-insensitive) arrives.
- `_connect_once(config, *, ssl_context, connection_factory=None) -> None`
- `_run_forever(config) -> None`
- `main(argv=None) -> None`
- `_value_fallback_loop(*, config, tracker, client) -> None`

## src/kalshi_alpha/exec/collectors/tob_logger.py
Module doc: Bounded top-of-book snapshot + quote-intent logger for index ladders.
Classes:
- `TobSnapshotLogger()`
Functions:
- `build_tob_snapshot(*, run_id, ts_utc, series, window_label, window_ts_utc, window_ts_et, market_ticker, market_id, orderbook, depth=DEFAULT_TOB_DEPTH, max_bytes=DEFAULT_TOB_MAX_BYTES) -> dict[str, Any] | None`
- `build_quote_intent(*, run_id, ts_utc, series, window_label, window_ts_utc, window_ts_et, market_ticker, market_id, side, price, size, tob_ts=None, max_bytes=DEFAULT_INTENT_MAX_BYTES) -> dict[str, Any] | None`
- `enforce_snapshot_bounds(snapshot, *, max_bytes=DEFAULT_TOB_MAX_BYTES) -> dict[str, Any] | None`
- `_trim_levels(levels) -> list[dict[str, Any]] | None`
- `_window_id(series, window_label, window_ts_utc, fallback_ts) -> str`
- `_top_levels(orderbook, *, depth) -> tuple[list[dict[str, Any]], tuple[float | None, float | None], tuple[float | None, float | None]]`
- `_sorted_levels(entries, *, descending) -> list[tuple[float, float]]`
- `_safe_float(value) -> float | None`
- `_estimate_size(payload) -> int`
- `_as_iso(moment) -> str`

## src/kalshi_alpha/exec/fees.py
Module doc: Execution-time fee helpers backed by configs/fees.json.
Classes:
- `FeeSeriesRule()` — Per-series fee coefficients used for order-level rounding.
- `FeeConfig()`
Functions:
- `_round_up(value, quantum, *, mode) -> Decimal`
- `_to_decimal(value, *, field) -> Decimal`
- `load_fee_config(path=None) -> FeeConfig` — Load maker/taker fee coefficients for indices.
- `order_fee(*, series, price, contracts, liquidity, config=None) -> Decimal` — Compute the rounded fee charged for a single order submission.
- `fee_breakdown(*, series, price, contracts, liquidity='maker', config=None) -> dict[str, Decimal]` — Return per-order and per-contract fee metrics for metadata logging.

## src/kalshi_alpha/exec/gate_utils.py
Module doc: Shared helpers for quality gate configuration and artifacts.
Functions:
- `resolve_quality_gate_config_path() -> Path`
- `write_go_no_go(result, *, scope=None, scoped_blockers=None, unscoped_blockers=None, extra=None) -> Path`

## src/kalshi_alpha/exec/heartbeat.py
Module doc: Heartbeat and kill-switch utilities for execution pipelines.
Functions:
- `_state_dir() -> Path`
- `default_heartbeat_path() -> Path`
- `default_kill_switch_path() -> Path`
- `write_heartbeat(*, mode, monitors=None, extra=None, now=None, path=None) -> Path` — Persist the latest execution heartbeat.
- `load_heartbeat(path=None) -> dict[str, Any] | None`
- `heartbeat_stale(*, threshold, now=None, path=None) -> tuple[bool, dict[str, Any] | None]`
- `resolve_kill_switch_path(path=None) -> Path`
- `kill_switch_engaged(path=None) -> bool`

## src/kalshi_alpha/exec/housekeep.py
Module doc: Artifact retention housekeeping utility.
Classes:
- `Candidate(NamedTuple)`
Functions:
- `parse_args(argv=None) -> argparse.Namespace`
- `main(argv=None) -> None`
- `_prune_root(root, cutoff) -> int`
- `_collect_candidates(root) -> Iterable[Candidate]`
- `_detect_category(path) -> str | None`
- `_select_preserved(candidates) -> dict[str, Path]`
- `_delete_path(path) -> None`

## src/kalshi_alpha/exec/index_paper_ledger.py
Module doc: Lightweight JSONL ledger for index paper (dry) trades.
Functions:
- `_resolve_path(ledger_path=None) -> Path`
- `_normalize_timestamp(value) -> str`
- `_normalize_side(value) -> str`
- `_coerce_float(field, value) -> float`
- `log_index_paper_trade(record, *, ledger_path=None) -> Path` — Append a single index paper trade to the JSONL ledger.

## src/kalshi_alpha/exec/ingest/polygon_index.py
Module doc: Download Polygon index history into the raw datastore.
Functions:
- `_parse_args(argv=None) -> argparse.Namespace`
- `_parse_date(label) -> datetime`
- `main(argv=None) -> None`

## src/kalshi_alpha/exec/ledger/__init__.py
Module doc: Paper trading ledger utilities.
Classes:
- `ExecutionMetrics(TypedDict)`
- `FillRecord()`
- `PaperLedger()`
Functions:
- `simulate_fills(proposals, orderbooks, *, mode='top', slippage_model=None, schedule=DEFAULT_FEE_SCHEDULE, artifacts_dir=None, fill_estimator=None, alpha_curve=None, slippage_curve=None, ledger_series=None, market_event_lookup=None, manifest_path=None) -> PaperLedger`
- `_round_cents(value) -> float`
- `_derive_fill_price(proposal, orderbook, *, mode, slippage_model) -> tuple[float, float]`
- `_expected_value_with_fill(proposal, fill_price, *, schedule, contracts_override=None) -> float`
- `_best_bid_price(orderbook) -> float | None`
- `_best_ask_price(orderbook) -> float | None`
- `_side_depth_total(side, orderbook) -> float`
- `_seconds_until_event(event_label, timestamp_et) -> float | None`
- `_parse_event_timestamp(event_label) -> datetime | None`

## src/kalshi_alpha/exec/ledger/aggregate.py
Module doc: Aggregate paper ledger CSV outputs into a single Parquet dataset.
Functions:
- `parse_args(argv=None) -> argparse.Namespace`
- `main(argv=None) -> None`
- `_discover_ledger_csv(reports_dir) -> list[Path]`
- `_write_empty_output(output_path) -> None`

## src/kalshi_alpha/exec/ledger/schema.py
Module doc: Typed ledger schema definitions.
Classes:
- `LedgerRowV1(BaseModel)` — Canonical ledger row schema.

## src/kalshi_alpha/exec/limits.py
Module doc: Limit enforcement helpers for proposal generation and broker validation.
Classes:
- `LimitViolation(RuntimeError)` — Raised when a proposal exceeds PAL or loss budgets.
- `LossBudget()`
- `ProposalLimitChecker()` — Applies PAL and stop-loss budgets before proposals are admitted.

## src/kalshi_alpha/exec/live_smoke.py
Module doc: Read-only smoke test for Kalshi index ladders.
Classes:
- `SeriesStatus()`
Functions:
- `_now_et() -> datetime`
- `_target_hour_label(now_et) -> str`
- `_series_map(client) -> dict[str, str]`
- `_event_tickers(client, series_id) -> list[str]`
- `_check_u_series(client, now_et) -> list[SeriesStatus]`
- `_outstanding_summary() -> dict[str, Any]`
- `run_smoke(base_url=None) -> tuple[int, dict[str, Any]]`
- `main(argv=None) -> None`

## src/kalshi_alpha/exec/market_status.py
Module doc: Print Polygon/Massive market status for ops checks.
Functions:
- `main(argv=None) -> int`

## src/kalshi_alpha/exec/monitors/__init__.py
Module doc: Runtime monitoring entry points.
No top-level functions or classes.

## src/kalshi_alpha/exec/monitors/cli.py
Module doc: Command-line entry point for runtime monitors.
Functions:
- `parse_args(argv=None) -> argparse.Namespace`
- `main(argv=None) -> int`
- `_update_report(report_path, summary_lines, generated_at) -> None`
- `_maybe_notify(summary, results) -> None`

## src/kalshi_alpha/exec/monitors/fee_rules.py
Module doc: Helper utilities for fee/rule watcher artifacts.
Functions:
- `load_status(path=None) -> dict[str, object] | None`
- `is_ready(payload=None) -> tuple[bool, str | None]`

## src/kalshi_alpha/exec/monitors/freshness.py
Module doc: Data feed freshness monitor for ramp readiness.
Classes:
- `FeedState()` — Computed freshness payload for a single feed.
- `FreshnessConfig()`
Functions:
- `load_config(path=None) -> FreshnessConfig` — Load freshness configuration and merge with defaults.
- `compute_freshness(*, config=None, now=None, proc_root=None) -> tuple[list[FeedState], dict[str, Any]]` — Compute per-feed freshness state and aggregate metrics.
- `write_freshness_artifact(*, config_path=None, output_path=None, now=None, proc_root=None, emit_table=True) -> dict[str, Any]` — Compute freshness metrics and persist monitor artifact.
- `load_artifact(path=None) -> dict[str, Any] | None` — Load freshness artifact from disk.
- `summarize_artifact(payload, *, artifact_path, scope=None) -> dict[str, Any]` — Normalize the freshness artifact into a ramp-friendly summary.
- `_apply_scope_to_summary(summary, scope) -> dict[str, Any]`
- `_normalize_scope(scope) -> str | None`
- `_fetch_market_status() -> dict[str, Any] | None`
- `_market_status_summary(payload) -> dict[str, Any]`
- `_inactive_market_reason(payload) -> str | None`
- `_feed_in_scope(feed, scope) -> bool`
- `main(argv=None) -> int`
- `_evaluate_feed(feed_id, cfg, now, proc_root, raw_root) -> FeedState`
- `_evaluate_polygon_ws(cfg, label, required, now, raw_root) -> FeedState`
- `_evaluate_bls(cfg, label, required, now, proc_root) -> FeedState`
- `_evaluate_claims(cfg, label, required, now, proc_root) -> FeedState`
- `_evaluate_treasury(cfg, label, required, now, proc_root) -> FeedState`
- `_evaluate_cleveland(cfg, label, required, now, proc_root) -> FeedState`
- `_evaluate_aaa(cfg, label, required, now, proc_root) -> FeedState`
- `_evaluate_weather(cfg, label, required, now, proc_root) -> FeedState`
- `_latest_parquet(proc_root, namespace) -> Path | None`
- `_latest_snapshot_path(raw_root, namespace) -> Path | None`
- `_parse_snapshot_timestamp(name) -> datetime | None`
- `_extract_datetime(value) -> datetime | None`
- `_extract_date_as_datetime(value) -> datetime | None`
- `_ensure_utc(moment) -> datetime`
- `_age_minutes(last_ts, now) -> float | None`
- `_age_days(last_ts, now) -> float | None`
- `_business_days(start, end) -> float`
- `_to_float(value, *, default=None) -> float | None`
- `_round(value) -> float | None`
- `_build_table(states) -> list[dict[str, Any]]`
- `_json_safe(data) -> object`

## src/kalshi_alpha/exec/monitors/runtime.py
Module doc: Compute runtime execution monitors and persist artifacts.
Classes:
- `MonitorResult()` — Container for an individual monitor evaluation.
- `RuntimeMonitorConfig()` — Configuration knobs for runtime monitor evaluation.
Functions:
- `compute_runtime_monitors(*, config=None, telemetry_root=TELEMETRY_ROOT, ledger_path=LEDGER_PATH, alpha_state_path=ALPHA_STATE_PATH, drawdown_state_dir=None, now=None) -> list[MonitorResult]` — Evaluate runtime monitors and return structured results.
- `write_monitor_artifacts(results, *, artifacts_dir=MONITOR_ARTIFACTS_DIR, generated_at=None) -> list[Path]` — Persist monitor results to JSON artifacts.
- `build_report_summary(results) -> list[str]` — Build human-readable summary lines for REPORT.md.
- `_monitor_ev_gap(ledger, cfg, moment) -> MonitorResult`
- `_monitor_fill_vs_alpha(ledger, alpha_state, cfg, moment) -> MonitorResult`
- `_monitor_ev_sequential(result, params) -> MonitorResult`
- `_monitor_freeze_windows(evaluations) -> MonitorResult`
- `_monitor_drawdown(drawdown_state_dir, cfg, moment) -> MonitorResult`
- `_monitor_ws_disconnects(events, cfg, moment) -> MonitorResult`
- `_monitor_auth_errors(events, cfg) -> MonitorResult`
- `_monitor_kill_switch(path) -> MonitorResult`
- `_load_telemetry_events(base_dir, *, since) -> list[dict[str, Any]]`
- `_load_ledger(path) -> pl.DataFrame`
- `_load_alpha_state(path) -> dict[str, float]`
- `_parse_timestamp(value) -> datetime | None`
- `_ensure_utc(moment) -> datetime`

## src/kalshi_alpha/exec/monitors/sequential.py
Module doc: Sequential change-detection guardrails for EV deltas.
Classes:
- `SequentialGuardParams()`
- `SequentialGuardResult()`
Functions:
- `evaluate_sequential_guard(ledger, *, params, window_start=None) -> SequentialGuardResult` — Evaluate a CuSum-style sequential guard on Δbps streams.

## src/kalshi_alpha/exec/monitors/sigma_drift.py
Module doc: Helpers for sigma drift monitor artifacts.
Functions:
- `load_artifact(path=None) -> dict[str, object] | None`
- `shrink_for_series(series, *, artifact=None) -> float | None`

## src/kalshi_alpha/exec/monitors/summary.py
Module doc: Helpers for aggregating persisted monitor artifacts.
Classes:
- `MonitorArtifactsSummary()`
Functions:
- `summarize_monitor_artifacts(artifacts_dir, *, now, window) -> MonitorArtifactsSummary` — Aggregate monitor JSON artifacts into a summary snapshot.
- `_normalize_status(raw) -> str`
- `_parse_timestamp(value) -> datetime | None`

## src/kalshi_alpha/exec/pilot/__init__.py
Module doc: Pilot execution helpers.
No top-level functions or classes.

## src/kalshi_alpha/exec/pilot/config.py
Module doc: Utilities for loading pilot mode configuration.
Classes:
- `PilotConfig()` — Immutable configuration for live pilot sessions.
Functions:
- `_normalize_series(series) -> tuple[str, ...]`
- `resolve_pilot_config_path(override=None) -> Path | None`
- `load_pilot_config(path=None) -> PilotConfig`

## src/kalshi_alpha/exec/pilot/runtime.py
Module doc: Runtime helpers for configuring pilot sessions.
Classes:
- `PilotSession()` — Captures identifying metadata for a pilot run.
Functions:
- `_generate_session_id(*, prefix, series, now, token_factory) -> str`
- `apply_pilot_mode(args, *, now=None, token_factory=None) -> PilotSession | None` — Mutate CLI args in-place to enforce pilot constraints.
- `_delta_stats(records) -> tuple[float | None, float | None, int]`
- `_resolve_fill_gap(monitors, summary) -> float | None`
- `build_pilot_session_payload(*, session, ledger, monitors, monitor_summary, broker_status, generated_at=None) -> dict[str, Any]`
- `write_pilot_session_artifact(*, session, ledger, monitors, monitor_summary, broker_status, artifacts_dir=None, generated_at=None) -> Path`

## src/kalshi_alpha/exec/pilot_bundle.py
Module doc: Assemble a single tarball with key pilot readiness artifacts.
Classes:
- `BundleItem()`
Functions:
- `_safe_load_json(path) -> dict[str, object] | None`
- `parse_args(argv=None) -> argparse.Namespace`
- `main(argv=None) -> int`
- `_collect_core_reports(reports_dir) -> list[BundleItem]`
- `_collect_monitors(reports_dir) -> list[BundleItem]`
- `_collect_scoreboards(reports_dir) -> list[BundleItem]`
- `_collect_ladder_reports(reports_dir) -> list[BundleItem]`
- `_collect_telemetry(data_root, limit) -> list[BundleItem]`
- `_build_pilot_readme(reports_dir) -> str | None`
- `_add_text_file(bundle, name, content) -> None`
- `_add_manifest(bundle, manifest) -> None`

## src/kalshi_alpha/exec/pilot_readiness.py
Module doc: Compute pilot readiness for index ladders based on recent paper fills.
Classes:
- `SeriesReadiness()`
Functions:
- `_load_ledger(path) -> pl.DataFrame`
- `_series_fills(subset) -> float`
- `_delta_stats(subset) -> tuple[float, float]`
- `_alpha_gap_mean(subset) -> float`
- `calibration_age_days(series, now) -> float | None`
- `freshness_status() -> tuple[bool, list[str]]`
- `evaluate_readiness(frame, *, now=None, window_days=WINDOW_DAYS_DEFAULT, calibration_summary=None) -> list[SeriesReadiness]`
- `render_markdown(results, *, window_days=WINDOW_DAYS_DEFAULT, freshness_ok=True, freshness_reasons=None, calibration_results=None) -> str`
- `generate_report(*, ledger_path=LEDGER_PATH, output_path=REPORT_PATH, window_days=WINDOW_DAYS_DEFAULT, now=None) -> list[SeriesReadiness]`
- `main(argv=None) -> None`

## src/kalshi_alpha/exec/pipelines/calendar.py
Module doc: Calendar-aware run window resolution for daily pipelines.
Classes:
- `RunWindow()`
Functions:
- `resolve_run_window(*, mode, target_date, now, proc_root=None) -> RunWindow`
- `_resolve_pre_cpi(target_date, now, proc_root) -> RunWindow`
- `_resolve_pre_claims(target_date, now) -> RunWindow`
- `_resolve_teny_close(target_date) -> RunWindow`
- `_resolve_weather_cycle(target_date, now) -> RunWindow`
- `_latest_parquet(directory) -> Path | None`
- `_ensure_datetime(value) -> datetime`
- `_serialize_dt(value) -> dict[str, str] | None`
- `_us_holidays(year) -> set[date]`
- `_nth_weekday_of_month(year, month, weekday, n) -> date` — Return the date of the nth weekday (Monday=0) for the given month.
- `_last_weekday_of_month(year, month, weekday) -> date`

## src/kalshi_alpha/exec/pipelines/daily.py
Module doc: Daily orchestration pipeline for ladder strategies.
Functions:
- `parse_args(argv=None) -> argparse.Namespace`
- `_resolve_fill_alpha_value(fill_alpha_arg, series) -> tuple[float, bool]`
- `main(argv=None) -> None`
- `run_mode(mode, args) -> None`
- `run_ingest(args, log) -> None`
- `run_calibrations(args, log, heartbeat_cb=None) -> None`
- `_evaluate_quality_gates(args, now_utc, *, monitors=None, apply_side_effects=True) -> tuple[QualityGateResult, dict[str, object]]`
- `run_quality_gate_step(args, now_utc, log, *, monitors=None) -> QualityGateResult`
- `_write_latest_manifest(manifest_path) -> None`
- `_parse_date(value) -> date`
- `_apply_fill_realism_gate(log, ledger, monitors) -> tuple[float | None, QualityGateResult | None]`
- `_compute_next_window(*, mode, start_date, now_utc, max_days=14) -> tuple[date, RunWindow]` — Find the next run window on or after ``start_date`` that closes in the future.
- `_format_window_line(prefix, run_window, *, include_notes=False) -> str`
- `_print_next_window(mode, window_date, run_window) -> None`
- `_apply_snap_option(*, mode, args, now_utc) -> tuple[datetime, date, RunWindow, bool] | None` — Handle snap-to-window logic. Returns (now, date, window, waited) or None if we should exit.
- `run_scan(mode, args, log, *, series, fill_alpha_value, fill_alpha_auto) -> None`
- `resolve_series(mode) -> str | None`
- `run_step(name, log, func, metadata=None) -> None`
- `write_log(mode, log, now_utc) -> None`

## src/kalshi_alpha/exec/pipelines/preflight.py
Module doc: Lightweight preflight helper for daily pipeline windows.
Functions:
- `parse_args(argv=None) -> argparse.Namespace`
- `main(argv=None) -> None`

## src/kalshi_alpha/exec/pipelines/today.py
Module doc: Autonomous "today" orchestration that selects daily modes based on calendars.
Classes:
- `ScheduledRun()`
Functions:
- `parse_args(argv=None) -> argparse.Namespace`
- `_fmt_float(value) -> str` — Format floats for CLI forwarding without trailing zeros.
- `main(argv=None) -> None`
- `_print_outstanding(prefix) -> None`
- `_plan_runs(now, *, include_weather, proc_root=None) -> list[ScheduledRun]`
- `_now() -> datetime`
- `_badge_path() -> Path`
- `_latest_manifest_marker() -> Path`
- `_load_latest_manifest() -> Path | None`
- `_print_manifest_link() -> None`
- `_load_go_status(path) -> bool | None`

## src/kalshi_alpha/exec/pipelines/week.py
Module doc: Weekly orchestration wrapper running daily pipeline modes in sequence.
Classes:
- `WeekRun()`
Functions:
- `parse_args(argv=None) -> argparse.Namespace`
- `_fmt(value) -> str`
- `_default_modes(include_weather) -> list[str]`
- `main(argv=None) -> None`
- `_print_outstanding(prefix) -> None`
- `_paper_live_schedule(now, *, include_weather) -> list[WeekRun]`
- `_window_date(window, *, fallback) -> date`
- `_auto_resolve_week_run(mode, phase, target_date, now) -> WeekRun`
- `_advance_target_date(mode, phase, window, current_date) -> date`
- `_phase_adjusted_date(mode, phase, window, fallback) -> date`
- `_format_window_range(window) -> str | None`
- `_previous_business_day(candidate) -> date`
- `_next_weekday(start, target_weekday) -> date`

## src/kalshi_alpha/exec/policy/freeze.py
Module doc: Utilities for evaluating pre-event freeze windows per series family.
Classes:
- `FreezeEvaluation()`
Functions:
- `evaluate_freeze_for_series(series, *, now, proc_root=None) -> FreezeEvaluation`
- `_resolve_mode(series) -> str | None`
- `_serialize_dt(value) -> dict[str, str] | None`

## src/kalshi_alpha/exec/preflight_index.py
Module doc: GO/NO-GO checks for SPX/NDX index ladder windows.
Classes:
- `PreflightResult()`
Functions:
- `_ensure_et(moment) -> datetime`
- `_file_age_days(path, now) -> float | None` — Return age in days using generated_at when present, else mtime.
- `_parse_timestamp(value) -> datetime | None`
- `_basis_audit_path(root, series, asof) -> Path`
- `_check_basis_audit(*, series, asof, now_utc, root) -> tuple[bool, list[str], dict[str, object]]`
- `_calibration_check(*, now, params_root, max_age_days) -> tuple[bool, list[str], dict[str, float]]`
- `_polygon_ping(timeout) -> bool`
- `_missing_env_vars(vars_to_check) -> list[str]`
- `run_preflight(now_et, *, params_root=None, kill_switch_file=None, polygon_timeout=2.0, polygon_ping=None, require_kalshi=True, require_polygon=None, require_basis_audit=None, basis_root=None, series=None, freshness_artifact_path=None, require_freshness=None, freshness_scope=FRESHNESS_SCOPE) -> PreflightResult` — Evaluate GO/NO-GO checks for index ladder windows.
- `_parse_now(value) -> datetime | None`
- `_series_labels(series_horizons=SERIES_HORIZONS) -> tuple[str, ...]`
- `format_preflight_summary(result, *, label, series=None, broker=None) -> str`
- `write_go_no_go_artifact(result, *, output_path=None, source='preflight_index') -> Path`
- `_build_parser() -> argparse.ArgumentParser`
- `main(argv=None) -> int`

## src/kalshi_alpha/exec/quote_microprice.py
Module doc: Microprice calculator and replacement throttle for ladder quoting.
Classes:
- `MicropriceSignal()`
- `ReplacementThrottle()` — Simple time-window throttle that bounds replacements per ladder bin.
Functions:
- `compute_signal(orderbook, *, tick_size=0.01) -> MicropriceSignal` — Return microprice-derived signal using top-of-book depth weighting.

## src/kalshi_alpha/exec/quote_optim.py
Module doc: Quote optimization utilities: PMF skew gating, microprice bias, freshness widening.
Classes:
- `QuoteContext()` — Per-proposal signal bundle used to calculate EV penalties.
- `QuoteOptimizer()` — Heuristic EV penalty engine with replacement throttling.

## src/kalshi_alpha/exec/reports/__init__.py
Module doc: Generate markdown reports for ladder scans.
Functions:
- `_float_metric(metrics, key) -> float | None`
- `_int_metric(metrics, key) -> int`
- `_resolve_latest_go_artifact(candidate) -> Path | None` — Return the most recent go/no-go artifact matching ``candidate``.
- `_load_go_status(artifact_path) -> bool | None` — Return GO/NO-GO boolean from artifact, or ``None`` if unavailable.
- `write_markdown_report(*, series, proposals, ledger, output_dir, monitors=None, exposure_summary=None, manifest_path=None, go_status=None, go_artifact_path=GO_ARTIFACT_PATH, fill_alpha=None, mispricings=None, model_metadata=None, scorecard_summary=None, outstanding_summary=None, pilot_metadata=None, execution_metrics=None) -> Path`
- `_expected_vs_realized_rows(ledger) -> list[str]`
- `_confidence_badge(sample_size, t_stat) -> str`
- `_ev_plot_lines(expected, realized) -> list[str]`
- `_ev_honesty_rows(ledger, *, table_data=None, max_delta=None) -> tuple[list[str], float | None]`
- `_format_pilot_header(metadata) -> str`

## src/kalshi_alpha/exec/reports/ramp.py
Module doc: Compute pilot ramp readiness reports.
Classes:
- `RampPolicyConfig()`
Functions:
- `compute_ramp_policy(*, ledger_path=LEDGER_PATH, artifacts_dir=GO_NO_GO_DIR, monitor_artifacts_dir=MONITOR_ARTIFACTS_DIR, drawdown_state_dir=None, config=None, now=None, pilot_session_path=None, bin_overrides_path=None) -> dict[str, Any]`
- `write_ramp_outputs(policy, *, json_path=JSON_OUTPUT, markdown_path=MARKDOWN_OUTPUT) -> None`
- `main(argv=None) -> int`
- `_load_pilot_session(path) -> dict[str, Any] | None`
- `_normalize_bin_overrides(payload) -> dict[str, list[dict[str, Any]]]`
- `_load_bin_overrides(path) -> tuple[dict[str, list[dict[str, Any]]], Path | None]`
- `_safe_float(value) -> float | None`
- `_normalize_side(value, default=None) -> str | None`
- `_ev_honesty_by_series(session_payload, *, default_series) -> dict[str, list[dict[str, Any]]]`
- `_build_bin_records(series, ev_rows, threshold, manual_overrides) -> list[dict[str, Any]]`
- `_format_minutes(value) -> float | None`
- `_format_number_for_markdown(value) -> str`
- `_file_age_minutes(path, moment) -> float | None`
- `_dedupe_reasons(reasons) -> list[str]`
- `_aggregate_series(ledger, cfg, moment) -> list[dict[str, Any]]`
- `_load_ledger(path) -> pl.DataFrame`
- `_load_guardrail_events(artifacts_dir, *, since) -> dict[str, int]`
- `_parse_timestamp(value) -> datetime | None`
- `_ensure_utc(moment) -> datetime`

## src/kalshi_alpha/exec/reports/telemetry_volume.py
Module doc: Generate ops telemetry volume report for bounded TOB + quote-intent streams.
Functions:
- `_parse_args(argv=None) -> argparse.Namespace`
- `_parse_date(value) -> date`
- `_utc_today() -> date`
- `_stream_path(root, stream, run_id) -> Path`
- `_count_lines(path) -> tuple[int, int, int, int]`
- `_format_stream(*, name, path, stats) -> list[str]`
- `main(argv=None) -> Path`

## src/kalshi_alpha/exec/runners/__init__.py
Module doc: Command-line runners for ladder scanning workflows.
No top-level functions or classes.

## src/kalshi_alpha/exec/runners/micro_index.py
Module doc: Microlive runner for index ladders: one window, 1-lot maker quotes.
Functions:
- `_default_hourly_target(reference) -> time`
- `_parse_args(argv=None) -> argparse.Namespace`
- `_build_scan_args(args) -> list[str]`
- `_log_ops_window(series, *, reference, quiet) -> None`
- `_refit_execution_curves(series) -> None`
- `main(argv=None) -> None`

## src/kalshi_alpha/exec/runners/orders_doctor.py
Module doc: Command-line helper for cleaning up outstanding DRY orders.
Functions:
- `_parse_args(argv=None) -> argparse.Namespace`
- `_resolved_state(path) -> OutstandingOrdersState`
- `main(argv=None) -> None`

## src/kalshi_alpha/exec/runners/pilot.py
Module doc: Single-entry CLI wrapper for pilot ladder sessions.
Functions:
- `_parse_args(argv=None) -> argparse.Namespace`
- `_build_forward_args(config) -> list[str]`
- `main(argv=None) -> int`

## src/kalshi_alpha/exec/runners/pilot_close.py
Module doc: Entry point for maker-only close index pilot sessions.
Functions:
- `_parse_args(argv=None) -> argparse.Namespace`
- `_forward_args(series, config) -> list[str]`
- `main(argv=None) -> int`

## src/kalshi_alpha/exec/runners/pilot_hourly.py
Module doc: Entry point for maker-only hourly index pilot sessions.
Functions:
- `_parse_args(argv=None) -> argparse.Namespace`
- `_forward_args(series, config) -> list[str]`
- `main(argv=None) -> int`

## src/kalshi_alpha/exec/runners/risk_preview.py
Module doc: CLI to preview risk posture before running a ladder scan.
Functions:
- `parse_args(argv=None) -> argparse.Namespace`
- `main(argv=None) -> None`
- `_parse_date(value) -> date`
- `_load_portfolio_config(path) -> PortfolioConfig | None`
- `_load_pal_policy(path, series) -> PALPolicy | None`
- `_load_ledger(path, series) -> pl.DataFrame`
- `_compute_exposures(ledger, portfolio_config, series) -> tuple[dict[str, float], dict[str, float], dict[str, float]]`
- `_estimate_var(config, exposures) -> float`
- `_pal_limits_summary(policy) -> dict[str, float]`
- `_run_quality_gates(config_path) -> QualityGateResult`
- `_print_gate_result(result) -> None`

## src/kalshi_alpha/exec/runners/scan_ladders.py
Module doc: CLI scanner that produces dry-run order proposals for Kalshi ladders.
Classes:
- `Proposal()`
- `ScanOutcome()`
- `BinConstraintEntry()` — Represents a per-bin EV honesty constraint sourced from readiness outputs.
- `BinConstraintResolver()` — Lookup helper that applies per-bin contract caps/weights.
Functions:
- `_load_data_freshness_summary(scope) -> dict[str, object]`
- `_resolve_quality_gate_scope(args) -> str | None`
- `_safe_float(value) -> float | None`
- `_resolve_fee_paths(series) -> dict[str, str | None]`
- `_polygon_ws_freshness_detail(*, gate_details, data_summary, fatal_reason) -> dict[str, object]`
- `_polygon_ws_age_ms(summary) -> float | None`
- `_freshness_fatal_reason(summary, *, require_polygon_ws) -> str | None`
- `_resolve_fill_alpha_arg(fill_alpha_arg, series) -> tuple[float, bool]`
- `_load_ev_honesty_constraints(series, readiness_path=None) -> BinConstraintResolver | None`
- `_load_honesty_clamp(series) -> float | None`
- `_clear_dry_orders_start(*, enabled, broker_mode, quiet, state=None) -> dict[str, int]` — Optionally clear outstanding DRY orders before generating proposals.
- `main(argv=None) -> None`
- `_build_client(fixtures_root, *, use_online) -> KalshiPublicClient`
- `_run_discovery(args) -> None`
- `_resolve_discover_day(args) -> date`
- `_build_pal_guard(args) -> PALGuard`
- `_build_risk_manager(args) -> PortfolioRiskManager | None`
- `_maybe_simulate_ledger(args, proposals, client, *, orderbooks=None, fill_alpha=None, series=None, events=None, markets=None) -> PaperLedger | None`
- `_attach_series_metadata(*, proposals, series, driver_fixtures, offline) -> None`
- `_maybe_write_report(args, proposals, ledger, monitors, exposure_summary, manifest_path, go_status, fill_alpha, mispricings=None, model_metadata=None, outstanding_summary=None, pilot_metadata=None, execution_metrics=None) -> None`
- `_window_label_from_monitors(monitors) -> str | None`
- `_window_timestamps_from_monitors(monitors) -> tuple[str | None, str | None]`
- `_log_index_paper_trades(proposals, monitors) -> None`
- `execute_broker(*, broker_mode, proposals, args, monitors, quiet, go_status=None) -> dict[str, object] | None`
- `_proposal_to_broker_order(proposal) -> BrokerOrder`
- `_enforce_broker_guards(proposals, args) -> None`
- `_quality_gate_for_broker(args, monitors, *, data_freshness_summary=None, scope=None) -> QualityGateResult`
- `_archive_and_replay(*, client, series, events, markets, orderbooks, proposals_path, driver_fixtures, scanner_fixtures, model_metadata=None) -> tuple[Path | None, Path | None]`
- `_enrich_manifest(manifest_path, *, proposals_path, driver_fixtures, scanner_fixtures) -> None`
- `_load_replay_for_ev_honesty(path) -> list[dict[str, object]]`
- `_compute_ev_honesty_rows(proposals, replay_rows) -> tuple[list[dict[str, object]], float | None]`
- `_apply_ev_honesty_gate(monitors, *, threshold) -> None`
- `_parse_date_arg(value) -> date`
- `_parse_datetime_arg(value) -> datetime`
- `parse_args(argv) -> argparse.Namespace`
- `_parse_hour_label(ticker) -> tuple[int, int] | None`
- `_format_hour_label(hour) -> str`
- `_default_hourly_target(now_utc) -> time`
- `_ops_window_for_series(series) -> IndexOpsWindow | None`
- `_cancel_buffer_seconds(series=None) -> float`
- `_ops_window_metadata(series, now_utc, *, target_time=None) -> dict[str, object]`
- `_scheduler_window_payload(window) -> dict[str, object] | None`
- `_u_series_roll_decision(now_utc) -> dict[str, object]`
- `_filter_events_by_hour(events, target_hour) -> list[Event]`
- `_filter_u_series_events(events, *, decision) -> list[Event]`
- `_expected_rule_hour(series_code) -> int | None`
- `_validate_index_rules(series, events, rule) -> dict[str, object]`
- `_clock_skew_seconds(reference_utc=None) -> float`
- `scan_series(*, series, client, min_ev, contracts, pal_guard, driver_fixtures, strategy_name, maker_only, allow_tails, risk_manager, max_var, offline, sizing_mode, kelly_cap, uncertainty_penalty=0.0, ob_imbalance_penalty=0.0, ev_honesty_shrink=0.9, daily_loss_cap=None, weekly_loss_cap=None, mispricing_only=False, max_legs=4, prob_sum_gap_threshold=0.0, model_version='v15', pilot_config=None, bin_constraints=None, now_override=None, target_time_override=None, var_limiter=None, correlation_guard=None, quote_optimizer=None, freshness_ms=None, sniper=False, sniper_threshold=0.05) -> ScanOutcome`
- `_strategy_pmf_for_series(*, series, strikes, fixtures_dir, override, offline, model_version='v15', orderbook_imbalance=None, event_timestamp=None, target_time=None) -> tuple[list[LadderBinProbability], dict[str, object]]`
- `_load_history(fixtures_dir, namespace) -> list[dict[str, object]]`
- `_polygon_client_cached() -> PolygonIndicesClient`
- `_load_index_snapshot(symbol, *, offline, fixtures_dir) -> IndexSnapshot`
- `_polygon_fixture_path(symbol, fixtures_dir) -> Path`
- `_snapshot_from_payload(symbol, payload) -> IndexSnapshot`
- `_maybe_float(value) -> float | None`
- `_parse_timestamp(value) -> datetime | None`
- `_minutes_to_target(now_utc, target) -> int`
- `_resolve_index_price(snapshot) -> float`
- `_macro_calendar_lookup(history, *, offline, fixtures_dir) -> dict[str, dict[str, float]]`
- `_normalize_macro_date(value) -> str | None`
- `_strip_dummy_prefix(column) -> str`
- `_market_survival_from_pmf(pmf, rungs) -> list[float]`
- `_adjacent_indices(pmf, rung_count) -> set[int]`
- `_limit_proposals_for_pilot(proposals, *, max_unique_bins) -> tuple[list[Proposal], int]`
- `_is_monotone(sequence) -> bool`
- `_evaluate_market(*, market_id, market_ticker, rungs, market_survival, strategy_survival, min_ev, contracts, pal_guard, allowed_indices, maker_only, risk_manager, max_var, strategy_name, sizing_mode, kelly_cap, uncertainty_penalty, ob_imbalance_penalty, daily_budget, weekly_budget, limit_checker, series_ticker, ev_shrink, bin_constraints=None, var_limiter=None, correlation_guard=None, quote_optimizer=None, orderbook=None, freshness_ms=None, replacement_throttle=None, sniper=False, sniper_threshold=0.05, now_ts=None) -> list[Proposal]`
- `_choose_side(per_contract_evs, *, maker_only) -> tuple[OrderSide, float]`
- `_collect_cdf_diffs(*, market_id, market_ticker, rungs, market_survival, strategy_survival) -> list[dict[str, object]]`
- `write_proposals(*, series, proposals, output_dir) -> Path`
- `_write_cdf_diffs(diffs) -> Path | None`
- `_load_portfolio_config() -> PortfolioConfig | None`
- `_compute_exposure_summary(proposals) -> dict[str, object]`
- `_find_series(client, ticker) -> Series`
- `_model_drift_flag(series_ticker) -> bool`
- `_tz_not_et() -> bool`

## src/kalshi_alpha/exec/scanners/__init__.py
Module doc: Utilities that reconcile ladder prices with strategy distributions.
No top-level functions or classes.

## src/kalshi_alpha/exec/scanners/cpi.py
Module doc: CPI ladder scanner utilities.
Functions:
- `strategy_pmf(strikes, *, fixtures_dir, offline=False, model_version='v15') -> list[LadderBinProbability]`

## src/kalshi_alpha/exec/scanners/fast_index.py
Module doc: Fast offline index scan helpers used by --fast-fixtures.
Classes:
- `FastIndexConfig()`
Functions:
- `_parse_timestamp(value) -> datetime`
- `_load_minutes(symbol, fixtures_root) -> pl.DataFrame`
- `_latest_price(frame) -> tuple[datetime, float]`
- `_build_row(*, series, target_hour, timestamp, price, contracts, min_ev) -> dict[str, object]`
- `_write_csv(rows, path) -> None`
- `_resolve_series(raw, *, default) -> tuple[str, ...]`
- `run_fast_hourly(args) -> None`
- `run_fast_close(args) -> None`

## src/kalshi_alpha/exec/scanners/index_scan_common.py
Module doc: Shared helpers for index ladder scanner CLIs.
Classes:
- `ScannerConfig()`
- `OpportunityRow()`
Functions:
- `_build_client(fixtures_root, *, offline) -> KalshiPublicClient`
- `_load_pal_policy(series) -> PALPolicy`
- `_pal_guard(series) -> PALGuard`
- `_simulate_execution(*, proposals, client, orderbooks, series_label, events, markets) -> PaperLedger`
- `_pair_rows(*, proposals, ledger, min_ev) -> list[OpportunityRow]`
- `_select_top_bins(rows, *, max_bins) -> list[OpportunityRow]`
- `_write_outputs(*, rows, series, output_dir, timestamp, monitors, ledger) -> tuple[Path | None, Path | None]`
- `run_index_scan(config) -> dict[str, dict[str, Path | None]]`
- `_persist_structure_artifact(series, monitors) -> None`
- `build_parser(default_series) -> argparse.ArgumentParser`
- `parse_timestamp(value) -> datetime | None`

## src/kalshi_alpha/exec/scanners/scan_index_close.py
Module doc: Scanner helpers for daily close index ladders.
Functions:
- `_preparse_fast_flags(argv) -> tuple[argparse.Namespace, list[str]]`
- `evaluate_close(strikes, yes_prices, inputs, *, contracts=1, min_ev=0.05) -> IndexScanResult`
- `main(argv=None) -> None`

## src/kalshi_alpha/exec/scanners/scan_index_hourly.py
Module doc: Scanner helpers and CLI for intraday hourly index ladders.
Classes:
- `QuoteOpportunity()`
- `IndexScanResult()`
Functions:
- `_preparse_fast_flags(argv) -> tuple[argparse.Namespace, list[str]]`
- `evaluate_hourly(strikes, yes_prices, inputs, *, contracts=1, min_ev=0.05) -> IndexScanResult`
- `main(argv=None) -> None`

## src/kalshi_alpha/exec/scanners/scan_index_noon.py
Module doc: Deprecated alias for hourly index ladder scanner CLI.
Functions:
- `main(argv=None) -> None`

## src/kalshi_alpha/exec/scanners/utils.py
Module doc: Utilities shared by ladder scanners.
Functions:
- `pmf_to_survival(pmf, strikes) -> list[float]`
- `expected_value_summary(*, contracts, yes_price, event_probability, schedule=DEFAULT_FEE_SCHEDULE, series=None, market_name=None) -> dict[str, float]`

## src/kalshi_alpha/exec/scoreboard.py
Module doc: Generate rolling performance scoreboards.
Functions:
- `_series_fills(frame) -> float`
- `_alpha_model_metrics(curve, subset) -> tuple[float | None, float | None]`
- `_slippage_model_metrics(curve, subset) -> tuple[float | None, float | None]`
- `parse_args(argv=None) -> argparse.Namespace`
- `main(argv=None) -> None`
- `_load_ledger() -> pl.DataFrame`
- `_load_calibrations() -> pl.DataFrame`
- `_load_alpha_state() -> dict[str, float]`
- `_load_freshness_summary() -> dict[str, object]`
- `_format_freshness_metrics(summary) -> list[str]`
- `_build_summary(ledger, calibrations, alpha_state, window_days, *, calibration_summary=None) -> list[dict[str, object]]`
- `_load_gate_metrics(window_days) -> dict[str, dict[str, int]]`
- `_load_honesty_metrics(window_days) -> dict[str, dict[str, object]]`
- `_confidence_badge(sample_size, t_stat) -> str`
- `_ev_plot_lines(expected, realized) -> list[str]`
- `_write_markdown(summary, window_days, output, *, freshness_ok=True, freshness_reasons=None, freshness_summary=None, slo_metrics=None, calibration_summary=None) -> None`
- `_slo_markdown_lines(entry, *, window_days) -> list[str]`
- `_load_structure_artifact(series) -> dict[str, object] | None`

## src/kalshi_alpha/exec/scoreboard_index_paper.py
Module doc: Scoreboard for index paper (dry) ledger entries.
Classes:
- `LedgerEntry()`
Functions:
- `_resolve_ledger_path(path_arg) -> Path`
- `_default_date_range(days=7) -> tuple[date, date]`
- `_parse_timestamp(value) -> datetime | None`
- `_parse_record(payload) -> LedgerEntry | None`
- `load_entries(ledger_path, *, start_date, end_date) -> list[LedgerEntry]`
- `aggregate(entries) -> list[dict[str, object]]`
- `render_markdown(summaries, *, start_date, end_date) -> str`
- `_build_output_path(output_arg, end_date) -> Path`
- `main(argv=None) -> int`

## src/kalshi_alpha/exec/slo.py
Module doc: Service level objective (SLO) aggregations for scoreboard + telemetry exports.
Classes:
- `SLOSeriesMetrics()` — Container for per-series SLO measurements.
Functions:
- `collect_metrics(summary, *, series=None, reports_root=Path('reports'), raw_root=Path('data/raw'), lookback_days=DEFAULT_LOOKBACK_DAYS, now=None, var_limits=None) -> dict[str, SLOSeriesMetrics]` — Build a SLO metrics map keyed by series.
- `publish_cloudwatch(metrics, *, namespace='KalshiSys/SLO', region=None, profile=None) -> None` — Push metrics to CloudWatch (best-effort).
- `parse_args(argv=None) -> argparse.Namespace`
- `main(argv=None) -> int`
- `_ensure_utc(moment) -> datetime`
- `_safe_float(value) -> float | None`
- `_fill_gap_percentage(value) -> float | None`
- `_populate_freshness(metrics, *, raw_root, lookback_days, now) -> None`
- `_populate_time_at_risk(metrics, *, reports_root, lookback_days, now) -> None`
- `_populate_var_headroom(metrics, *, reports_root, lookback_days, now, var_limits) -> None`
- `_percentile(values, percentile) -> float`
- `_iter_snapshot_latencies(*, series, raw_root, lookback_days, now, max_samples=5000) -> Iterable[float]`
- `_snapshot_latency_ms(path) -> float | None`
- `_collect_report_values(*, series, reports_root, threshold, pattern) -> Iterable[float]`
- `_series_report_paths(*, series, reports_root) -> Iterable[Path]`
- `_parse_report_date(path) -> date | None`
- `_extract_numeric(contents, pattern) -> Iterable[float]`
- `_cloudwatch_datums(metrics) -> Iterable[dict[str, object]]`
- `_chunk(sequence, size) -> Iterable[list[dict[str, object]]]`

## src/kalshi_alpha/exec/state/orders.py
Module doc: Persistence utilities for tracking outstanding broker orders.
Classes:
- `OutstandingOrderRecord()` — Serializable record of an outstanding order submission.
- `OutstandingOrdersState()` — Persisted state container for outstanding broker orders.
Functions:
- `_default_state_path() -> Path`

## src/kalshi_alpha/exec/supervisor.py
Module doc: 24/7 supervisor daemon that orchestrates live index ladder scans.
Classes:
- `SupervisorConfig()`
- `Supervisor()`
Functions:
- `_default_kill_switch_path() -> Path`
- `_parse_series(value, *, default) -> tuple[str, ...]`
- `main(argv=None) -> None`

## src/kalshi_alpha/exec/supervisor_index.py
Module doc: Supervisor for index ladder windows with preflight and WS freshness gating.
Classes:
- `SupervisorIndexConfig()`
- `WSListener()` — Lightweight index websocket listener feeding a freshness sentry.
Functions:
- `_log(message, *, quiet=False) -> None`
- `_write_heartbeat(config, *, now_utc, now_et, ws_age_ms) -> None`
- `_emit_preflight_summary(result, *, config) -> None`
- `_pick_window(now_et) -> TradingWindow | None` — Return the active window or the next upcoming window.
- `_series_to_run(window, *, series_filter=None) -> tuple[str, ...]`
- `_default_runner(series, window, config, now_et) -> None`
- `_run_window(window, *, now_et, config, preflight_fn, ws_listener, runner, preflight_observer=None, preflight_override=None) -> tuple[bool, bool]` — Return (ran, terminal). terminal=True marks window complete.
- `_run_once(config, *, preflight_fn, ws_factory, runner, preflight_observer=None) -> None`
- `_run_loop(config, *, preflight_fn, ws_factory, runner, preflight_observer=None) -> None`
- `_parse_now(value) -> datetime | None`
- `_is_transient_preflight(reasons) -> bool`
- `_build_parser() -> argparse.ArgumentParser`
- `_parse_args(argv=None) -> argparse.Namespace`
- `_build_config(args) -> SupervisorIndexConfig`
- `main(argv=None) -> None`

## src/kalshi_alpha/exec/telemetry/__init__.py
Module doc: Telemetry utilities for execution flows.
No top-level functions or classes.

## src/kalshi_alpha/exec/telemetry/run_metadata.py
Module doc: Telemetry run metadata writer for bounded TOB/quote-intent runs.
Functions:
- `write_telemetry_run_metadata(*, run_id, output_dir, status, broker, telemetry_only, series=None, window=None, preflight=None, tob_path=None, quote_intents_path=None, max_bytes_per_window=DEFAULT_TOB_WINDOW_MAX_BYTES, max_tob_bytes=DEFAULT_TOB_MAX_BYTES, max_intent_bytes=DEFAULT_INTENT_MAX_BYTES) -> Path` — Write a run metadata JSON describing the telemetry capture context.

## src/kalshi_alpha/exec/telemetry/shipper.py
Module doc: Utility to bundle telemetry JSONL into artifacts for shipping.
Functions:
- `parse_args(argv=None) -> argparse.Namespace`
- `main(argv=None) -> int`
- `_resolve_day(day, now) -> date`
- `_telemetry_path(root, day) -> Path`

## src/kalshi_alpha/exec/telemetry/sink.py
Module doc: Append-only telemetry sink for live execution events.
Classes:
- `TelemetryEvent()` — Serializable telemetry event payload.
- `TelemetrySink()` — Durable JSONL telemetry sink with daily rotation.
- `TelemetryJsonlSink()` — Append-only JSONL telemetry sink with run-based rotation + bounds.
Functions:
- `sanitize_book_snapshot(snapshot, *, depth=DEFAULT_BOOK_DEPTH) -> dict[str, Any] | list[dict[str, Any]] | None` — Return a depth-limited, JSON-serialisable book snapshot.
- `_sanitize_book_levels(levels, *, depth) -> list[dict[str, Any]]`
- `_safe_float(value) -> float | None`
- `_utc_now() -> datetime`
- `_ensure_path(base, moment) -> Path`
- `_mask(value) -> str | None`
- `_sanitize(mapping) -> dict[str, Any]`
- `_coerce(value) -> object`

## src/kalshi_alpha/exec/window_guard.py
Module doc: Shared helpers for gating index runners to ET trading windows.
Functions:
- `parse_now_override(now_text) -> datetime` — Return a timezone-aware reference timestamp.
- `guard_series_window(series, *, now=None, quiet=False) -> tuple[bool, TradingWindow | None, TradingWindow | None]` — Check whether *series* is inside an active trading window.

## src/kalshi_alpha/markets/__init__.py
Module doc: Market utilities for Kalshi index ladders.
No top-level functions or classes.

## src/kalshi_alpha/markets/discovery.py
Module doc: Market discovery utilities for INX/NDX ladders.
Classes:
- `DiscoveredMarket()`
- `WindowDiscovery()`
Functions:
- `discover_markets_for_day(client, *, trading_day, series=None, status='open') -> list[WindowDiscovery]` — Return discovered markets for the provided trading day grouped by scheduler windows.
- `_normalize_series(series) -> tuple[str, ...]`
- `_collect_markets(client, *, trading_day, series_list, status) -> dict[str, list[DiscoveredMarket]]`
- `_group_by_event(markets, trading_day) -> list[DiscoveredMarket]`
- `_match_for_window(candidates, window, used_events) -> DiscoveredMarket | None`
- `_orphan_markets(all_series_markets, used_events) -> list[DiscoveredMarket]`
- `_derived_windows(markets) -> list[WindowDiscovery]`
- `_infer_close_from_ticker(ticker) -> datetime | None`

## src/kalshi_alpha/models/__init__.py
Module doc: Model primitives for kalshi-alpha.
No top-level functions or classes.

## src/kalshi_alpha/models/pmf_index.py
Module doc: Index PMF utilities with σ_tod curves and optional EOD variance bumps.
Classes:
- `EODBump()`
- `IndexPMFParameters()`
- `IndexPMFModel()` — Produce ladder PMFs from stored σ_tod + drift curves.
Functions:
- `load_model(series, target_type, target_label, *, root=None) -> IndexPMFModel`
- `load_parameters(series, target_type, target_label, *, root=None) -> IndexPMFParameters`
- `available_targets(series, target_type, *, root=None) -> list[str]`
- `_params_path(series, target_type, target_label, *, root) -> Path`
- `_series_root(series, target_type, *, root) -> Path`
- `_decode_curve(payload, *, component) -> dict[int, float]`
- `_nearest(curve, minutes) -> float`

## src/kalshi_alpha/replay/fill_model.py
Module doc: Convert TOB snapshots into conservative fill probability curves.
Functions:
- `parse_args(argv=None) -> argparse.Namespace`
- `main(argv=None) -> None`
- `_resolve_snapshot_files(explicit, directory) -> list[Path]`
- `_iter_snapshots(files) -> Iterable[dict[str, object]]`
- `_build_summary(entries, *, reference_contracts) -> dict[str, object]`

## src/kalshi_alpha/replay/polygon_index_replay.py
Module doc: Replay recorded Polygon websocket aggregates into the index pipeline artifacts.
Classes:
- `ReplayRecord()`
- `ReplayConfig()`
Functions:
- `parse_args(argv=None) -> argparse.Namespace`
- `main(argv=None) -> None`
- `_run_replay(config) -> None`
- `_replay_step(*, payload, timestamp, config, tracker) -> None`
- `_load_records(path) -> list[ReplayRecord]`
- `_filter_window(records, start_time, end_time) -> list[ReplayRecord]`
- `_parse_clock(value) -> time`
- `_parse_timestamp(value) -> datetime | None`
- `_ensure_message_dict(raw) -> dict[str, object]`
- `_json_loads(value) -> dict[str, object]`
- `_in_window(clock, start, end) -> bool`
- `_override_data_root(data_root) -> None`

## src/kalshi_alpha/risk/__init__.py
Module doc: Risk helpers for kalshi-alpha.
No top-level functions or classes.

## src/kalshi_alpha/risk/correlation.py
Module doc: Correlation-aware VaR limiter with inventory tilt support for index ladders.
Classes:
- `ProbabilitySurface()` — Accumulates strike → survival probability mappings per series.
- `CorrelationConfig()`
- `InventoryExposure()`
- `CorrelationAwareLimiter()` — Applies correlation-aware VaR caps and inventory tilt to ladder proposals.
Functions:
- `_normalize_series(value) -> str`
- `_strike_key(strike) -> str`

## src/kalshi_alpha/risk/var_index.py
Module doc: Simple per-family VaR limiter for index ladders.
Classes:
- `FamilyVarLimiter()`
Functions:
- `load_family_limits(path=None) -> Mapping[str, float]`

## src/kalshi_alpha/sched/__init__.py
Module doc: Scheduler helpers for hourly/EOD ladders plus regime-aware flags.
No top-level functions or classes.

## src/kalshi_alpha/sched/hotrestart.py
Module doc: Hot-restart snapshot utilities for maker ops.
Classes:
- `HotRestartSnapshot()`
- `HotRestartManager()` — Persist/restore scheduler state to enable <5s crash recovery.
Functions:
- `summarize_orders_state(state) -> dict[str, int]` — Convert OutstandingOrdersState into a compact summary.
- `_encode_window(window) -> dict[str, object] | None`

## src/kalshi_alpha/sched/regimes.py
Module doc: Trading-day regime flags for macro events (FOMC, CPI) with SLO overrides.
Classes:
- `RegimeFlags()`
Functions:
- `regime_for(moment=None, *, calendar_path=None) -> RegimeFlags`
- `_load_calendar(path) -> pl.DataFrame`

## src/kalshi_alpha/sched/windows.py
Module doc: US/Eastern-aware scheduler for hourly and close index ladder windows.
Classes:
- `TradingWindow()` — Resolved execution window with start/target/freeze metadata.
Functions:
- `_ops_config() -> tuple[IndexOpsWindow, IndexOpsWindow]`
- `windows_for_day(trading_day) -> list[TradingWindow]` — Return all ladder windows for the provided trading day.
- `current_window(series, moment=None) -> TradingWindow | None` — Return the active window for *series* at *moment*, if any.
- `next_window_for_series(series, moment=None) -> TradingWindow | None` — Return the next upcoming window for *series* after *moment* (default: now).
- `next_windows(now=None, *, limit=4) -> list[TradingWindow]` — Return the next *limit* upcoming windows from *now*.
- `_build_window(*, trading_day, target_hour, label, window, target_type) -> TradingWindow`
- `_resolve_window_start(window, target_et) -> datetime`
- `_ensure_et(moment) -> datetime`

## src/kalshi_alpha/strategies/__init__.py
Module doc: Strategy modules producing ladder probability distributions.
No top-level functions or classes.

## src/kalshi_alpha/strategies/base.py
Module doc: Shared helpers for strategy distributions.
Functions:
- `ladder_bins(strikes) -> list[tuple[float | None, float | None]]`
- `normalize(weights) -> list[float]`
- `gaussian_weight(mean, std, lower, upper) -> float`
- `_gaussian_cdf(x, mean, std) -> float`
- `pmf_from_gaussian(strikes, mean, std) -> list[LadderBinProbability]`
- `grid_distribution_to_pmf(distribution) -> list[LadderBinProbability]`

## src/kalshi_alpha/strategies/claims/__init__.py
Module doc: Initial jobless claims strategy with calibration utilities.
Classes:
- `ClaimsInputs()`
- `ClaimsRegressorWeights()`
Functions:
- `freeze_window(active_at=None) -> bool` — Return True once the Wednesday data freeze has begun (UTC).
- `_to_float(value, default) -> float`
- `_resolve_regressor_weights(weights) -> ClaimsRegressorWeights`
- `_regression_adjustment(inputs, weights) -> float`
- `_kalman_freeze_update(prior_mean, prior_variance, measurement, measurement_variance) -> tuple[float, float]`
- `pmf(strikes, inputs=None, *, calibration=None) -> list[LadderBinProbability]` — Return a distribution over initial claims ladder strikes.
- `pmf_v15(strikes, inputs=None, *, calibration=None, weights=None, freeze_measurement_noise=FREEZE_MEASUREMENT_NOISE) -> list[LadderBinProbability]` — Enhanced claims distribution using optional regressors and Kalman blending.
- `calibrate(history, window=12, strikes=None) -> pl.DataFrame` — Calibrate claims distribution parameters and persist evaluation metrics.
- `_damped_trend_forecast(history, *, holiday_adjust) -> float`
- `_claims_std(history) -> float`
- `_mean_claims(inputs) -> float`
- `_continuing_claims_adjustment(values) -> float`
- `_gt_topic_adjustment(score) -> float`
- `_params_from_history(history) -> dict[str, float]`
- `_brier_score(pmf_values, actual) -> float`
- `_bin_indicator(bin_prob, value) -> float`
- `_load_calibration() -> dict[str, float] | None`

## src/kalshi_alpha/strategies/cpi/__init__.py
Module doc: CPI strategy producing monthly and YoY distributions.
Classes:
- `CPIV15Config()`
- `CPIInputs()`
Functions:
- `_safe_float(value, fallback) -> float`
- `_load_v15_config(config_path=None) -> CPIV15Config`
- `load_v15_config(config_path=None) -> CPIV15Config` — Expose cached v1.5 configuration for downstream consumers.
- `nowcast(inputs=None, *, calibration=None) -> dict[float, float]` — Return a normalized distribution over 0.05pp grid points.
- `nowcast_v15(inputs=None, *, fixtures_dir=None, weights=None, component_overrides=None, calibration=None, offline=True, config_path=None) -> dict[float, float]`
- `map_to_ladder_bins(strikes, distribution) -> list[LadderBinProbability]`
- `calibrate(history, window=12) -> pl.DataFrame` — Calibrate CPI nowcast bias/std and persist evaluation metrics to parquet.
- `_belongs_to_bin(point, lower, upper) -> bool`
- `_select_mean(inputs) -> float`
- `_grid_around(center) -> list[float]`
- `_gaussian_weight(point, mean, std) -> float`
- `_params_from_history(history) -> dict[str, float]`
- `_distribution_stats(distribution) -> tuple[float, float]`
- `_yoy_projection(entry, predicted_mom, actual_mom) -> tuple[float | None, float | None]`
- `_load_calibration() -> dict[str, float] | None`

## src/kalshi_alpha/strategies/cpi/components.py
Module doc: Component signals used in CPI v1.5 nowcasting.
Classes:
- `ComponentWeights()`
Functions:
- `component_signals(*, fixtures_dir=None, offline=True) -> dict[str, float]` — Return lightweight component signals used by CPI v1.5.
- `_resolve_candidates(base_dir, filename) -> list[Path]`
- `_gas_mtd_signal(base_dir) -> float | None`
- `_shelter_lag_signal(base_dir) -> float | None`
- `_used_car_signal(base_dir) -> float | None`
- `blend_component_shift(signals, weights) -> float`

## src/kalshi_alpha/strategies/gas/__init__.py
Module doc: Placeholder for future gasoline ladder strategies.
No top-level functions or classes.

## src/kalshi_alpha/strategies/index/__init__.py
Module doc: Index ladder strategies powered by Polygon data.
No top-level functions or classes.

## src/kalshi_alpha/strategies/index/backtest_polygon.py
Module doc: Minimal Polygon-only backtest harness for index ladders with Kalshi quotes.
Classes:
- `BacktestConfig()`
- `TradeResult()`
Functions:
- `_strike_grid(price, symbol, half_width=6) -> list[float]`
- `_target_price(frame, target_dt) -> float | None`
- `_now_snapshot(frame, target_dt) -> tuple[datetime, float] | None`
- `_minutes_to_target(now_ts, target_dt) -> float`
- `_simulate_day(frame_day, *, series, symbol, horizon, params, config) -> list[TradeResult]`
- `_horizons_for_series(series, override) -> list[str]`
- `run_backtest(panel, config) -> list[TradeResult]`
- `summarize(trades) -> dict[str, float]`
- `write_trades_csv(trades, path) -> None`
- `write_report(summary, path) -> None`

## src/kalshi_alpha/strategies/index/cdf.py
Module doc: CDF helpers shared by index strategies.
Classes:
- `LateDayVariance()`
- `EventTail()`
- `SigmaCalibration()`
Functions:
- `gaussian_pmf(strikes, *, mean, std, min_std=1.0) -> list[LadderBinProbability]`
- `survival_map(strikes, pmf) -> dict[float, float]`
- `probability_at_or_above(strike, strikes, pmf) -> float`
- `probability_between(lower, upper, strikes, pmf) -> float`
- `load_calibration(path, symbol, *, horizon, variant=None) -> SigmaCalibration`
- `_extract_minutes_curves(data, file_path) -> tuple[dict[int, float], dict[int, float]]`
- `_extract_pit_bias(data) -> float | None`
- `_extract_late_day_variance(data) -> LateDayVariance | None`
- `_extract_event_tail(data) -> EventTail | None`
- `_extract_optional_curve(data, key) -> dict[int, float] | None`
- `_load_calibration_cached(file_path) -> SigmaCalibration`
- `_extract_sigma_now(data) -> float | None`
- `_extract_metadata(data) -> Mapping[str, Any] | None`
- `_resolve_calibration_file(path, symbol, horizon, *, variant) -> Path`
- `_symbol_slug(symbol) -> str`
- `_nearest(curve, minutes) -> float`

## src/kalshi_alpha/strategies/index/close_range.py
Module doc: End-of-day close range strategy for index ladders.
Classes:
- `CloseInputs()`
Functions:
- `pmf(strikes, inputs, *, calibration=None) -> list[LadderBinProbability]`
- `_resolve_series(series) -> IndexSymbol`
- `_load_default_calibration(meta) -> SigmaCalibration`
- `_late_day_variance(inputs, calibration, minutes_to_target) -> float`
- `_event_multiplier(inputs, calibration) -> float`

## src/kalshi_alpha/strategies/index/fill_model.py
Module doc: Lightweight maker fill probability heuristic for index ladders.
Functions:
- `estimate_maker_fill_prob(distance_to_mid_cents, time_to_expiry_minutes, spread_cents) -> float` — Return a conservative fill probability for a maker quote.

## src/kalshi_alpha/strategies/index/hourly_above_below.py
Module doc: Intraday hourly above/below strategy for index ladders.
Classes:
- `HourlyInputs()`
Functions:
- `pmf(strikes, inputs, *, calibration=None) -> list[LadderBinProbability]`
- `_resolve_series(series) -> IndexSymbol`
- `_load_hourly_calibration(meta, variant) -> SigmaCalibration`
- `_event_multiplier(inputs, calibration) -> float`
- `_target_hour_variant(target_hour) -> str | None`
- `_skewnorm_pmf(strikes, *, mean, std, skew, min_std) -> list[LadderBinProbability]`

## src/kalshi_alpha/strategies/index/model_polygon.py
Module doc: Simple Polygon-only distribution model for SPX/NDX ladders.
Classes:
- `ModelParams()`
Functions:
- `_ensure_timestamp(series) -> pd.Series`
- `_target_price_map(df, target_clock) -> dict[pd.Timestamp, float]` — Return a per-day target price map (ET) using last price <= target.
- `_normalized_returns(df, *, target_clock) -> tuple[np.ndarray, list[float]]`
- `fit_from_panel(panel, horizon, *, symbols=None) -> dict[str, ModelParams]` — Fit per-symbol Student-t parameters from a Polygon index panel.
- `predict_pmf(now_state, params, ladder_strikes) -> list[LadderBinProbability]` — Generate a ladder PMF from fitted parameters.
- `save_params(params, path) -> Path` — Persist parameter dict to JSON.
- `load_params(path) -> dict[str, object]`
- `params_path(series, horizon, root=None) -> Path` — Return default params.json path for a given series/horizon pair.

## src/kalshi_alpha/strategies/index/noon_above_below.py
Module doc: Backward-compatible shim for hourly index above/below strategy.
No top-level functions or classes.

## src/kalshi_alpha/strategies/teny/__init__.py
Module doc: 10-year Treasury yield strategy with factor calibration.
Classes:
- `TenYInputs()`
Functions:
- `pmf(strikes, inputs=None, *, calibration=None) -> list[LadderBinProbability]`
- `pmf_v15(strikes, inputs=None, *, calibration=None, dummy_weights=None, imbalance_threshold=IMBALANCE_THRESHOLD, imbalance_multiplier=IMBALANCE_MULTIPLIER) -> list[LadderBinProbability]`
- `_shock_dummy_adjustment(dummies, weights) -> float`
- `_apply_imbalance_spread(spread, inputs, threshold, multiplier) -> float`
- `_extract_event_time(value) -> time | None`
- `_sample_std(values) -> float`
- `_slope_factor(inputs) -> float`
- `calibrate(history) -> pl.DataFrame`
- `_ols_betas(macro, slope_terms, deltas) -> tuple[float, float]`
- `_load_calibration() -> dict[str, float] | None`
- `_macro_dummies_lookup(history) -> tuple[dict[str, dict[str, float]], list[str]]`
- `_normalize_history_date(value) -> str | None`
- `_strip_dummy_prefix(column) -> str`

## src/kalshi_alpha/strategies/weather/__init__.py
Module doc: Weather strategy stubs enforcing NOAA/NWS DCR settlement requirements.
Classes:
- `EnsembleMember()`
- `WeatherInputs()`
Functions:
- `_to_float(value) -> float`
- `pmf(strikes, ensemble=None, inputs=None, *, calibration=None) -> list[LadderBinProbability]`
- `settlement_reminder(station_id) -> str`
- `calibrate(history) -> pl.DataFrame`
- `_load_calibration() -> WeatherCalibration | None`

## src/kalshi_alpha/structures/__init__.py
Module doc: Structure-level utilities (allocators, range builders, hedges).
No top-level functions or classes.

## src/kalshi_alpha/structures/allocator.py
Module doc: Capital allocator for INX/NDX structures with correlation-aware VaR guardrails.
Classes:
- `SeriesWindowSample()` — Single window observation used to calculate EV×fill×honesty Sharpe.
- `RollingStats()`
- `RollingSharpeWindow()` — Maintains a rolling EV×fill×honesty Sharpe estimate per series.
- `VarSnapshot()` — Simplified headroom view derived from the correlation-aware VaR guard.
- `SeriesBudget()`
- `AllocationResult()`
- `AllocatorConfig()`
- `Allocator()` — Calculates per-series capital budgets using EV×fill×honesty Sharpe.
Functions:
- `_series_key(series) -> str`
- `_clamp01(value) -> float`
- `load_scoreboard_history(path) -> dict[str, list[SeriesWindowSample]]` — Load allocator samples from a scoreboard summary JSON payload.
- `correlation_var_snapshot(config_path=None) -> VarSnapshot` — Convenience helper returning the baseline VaR snapshot from config only.

## src/kalshi_alpha/structures/range_ab.py
Module doc: Construct hedged Range↔AB structures from adjacent strikes.
Classes:
- `StructureLeg()`
- `RangeABStructure()`
Functions:
- `_structure_id(market_id, lower, upper) -> str`
- `build_range_structures(*, series, market_id, market_ticker, rungs, strategy_survival, contracts, schedule_series=None) -> list[RangeABStructure]` — Return Range↔AB structures formed by adjacent strikes.

## src/kalshi_alpha/utils/env.py
Module doc: Environment loading utilities.
Functions:
- `load_env() -> None` — Load environment variables from .env files once.

## src/kalshi_alpha/utils/family.py
Module doc: Family helpers for focusing execution on index ladders.
Functions:
- `resolve_family(value=None) -> str` — Resolve the requested family from an explicit value or the FAMILY env var.
- `filter_index_series(series) -> list[str]` — Return only index ladder tickers from *series*.
- `is_index_family(value) -> bool` — Shortcut to test whether *value* resolves to the index family.

## src/kalshi_alpha/utils/http.py
Module doc: HTTP utilities with caching support.
Classes:
- `HTTPError(RuntimeError)` — Raised when an HTTP request fails.
Functions:
- `fetch_with_cache(url, cache_path, *, session=None, force_refresh=False, timeout=15.0, headers=None) -> bytes` — Fetch a URL with basic ETag/Last-Modified caching.

## src/kalshi_alpha/utils/keys.py
Module doc: Secure secret loaders with macOS Keychain support.
Functions:
- `_is_macos() -> bool`
- `_run_security(args) -> subprocess.CompletedProcess[str]`
- `_keychain_lookup(label) -> str | None`
- `load_secret(*, keychain_label, env_var, strip=True) -> str | None`
- `load_polygon_api_key() -> str | None`

## src/kalshi_alpha/utils/secrets.py
Module doc: Utilities for detecting and redacting sensitive strings.
Functions:
- `redacted(text) -> str`
- `ensure_safe_payload(payload) -> None` — Raise ValueError if payload contains forbidden tokens.
- `_contains_token(obj, token) -> bool`

## tools/__init__.py
Module doc: Utility scripts for kalshi-sys tooling.
No top-level functions or classes.

## tools/agentic/gpt_bundle.py
Module doc: gpt_bundle.py
Functions:
- `run(cmd, cwd=None) -> Tuple[int, str]`
- `git_root(start) -> Optional[Path]`
- `ensure_repo_snapshot(repo) -> Optional[Path]`
- `list_changed_files(repo) -> list[str]`
- `add_file_if_small(z, repo, rel_path, max_bytes=120000) -> None`
- `main() -> int`

## tools/agentic/project_state_refresh.py
Module doc: project_state_refresh.py
Functions:
- `run(cmd, cwd=None) -> Tuple[int, str]`
- `git_root(start) -> Optional[Path]`
- `ensure_templates(project_state_dir) -> None`
- `write_generated(repo, project_state_dir) -> None`
- `zip_project_state(repo, project_state_dir, out_zip) -> Path`
- `main() -> int`

## tools/agentic/repo_snapshot.py
Module doc: Create a deterministic, lightweight repo snapshot for GPT review.
Functions:
- `run(cmd, cwd=None) -> Tuple[int, str]`
- `git_root(start) -> Optional[Path]`
- `git_ls_files(repo) -> list[str]`
- `guess_language_counts(paths) -> dict[str, int]`
- `main() -> int`

## tools/build_fillcalib_dataset.py
Module doc: Build fill calibration datasets + conservative maker fill curves from telemetry.
Classes:
- `TobSnapshot()`
- `QuoteIntent()`
- `BucketKey()`
Functions:
- `parse_args(argv=None) -> argparse.Namespace`
- `build_fillcalib_dataset(*, series, date_from, date_to, telemetry_root, horizon_seconds=DEFAULT_HORIZON_SECONDS, min_samples=DEFAULT_MIN_SAMPLES, scaler=DEFAULT_SCALER, max_fill=DEFAULT_MAX_FILL) -> tuple[dict[str, object], list[dict[str, object]]]`
- `write_outputs(*, payload, rows, series, date_to, output_curves, output_report, output_parquet, write_parquet, max_parquet_rows) -> dict[str, Path | None]`
- `_render_report(payload, rows, *, series) -> str`
- `_load_tob_snapshots(*, telemetry_root, series, date_from, date_to) -> tuple[dict[str, list[TobSnapshot]], int]`
- `_load_quote_intents(*, telemetry_root, series, date_from, date_to) -> list[QuoteIntent]`
- `_build_tob_index(tob_snapshots) -> dict[str, tuple[list[datetime], list[TobSnapshot]]]`
- `_proxy_fill(intent, times, snapshots, horizon_seconds) -> int`
- `_quote_distance_to_touch(intent, times, snapshots) -> float | None`
- `_time_to_expiry(intent) -> tuple[float | None, str]`
- `_bin_distance(distance) -> str`
- `_bin_time(minutes) -> str`
- `_iter_paths(root) -> Iterable[Path]`
- `_iter_json_lines(path) -> Iterator[dict[str, object]]`
- `_record_type_ok(record, expected) -> bool`
- `_parse_ts(value) -> datetime | None`
- `_safe_float(value) -> float | None`
- `_within_range(ts, date_from, date_to) -> bool`
- `main(argv=None) -> None`

## tools/failover_smoke.py
Module doc: CLI smoke test for DualFeedFailover (synthetic timeline).
Functions:
- `_parse_args(argv=None) -> argparse.Namespace`
- `_synthetic_feed(name, *, interval_ms, outage_start, outage_duration_ms, duration_seconds) -> list[FeedSample[int]]`
- `_merge_samples(primary, secondary) -> Iterator[FeedSample[int]]`
- `main(argv=None) -> None`

## tools/gpt_bundle_builder.py
Module doc: Stage GPT bundle contents with fail-closed artifact checks.
Classes:
- `BundleBuildError(RuntimeError)`
Functions:
- `_copy_file(src, dest, staging_root, staged_files) -> None`
- `_copy_tree(src_dir, dest_dir, staging_root, staged_files) -> None`
- `_require_path(path, label) -> None`
- `_missing_artifacts_from_stage(artifacts_content, staged_files, workspace_root) -> list[str]`
- `stage_bundle(workspace_root, run_name, staging_root) -> set[str]`
- `_parse_args(argv=None) -> argparse.Namespace`
- `main(argv=None) -> int`

## tools/project_state_build.py
Module doc: Generate project_state/_generated artifacts (inventory, symbols, imports, make targets).
Classes:
- `PySymbol()`
Functions:
- `utc_now_iso() -> str`
- `should_skip_dir(rel) -> bool`
- `iter_files(root) -> Iterable[Path]`
- `role_for_path(rel) -> str`
- `build_repo_inventory(root) -> Dict`
- `read_text(path) -> str`
- `format_default(node) -> str`
- `format_args(args) -> str`
- `format_signature(node) -> str`
- `first_line(text) -> str`
- `module_name_for_path(path, root) -> Optional[str]`
- `package_for_module(module, path) -> Optional[str]`
- `collect_python_files(root) -> List[Path]`
- `parse_symbols(root) -> Tuple[Dict, Dict[str, Set[str]]]`
- `parse_make_targets(makefile) -> List[str]`
- `main() -> int`

## tools/replay.py
Module doc: Replay recorded Kalshi sessions to validate EV parity.
Classes:
- `ReplayRun()`
Functions:
- `parse_args(argv=None) -> argparse.Namespace`
- `main(argv=None) -> int`
- `_resolve_date(label) -> date`
- `_parse_families(raw) -> set[str]`
- `_parse_hours(raw) -> set[int]`
- `_discover_runs(*, target_date, families, hours, raw_root) -> list[ReplayRun]`
- `_market_hour(path) -> int | None`
- `_timestamp_from_dir(name, base_date) -> datetime | None`
- `_replay_single(run) -> pl.DataFrame | None`
- `_summarize(frame, target_date, *, epsilon) -> dict[str, object]`
- `_write_plot(windows, epsilon, output) -> None`
- `_read_json(path) -> dict | list | None`
- `_parse_timestamp(value) -> datetime | None`

## tools/settlement_basis_audit.py
Module doc: Settlement basis audit for index ladder windows (Polygon vs Kalshi expiration value).
Classes:
- `PolygonWindowValue()`
- `KalshiSettlementValue()`
- `_KalshiAuthenticatedClient()` — Adapter for Kalshi trade-api/v2 using the KalshiHttpClient interface.
Functions:
- `_kalshi_series_ticker(series) -> str | None`
- `_close_time_from_ticker(ticker) -> datetime | None`
- `_normalize_market_payload(payload, *, series_ticker) -> dict[str, Any]`
- `_parse_day(value) -> date`
- `_parse_args(argv=None) -> argparse.Namespace`
- `_resolve_paths(day, series, *, out_report, out_data, out_json) -> tuple[Path, Path, Path]`
- `_parse_iso(value) -> datetime | None`
- `_safe_float(value) -> float | None`
- `_load_polygon_offline_values(fixtures_root, *, day, series) -> dict[str, PolygonWindowValue]`
- `_polygon_values_online(series, windows) -> dict[str, PolygonWindowValue]`
- `_find_field(payload, key, path='') -> tuple[float, str] | None`
- `_extract_kalshi_value(payload) -> KalshiSettlementValue | None`
- `_collect_strikes(client, event_id) -> list[float]`
- `_nearest_strike(value, strikes) -> tuple[float | None, float | None]`
- `_median_spacing(strikes) -> float | None`
- `_load_frame(path) -> pl.DataFrame`
- `_write_frame(frame, path) -> None`
- `_git_sha() -> str`
- `_format_float(value) -> str`
- `_series_stats(series) -> dict[str, float | None]`
- `_quantiles(series, quantiles) -> dict[str, float | None]`
- `_normalize_float(value) -> float | None`
- `_load_quote_distance(series) -> tuple[float | None, str | None]`
- `_build_per_window_deltas(frame) -> list[dict[str, object]]`
- `_compute_flip_risk(*, series, basis_series, strike_spacings) -> dict[str, object]`
- `_build_summary(frame, *, day, series) -> dict[str, object]`
- `_render_report(frame, *, day, series, dataset_path, command, summary) -> str`
- `_windows_for_series(day, series) -> list[TradingWindow]`
- `_discover_markets(client, *, day, series) -> tuple[dict[str, DiscoveredMarket], dict[str, WindowDiscovery]]`
- `_discover_markets_authenticated(client, *, day, series) -> tuple[dict[str, DiscoveredMarket], dict[str, WindowDiscovery]]`
- `_build_dataset(*, day, series, offline_fixtures) -> pl.DataFrame`
- `main(argv=None) -> None`

## tools/sync_vendor_docs.py
Classes:
- `Source()`
Functions:
- `sha256_bytes(b) -> str`
- `main() -> None`

## tools/verify_gpt_bundle.py
Module doc: Verify GPT bundle completeness and diff hygiene.
Classes:
- `BundleVerificationError(Exception)`
Functions:
- `_read_text(zf, name) -> str`
- `_find_run_names(paths) -> set[str]`
- `_placeholder_lines(content) -> list[str]`
- `_is_empty_or_pending(content) -> bool`
- `_has_patch_hunks(content) -> bool`
- `_extract_artifact_paths(content) -> list[str]`
- `_missing_artifacts_from_bundle(artifacts_content, bundle_paths, bundle_root_prefix, workspace_root) -> list[str]`
- `verify_bundle(bundle_path) -> list[str]`
- `_parse_args(argv=None) -> argparse.Namespace`
- `main(argv=None) -> int`
