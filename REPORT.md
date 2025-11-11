# Execution Snapshot — 2025-11-10

- **Fees:** added `configs/fees.json` (versioned indices taker coefficients + rounding metadata) and a dedicated loader under `kalshi_alpha.exec.fees`. Scanner metadata now reports the config path and EV components use true per-order rounding. Unit coverage: `tests/test_exec_fees.py` (1/2/100/1000-lot goldens).
- **Discovery:** fresh `python -m kalshi_alpha.exec.runners.scan_ladders --discover` mode plus the new `kalshi_alpha.markets.discovery` helper ensure INX/NDX hourly + close ladders are listed before the scheduler arms. Covered via `tests/test_market_discovery.py` and `tests/test_scan_ladders_discover.py`.
- **PMF & calibration:** bridge lives in `kalshi_alpha.models.pmf_index` with new hourly/EOD jobs (`jobs/calib_hourly.py`, `jobs/calib_eod.py`) writing params + plots; CLI coverage uses the index fixtures in `tests/test_pmf_calib_jobs.py` / `tests/test_index_pmf_model.py`.
- **Honesty metrics:** added `python -m report.honesty` (reliability curves, Brier/ECE, clamps). Scoreboard prints the metrics, `scan_ladders` shrinks EV via the clamp, and `tests/test_report_honesty.py` guards the flow.
- **Fill realism:** TOB recorder (`python -m kalshi_alpha.exec.collectors.kalshi_tob`) plus fill-model builder feed `kalshi_alpha.core.execution.fillprob`; scan_ladders now downshifts fill alpha using the curve (`tests/test_kalshi_tob_recorder.py`, `tests/test_fill_model.py`, `tests/test_fillprob.py`).
- **Risk:** per-family VaR limiter (`kalshi_alpha.risk.var_index`) caps SPX/NDX exposure and emits telemetry; unit coverage in `tests/test_var_limiter.py`.
- **AWS jobs & parity:** added `scripts/aws_job.py`, Dockerfile, `make aws-calib`/`aws-replay`, and `scripts/parity_gate.py` (`make parity-ci`).
- **Scheduler & safety:** introduced `kalshi_alpha.sched.windows` (US/Eastern hourly + close windows). Scanner emits `scheduler_window` and auto issues `scheduler_t_minus_2s` cancel-all before each target. Maker-only is the default; taker paths require `--allow-taker`.
- **Freshness sentry:** `kalshi_alpha.data.ws_sentry.WSFreshnessSentry` powers the new final-minute guard. When `ws_freshness_age_ms > 700` inside the last minute, scanner enters `polygon_ws_final_minute_stale`, clears proposals, and marks cancel-all. Scoreboards now surface Polygon WS latency in the header (see `reports/scoreboard_7d.md`).
- **Docs & ops:** published `.env.local.example` plus per-window runbooks (`docs/runbooks/hourly.md`, `docs/runbooks/eod.md`) covering scheduler timings, freshness guard, kill-switch steps, and the new discovery/honesty/AWS shim flows. README updated to highlight the defaults.
- **Verification:** `pytest -q` (unit), plus manual `kalshi-scan --series INXU --offline` smoke to confirm fee paths and scheduler metadata. Scoreboard regeneration confirms fresh metrics rendered; kill-switch tests assert the guard triggers before quality gates.

## Runtime Monitors
<!-- monitors:start -->
_No runtime monitor run recorded._
<!-- monitors:end -->

## ANALYSIS
- NO-GO (go=0, no_go=1): CPI misses min fills (0<300), Δbps (-4718.34<+6), and t-stat (-44.32<2); ev_gap monitor now ALERT, while other gates stay green.
- n_fills=0 (sample_size=2 trades recorded) with mean Δbps after fees -4718.34, t-stat -16.75 in session log, CuSum state NO_DATA; ledger aggregates show 8 proposals with zero realized fills.
- Freshness: ledger_age_minutes=0.16 (data/proc/ledger_all.parquet rows=8) and monitors_age_minutes=0.08; freeze window clear with next CPI freeze opening 2025-11-11 06:00 ET.
- Monitors: ev_gap ALERT (mean Δbps -4718.34, t=-44.32); fill_vs_alpha / ev_seq_guard NO_DATA; auth_error, drawdown (daily +24.00 / weekly +150.33), ws_disconnect, kill_switch all OK.
- EV honesty bins: CPI-2025-10-MOM YES@0.18 and NO@0.47 remain Δ=0.00, no caps/flags applied.
- Connectivity: `make live-smoke` now succeeds against prod (balance + markets endpoints reachable with new API key).
- Latest bundle: reports/pilot_bundle_20251103_001419.tar.gz (includes readiness, monitors, ledger snapshot, telemetry).
