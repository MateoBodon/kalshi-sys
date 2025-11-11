# Changelog

# Changelog

## 2025-11-11
- Landed correlation-aware VaR caps and inventory tilt: `kalshi_alpha.risk.correlation` feeds `scan_ladders` with cross-bin/cross-index sizing adjustments, new config `configs/index_correlation.yaml`, and unit coverage in `tests/test_correlation_var.py`.
- Added dynamic quote optimization (`kalshi_alpha.exec.quote_optim`) wired into `scan_ladders` to apply PMF-skew/microprice penalties, freshness widening, and replacement throttles; covered via `tests/test_quote_optim.py`.
- Built the dual-feed failover controller (`kalshi_alpha.data.failover`) plus `python -m tools.failover_smoke --dry-run`; CI/unit coverage lives in `tests/test_failover.py`.
- Shipped `kalshi_alpha.sched.hotrestart.HotRestartManager` with hot-restart snapshots (`data/proc/state/hot_restart.json`) and regression tests (`tests/test_hotrestart.py`); runbooks reference the workflow.
- Tightened ΔEV parity CI: `scripts/parity_gate.py` now enforces per-window thresholds, emits `reports/_artifacts/monitors/ev_gap.json`, and is exercised by `tests/test_parity_gate.py`.
- Documentation refresh: hourly/EOD runbooks include hot-restart + dual-feed guidance, new `docs/runbooks/outage_playbook.md`, and a reusable [Post-Mortem Template](docs/runbooks/postmortem_template.md).
- Added market discovery plumbing (`kalshi_alpha.markets.discovery` + `scan_ladders --discover`) so ops can confirm INX/NDX ladders before arming the scheduler; fixtures/tests live under `tests/test_market_discovery.py` and `tests/test_scan_ladders_discover.py`.
- Landed the new PMF bridge (`kalshi_alpha.models.pmf_index`) plus hourly/EOD calibration jobs (`jobs/calib_hourly.py`, `jobs/calib_eod.py`), persistence paths, and plotting hooks — see `tests/test_pmf_calib_jobs.py` and `tests/test_index_pmf_model.py`.
- `python -m report.honesty` now computes reliability curves, Brier, and ECE; scoreboard consumes the artifact, scan_ladders applies per-series clamps, and telemetry exposes the shrink factors.
- Built the TOB recorder + fill-model pipeline (`kalshi_alpha.exec.collectors.kalshi_tob`, `kalshi_alpha.replay.fill_model`, `kalshi_alpha.core.execution.fillprob`) so fill alpha automatically downshifts off real depth snapshots.
- Enforced per-family VaR caps via `kalshi_alpha.risk.var_index` and surfaced exposure snapshots in scanner monitors.
- Shipped the AWS shim (`scripts/aws_job.py`, Dockerfile) with `make aws-calib` / `make aws-replay`, plus the ΔEV parity gate (`scripts/parity_gate.py`, `make parity-ci`).

## 2025-11-03
- Added `kalshi_alpha.exec.monitors.freshness` with configurable thresholds (`configs/freshness.yaml`) for CPI, Claims, TenY, Cleveland, AAA Gas, and NWS climate feeds; the monitor emits `reports/_artifacts/monitors/freshness.json` and a CLI table (`make freshness-smoke`).
- Ramp readiness now ingests the freshness artifact, surfaces a “Data Freshness” table in JSON/Markdown, and stamps `STALE_FEEDS` when any required feed is stale, missing, out-of-range (AAA), or misaligned (TenY series identity).
- `scan_ladders` short-circuits the pre-submit gate whenever `required_feeds_ok` is false, sharing the same artifact and reasons as the readiness report.
- Weather freshness is scoped to the active station list; stale stations are enumerated in readiness and block GO decisions.
- Documentation refreshed with the new data freshness workflow, and sample monitor output (`reports/_artifacts/monitors/freshness.json`) added for reference.

## 2025-11-02 (Sprint 7)
- `pilot_session.json` now records the target `family`, normalized `cusum_state`, fill realism gap, and full alert summary; tests cover payload structure and artifact writes.
- Pilot scans import `ev_honesty_bins` from `reports/pilot_ready.json` and enforce the recommended per-bin weights/caps before sizing orders, even when the series-level decision is GO.
- `pilot_readiness.md` renders a per-bin EV honesty table alongside the existing summary metrics, and `README_pilot.md` includes an explicit final GO/NO-GO decision with rationale.
- Bundle checklist and runbook guidance updated to highlight per-bin enforcement, freshness gates, and the richer session metadata.

## 2025-11-02 (Sprint 6)
- Introduced `python -m kalshi_alpha.exec.runners.pilot` as the single pilot entrypoint; it auto-enforces maker-only sizing, per-bin clamps from the pilot config, and records structured session metadata.
- Pilot runs now persist `reports/_artifacts/pilot_session.json` with trades, Δbps/t-stat, CuSum status, fill realism, and recent monitor alerts. Ramp readiness ingests the session file to surface per-bin EV honesty alongside optional manual caps/weights.
- `compute_ramp_policy` exposes `ledger_age_minutes` / `monitors_age_minutes` in `pilot_ready.json`, generates per-bin EV summaries (`ev_honesty_bins`), and carries per-bin overrides into the bundle README.
- `python -m kalshi_alpha.exec.pilot_bundle` now packages the session artifact and a generated `README_pilot.md` checklist covering EV honesty flags, CuSum, freeze violations, drawdown, WS/auth health, and freshness thresholds.
- Expanded mypy/ruff coverage to the new pilot modules and refreshed the pilot test suite (session JSON, bundle contents, staleness gates, per-bin overrides, kill-switch/freeze guards).

## 2025-11-02 (Sprint 4)
- Added ledger/monitor freshness checks plus panic-backoff aggregation to the pilot ramp report; sequential CuSum and freeze-window violations now force series-level `NO-GO` decisions.
- Extended runtime monitors with `ev_seq_guard`, `freeze_window`, and inline kill-switch visibility; panic backoff is emitted when three monitor families alert inside 30 minutes.
- Introduced `python -m kalshi_alpha.exec.pilot_bundle` (`make pilot-bundle`) to bundle pilot readiness JSON/Markdown, monitors, scoreboards, ladder reports, and a telemetry slice into a single tarball with manifest metadata.
- Updated the runbook with the new freeze policy, pilot bundle workflow, rollback guidance, and review checklist.

## 2025-11-02
- Hardened the Kalshi HTTP client with header-only RSA-PSS signing (no bearer tokens), exponential backoff, and structured logging.
- Refactored `LiveBroker` to rely on the header-signed client, enforce idempotency via locking, and guard against duplicate submissions.
- Added integration tests validating signature construction, retry behaviour, query exclusion, and millisecond timestamps; refreshed broker safety tests to use the new client abstraction.
- Documented the credential expectations and connectivity flow in `.env.example`, `README.md`, and `docs/RUNBOOK.md`.
- Introduced an authenticated websocket client with reconnect/backoff, plus a live smoke CLI path (`sanity_check --live-smoke`) that exercises read-only REST checks.
- Expanded the paper ledger to capture latency, partial fills, slippage ticks, and expected vs. realized EV; scoreboard and ladder reports now plot EV honesty with confidence badges.
- Added automatic fill ratio and slippage calibration from live ledgers with persisted state (`fill_alpha.json`, `slippage.json`) and regression tests.
- Landed production risk configs (`pal_policy.yaml`, `portfolio.yaml`, `quality_gates.yaml`) with CI guardrails to prevent accidental loosening of limits.
- Introduced the telemetry sink (`data/raw/kalshi/.../exec.jsonl`) plus helper make targets (`make telemetry-smoke`, `make report`, `make live-smoke`).
- Extended telemetry to capture REST/WS latency, sanitized order books, and auth streak metadata; new monitor CLI (`make monitors`) produces JSON artifacts and optional Slack alerts.
- Added a ramp policy engine (`make pilot-readiness`) that enforces fill/Δbps/t-stat criteria, emits GO/NO-GO multipliers, and writes machine-readable readiness JSON.
- Published systemd timers and logrotate templates under `configs/` for daily runs, telemetry shipping, and recurring monitors.
