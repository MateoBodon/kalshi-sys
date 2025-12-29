# Changelog

## 2025-12-29
- Added calibration-age inspector + CLI report (`reports/calibration/calibration_ages_<ASOF_DATE>.md`) and surfaced calibration status in pilot readiness + scoreboard outputs.
- Preflight calibration NO-GO reasons now include explicit params file paths for stale/missing calibration artifacts.
- Pilot readiness JSON/markdown now includes calibration age summaries and writes the calibration ages report during `make pilot-readiness`.
- Fix: `make calibrate-index` now invokes the `jobs` calibration modules with the correct import path (`PYTHONPATH=src`), and index calibration params were refreshed from live Polygon data.

## 2025-12-26
- Added fill calibration dataset builder that derives conservative maker fill curves from TOB + quote-intent telemetry, with proxy-fill disclaimers and optional parquet output.
- Scanner now records fill-curve status (uncalibrated only when curves are missing/invalid) and clamps fill alpha from the latest fillcalib curves.
- Freshness monitor now consults Polygon `/v1/marketstatus/now` so closed/extended hours do not flag `polygon_index.websocket` as stale.
- Added `python -m kalshi_alpha.exec.market_status` CLI for ops to print market status + server time.
- Verified telemetry-only dry-run captures TOB + quote-intent artifacts, wrote ops telemetry volume report, and proved retention pruning with a synthetic old-file deletion.
- `make gpt-bundle` now includes telemetry artifacts (`data/proc/telemetry/*` and `reports/ops/telemetry_volume_*.md`) when present.
- `make gpt-bundle` now includes fillcalib curves, pilot readiness reports, and calibration markdown, and fails closed when ARTIFACTS.md lists files missing from the bundle.
- Added a gpt-bundle staging helper + regression tests to ensure calibration subtrees and readiness artifacts are always included and ARTIFACTS omissions fail closed.

## 2025-12-25
- Fix: Polygon indices REST snapshot fallback now uses `/v3/snapshot/indices`, parses v3 fields (value/session/last_updated), and fails closed on NOT_ENTITLED responses with updated unit tests.
- Fix: Polygon WS REST fallback now consults `/v1/marketstatus/now` and skips fallback during closed/extended hours (logs status + serverTime), with new unit coverage.

## 2025-12-23
- Gitignored `reports/` and removed tracked report outputs so report artifacts stay local-only.
- Updated `AGENTS.md` and `docs/DOCS_AND_LOGGING_SYSTEM.md` policy docs (scope/safety/logging refresh).
- Ticket #101: decoupled index GO/NO-GO from macro freshness with explicit scope metadata, index-specific freshness/quality gate configs, and scoped go/no-go artifacts.
- Ticket #102: promoted settlement basis audit to a preflight gate with daily JSON/MD artifacts, flip-risk summaries, and fixture-based tests.
- Ticket #103: added bounded TOB + quote-intent telemetry capture to `data/proc/telemetry` (gzipped), per-window caps, and housekeeping retention.
- Fix: settlement basis audit now uses authenticated trade-api/v2 with KX series mapping for index ladders.
- Ticket #103 RETRY: added telemetry-only dry-run override with run metadata, ops telemetry volume report, and improved GPT bundle diff range.

## 2025-12-22
- Rebuilt `project_state/` snapshot with generated inventories, symbol index, dependency graph, and navigation index.
- Ticket #10: validated CloudWatch agent config on Ubuntu, switched log shipping config to syslog tailing, and captured aws logs proof for kalshi-supervisor-index.
- Ticket #9 RETRY: captured EC2 systemd proof (venv ExecStart, no PYTHONPATH, dry-run running) and updated run log.
- Ticket #9: hardened packaging/systemd by switching supervisor unit + runbooks to venv python (no PYTHONPATH), added EC2 bootstrap steps, and added scipy/pandas runtime deps for index models.
- Ticket #6: `preflight_index` and `supervisor_index` now emit GO/NO-GO summary lines, always write `reports/_artifacts/go_no_go.json`, and have stdout fixture coverage for both CLIs.
- Ticket #8: refreshed AWS supervisor dry-run wiring with `--series INXU` in the systemd unit, added CloudWatch journald config + index monitor timer templates, and expanded the AWS runbook with EC2/systemd/CloudWatch copy/paste steps.
- Ticket #8: captured EC2 systemd + CloudWatch proof for two dry-run windows (sanitized excerpts in run log).

## 2025-12-21
- Added a settlement basis audit CLI for index ladders, offline fixtures + unit tests, and daily report outputs comparing Polygon window values to Kalshi expiration values.
- Fixed backtest CLI subprocess tests to inject repo PYTHONPATH so `python -m kalshi_alpha...` resolves in pytest runs.
- Added bounded TOB snapshot + quote-intent logging for index ladders, a fill-calibration dataset builder, and reporting notes for calibration inputs.
- Enforced pilot broker-boundary safety checks (maker-only crossing, window freeze, caps, kill switch) with new live-broker tests and explicit ack/env gating.
- Ticket #4 RETRY: tightened pilot config to index-only ladders, enforced TOB freshness at the live broker boundary, and added stale-TOB + kill-switch queue safety tests.
- Added AWS supervisor runbook + on-call checks, a systemd supervisor template, and supervisor_index CLI aliases for `--dry-run` and `--series` with tests; stabilized the macro-stale index scanner fixture test with a fixed `--now` and clock-skew override.
- Added GPT bundle verification tooling, hardened `make gpt-bundle` diff generation, and added bundle hygiene tests for missing artifacts/placeholder diffs.
- Gitignored local run logs and GPT bundle artifacts (keep on disk, include in bundles for review).

## 2025-12-20
- Scoped index GO/NO-GO evaluation to index-only freshness + quality gates, added quality-gates scope plumbing for index runners, and updated index readiness/scoreboard outputs plus tests so macro staleness no longer blocks index runs.

## 2025-11-20
- Critical Fix: Polygon batch websocket payloads (`stream_aggregates`) now accept list or dict messages, preventing supervisor crashes on Massive bursts.
- Feat: Skew-Normal Pricing for hourly index above/below; added `skew` input to bias downside tails and protect maker inventory.
- New 24/7 `kalshi_alpha.exec.supervisor` daemon orchestrates hourly INXU/NASDAQ100U scans and the 15:50 EOD close run; it keeps Polygon indices websockets alive, trips the kill switch when latency/age exceeds 500 ms, and drops heartbeats so ops can see status at a glance.
- `scan_ladders --sniper` now hits top-of-book mispricings (>5% probability gap) as taker orders, caps size to visible depth, tags liquidity in metadata, and records sniper counts/thresholds in monitors for dashboards.
- Added proof-of-fill CLI (`scripts/proof_of_fill.py`) to reconcile Kalshi order history with the ledger, print per-window fill/PnL tables, and persist `pnl_window_YYYY-MM-DD.parquet` artifacts.
- Introduced staged sizing via `configs/size_ladder.yaml` plus loader helpers; the live loop now respects the current ladder stage and final-minute freshness freezes.
- New promotion audit helper (`scripts/check_promotion_ladder.py`) scores recent PnL/SLO artifacts to recommend stage upgrades; unit tests cover ladder parsing and promotion paths.
- Expanded tests for live close gating and freshness breaches, keeping the hourly loop aligned with discovery close targets and NO-GO rules.

## 2025-11-11
- Rolled out `kalshi_alpha.exec.slo` + scoreboard SLO lines (freshness/time-at-risk/VaR headroom) with optional CloudWatch publishing via `python -m kalshi_alpha.exec.scoreboard --publish-slo-cloudwatch`.
- Added `python -m report.digest` (daily Markdown + PNG digest with optional S3 upload) and linked it from REPORT/runbooks.
- Delivered `monitor/drift_sigma_tod.py` + `kalshi_alpha.exec.monitors.sigma_drift`; scanner shrink factors now respect Sigma drift alerts (`tests/test_sigma_drift_monitor.py`).
- Landed the fee/rule watcher CLI (`monitor/fee_rules_watch.py`), stateful ack flow, and a runtime gate that blocks `scan_ladders` until the change is acknowledged (`tests/test_fee_rules_watch.py`).
- Introduced `kalshi_alpha.exec.limits` (public `LossBudget` + `ProposalLimitChecker`) so PAL + daily/weekly stops are enforced during proposal generation; runners and tests were updated accordingly (`tests/test_limits.py`, `tests/test_u_hourly_rotation.py`).
- Documented promotion milestones in `docs/promotion_ladder.md` and wired hourly/EOD runbooks to the fee watcher, sigma drift monitor, and daily digest workflow.
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
