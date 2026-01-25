# Pipeline Flow

## Metadata
- Generated: 2026-01-10T11:43:04Z
- Git SHA: 31316e59451269689f2da173d8a9c6d9049d3d5e
- Branch: codex/TICKET-111_project_state_refresh
- Commands: `python tools/project_state_build.py`, `python3 tools/agentic/project_state_refresh.py --zip`, `rg --files`, `sed -n '1,200p' README.md`, `sed -n '1,200p' docs/PROGRESS.md`, `sed -n '1,200p' CHANGELOG.md`, `sed -n '1,200p' pyproject.toml`, `sed -n '1,200p' Makefile`

## Primary entrypoints (index scope)
- `kalshi-scan` → `kalshi_alpha.exec.runners.scan_ladders:main` (core scanner).
- `python -m kalshi_alpha.exec.preflight_index` (GO/NO-GO checks for index ladders).
- `python -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run` (window supervisor w/ preflight + WS freshness).
- `python -m kalshi_alpha.exec.scanners.scan_index_hourly --series INXU NASDAQ100U --offline --fixtures-root tests/data_fixtures`.
- `python -m kalshi_alpha.exec.scanners.scan_index_close --series INX NASDAQ100 --offline --fixtures-root tests/data_fixtures`.

## Macro-capable entrypoints (often disabled by family scope)
- `python -m kalshi_alpha.exec.pipelines.daily --mode pre_cpi|pre_claims|teny_close|weather_cycle`.
- `python -m kalshi_alpha.exec.pipelines.today` (calendar driven).
- `python -m kalshi_alpha.exec.pipelines.week` (weekly orchestrator).

## Calibration jobs
- `python -m jobs.calibrate_hourly --series INXU NASDAQ100U`
- `python -m jobs.calibrate_close --series INX NASDAQ100`
- `python -m jobs.calibrate_index_polygon_model` (index PMF calibration helpers).
- Outputs: `data/proc/calib/index/<symbol>/<horizon>/params.json`.

## Index supervisor flow (hourly + close)
1) **Schedule**: `kalshi_alpha.sched.windows` determines ET windows and freeze times.
2) **Preflight**: `kalshi_alpha.exec.preflight_index` validates calibration freshness, env credentials, and kill-switch status.
3) **Websocket freshness**: `kalshi_alpha.data.ws_sentry` / `drivers.polygon_index_ws` enforce strict final-minute freshness.
4) **Runner**: `kalshi_alpha.exec.runners.micro_index` launches a single-window scan (dry or live).
5) **Scanner**: `kalshi_alpha.exec.runners.scan_ladders` generates proposals, applies fees (`configs/fees.json`), fill/slippage, and EV honesty gates.
6) **Risk & broker checks**: `exec/limits.py`, `core/risk/*`, and broker adapters (`brokers/kalshi/*`) enforce caps and maker-only rules.
7) **Artifacts**: writes reports under `reports/<SERIES>/<DATE>.md` and machine artifacts under `reports/_artifacts/`.

## Daily pipeline flow (macro or index)
1) **Ingest**: `kalshi_alpha.datastore.ingest` + drivers (offline fixtures or online fetch).
2) **Calibrate**: strategy-level calibrations (CPI/Claims/TENY/Weather) or index calibrations.
3) **Scan**: `scan_ladders` per series, with optional `--mispricing-only` gating.
4) **Gates**: `core.gates.quality_gates`, freshness monitors, and kill-switch checks.
5) **Ledger + reports**: `exec.ledger` / `exec.reports` / `exec.scoreboard`.

## Monitoring & reporting flow
- **Scoreboards**: `python -m kalshi_alpha.exec.scoreboard` reads ledger + monitors to generate `reports/scoreboard_7d.md` and `reports/scoreboard_30d.md`.
- **Pilot readiness**: `python -m kalshi_alpha.exec.reports.ramp` writes `reports/pilot_readiness.md` + `reports/pilot_ready.json`.
- **Digest**: `python -m report.digest --date yesterday --write` emits `reports/digests/*.md` and PNGs.
- **Fee/rule watcher**: `monitor/fee_rules_watch.py` hashes official docs and blocks scans until changes are acknowledged.

## Key configs used in flows
- `configs/index_ops.yaml` (window offsets, close windows, max bins, min EV).
- `configs/freshness.index.yaml` and `configs/quality_gates.index.yaml` (index-only gates).
- `configs/pilot.yaml` (pilot caps, maker-only, acknowledgement).
- `configs/pal_policy.yaml` / `configs/index_var.yaml` / `configs/index_correlation.yaml` (risk caps).
- `configs/fees.json` (fee coefficients and rounding).

## Typical artifact outputs
- Reports: `reports/<SERIES>/<YYYY-MM-DD>.md` (per scan window).
- Scoreboards: `reports/scoreboard_7d.md`, `reports/scoreboard_30d.md`.
- Pilot readiness: `reports/pilot_readiness.md`, `reports/pilot_ready.json`.
- Monitors: `reports/_artifacts/monitors/*.json`.
- Ledger: `data/proc/ledger/*.jsonl` (paper/live), `data/proc/ledger_all.parquet` (aggregate).
