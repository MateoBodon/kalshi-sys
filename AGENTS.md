# AGENTS.md — kalshi-sys (Index Ladders)

## Overview
Automated maker strategies for S&P 500 (INX/INXU) and Nasdaq-100 (NASDAQ100/NASDAQ100U) noon (12:00 ET AB/BL) and close (16:00 ET ranges). Do not leak secrets.

## Build & test
- Python 3.11, uv/poetry or pip-tools (project uses pytest, ruff, mypy).
- Run all checks:
  make test       # pytest -q
  make lint       # ruff + mypy
  make smoke      # offline scanners render to reports/

## Secrets
- Polygon key from macOS Keychain item: `kalshi-sys:POLYGON_API_KEY`. Fallback env: `POLYGON_API_KEY`.
- NEVER print(), log, or commit secrets. If key missing, fail fast.

## Data
- Historical: Polygon REST minute aggregates for I:SPX and I:NDX. Handle 50k-bar limit with ≤1-month chunks.
- Live windows: WebSocket aggregates (minute/second) only during 11:40–12:01 and 15:45–16:01 ET.

## Fees (hard rule)
- Index series (INX*, NASDAQ100*): maker fee = round_up(0.035 * C * P * (1-P)).
- General maker fee: 0.0175 coefficient.
- EV_after_fees must use the correct series mapping. Add tests (golden rows).

## Calibrations
- Noon & close: fit m_TOD(τ), late-day variance bump, PIT mapping.
- Refresh ≤14 days. Store under data/proc/calib/index/{spx,ndx}/{noon,close}/params.json.

## Scanners & thresholds
- Enforce EV_after_fees ≥ $0.05, maker-only, ≤2 bins/series.
- Δbps = EV_after_fees × α − slippage.
- Cancel quotes by T−2s.

## Readiness gates
- Freshness OK, calibration age ≤14d, paper fills ≥300/14d, Δbps ≥6, t≥2.
- Print GO/NO-GO with reasons. If NO, do not place live orders.

## Runbook (operators)
- Noon window: 11:45–12:00 ET; Close window: 15:50–16:00 ET.
- Preflight → smoke (data & auth) → scanners → optional microlive 1-lots → scoreboard → readiness.

## Code style
- Black/ruff defaults, mypy strict. Small focused modules. Tests for every fee/math path.
- Commit message: feat|fix|docs(scope): message.

## Commands the agent can run
- `make ingest-index START=YYYY-MM-DD END=YYYY-MM-DD`  
  Downloads Polygon minute bars for I:SPX and I:NDX into `data/raw/polygon/`.
- `make calibrate-index`  
  Builds noon (`jobs.calibrate_hourly`) and close (`jobs.calibrate_close`) params under `data/proc/calib/index/`.
- `make scan-index-noon` / `make scan-index-close`  
  Runs offline scanners, writing CSV + Markdown to `reports/index_ladders/<SERIES>/`.
- `make micro-index`  
  1-lot microlive dry run: forwards to `scan_ladders` pilot mode, refits α/slippage, regenerates scoreboard.
- `make pilot-readiness`  
  Generates ramp report + GO/NO-GO summary (requires `reports/_artifacts/monitors/freshness.json`).

## Safety
- Sandbox: ok to run build/tests and fetch data. Do not modify unrelated files. Stop on missing creds.
