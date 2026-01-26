# Runbook

## Setup
- `python3 -m venv .venv`
- `. .venv/bin/activate`
- `pip install -e ".[dev]"`

## Build
- Not applicable (Python package).

## Test
- `pytest -q` (canonical)
- `python -m ruff format .`
- `python -m ruff check .`

## Paper-safe entrypoints (index ladders)
- `python -m kalshi_alpha.exec.preflight_index`
- `python -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run`
- `python -m kalshi_alpha.exec.scanners.scan_index_hourly --series INXU NASDAQ100U --offline --fixtures-root tests/data_fixtures`
- `python -m kalshi_alpha.exec.scanners.scan_index_close --series INX NASDAQ100 --offline --fixtures-root tests/data_fixtures`

## Artifacts & outputs
- Reports: `reports/<SERIES>/<YYYY-MM-DD>.md` (gitignored local artifacts)
- Machine artifacts: `reports/_artifacts/*.json`, `reports/_artifacts/monitors/*.json`
- Ledger: `data/proc/ledger/*.jsonl`, aggregate `data/proc/ledger_all.parquet`
- Bundles: `docs/_bundles/*.zip`
- Agent run logs: `docs/agent_runs/<RUN_NAME>/`

## Environment variables (online/live only)
- `POLYGON_API_KEY` for Polygon REST/WS checks.
- `KALSHI_API_KEY_ID`, `KALSHI_PRIVATE_KEY_PEM_PATH` for live broker auth.

## Safety & gating notes
- Default posture is paper-only; live requires explicit CLI acks and credentials.
- Kill switch: presence of `data/proc/state/kill_switch` forces NO-GO.

## Debug
- If scans show `clock_skew_exceeded`, verify host time sync and timezone.
- If `polygon_ws_stale`, check websocket health and `reports/_artifacts/monitors/freshness.json`.
- If imports fail, ensure editable install or set `PYTHONPATH=src`.
