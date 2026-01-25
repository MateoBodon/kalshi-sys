# Experiments

## Metadata
- Generated: 2026-01-10T11:43:04Z
- Git SHA: 31316e59451269689f2da173d8a9c6d9049d3d5e
- Branch: codex/TICKET-111_project_state_refresh
- Commands: `python tools/project_state_build.py`, `python3 tools/agentic/project_state_refresh.py --zip`, `rg --files`, `sed -n '1,200p' README.md`, `sed -n '1,200p' docs/PROGRESS.md`, `sed -n '1,200p' CHANGELOG.md`, `sed -n '1,200p' pyproject.toml`, `sed -n '1,200p' Makefile`

## Status
- No `experiments/` directory exists in this repo at the time of this snapshot.
- Ad-hoc research outputs live under `reports/`, `report/`, and `data/proc/`.

## Representative research artifacts
- `reports/backtests/hourly/metrics.md` (index hourly backtest metrics).
- `reports/backtests/close/metrics.md` (index close backtest metrics).
- `reports/settlement_basis/2025-11-10_INXU.md` (basis audit sample).
- `report/agent_logs/` (local-only agent notes and proofs).

## Exclusions
- `reports/` and `data/` are excluded from deep parsing; only small representative files are sampled.
