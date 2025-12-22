# Experiments

## Metadata
- Generated: 2025-12-22T19:42:20Z
- Git SHA: a907a2eed87531d8178c3dc183d6f070182f9ebe
- Branch: codex/TICKET-000_project_state_rebuild
- Commands: `python tools/project_state_build.py`, `rg --files`, `sed -n '1,200p' README.md`, `sed -n '1,200p' docs/PROGRESS.md`, `sed -n '1,200p' CHANGELOG.md`, `sed -n '1,200p' pyproject.toml`, `sed -n '1,200p' Makefile`

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
