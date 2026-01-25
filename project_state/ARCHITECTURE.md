# Architecture

## Metadata
- Generated: 2026-01-10T11:43:04Z
- Git SHA: 31316e59451269689f2da173d8a9c6d9049d3d5e
- Branch: codex/TICKET-111_project_state_refresh
- Commands: `python tools/project_state_build.py`, `python3 tools/agentic/project_state_refresh.py --zip`, `rg --files`, `sed -n '1,200p' README.md`, `sed -n '1,200p' docs/PROGRESS.md`, `sed -n '1,200p' CHANGELOG.md`, `sed -n '1,200p' pyproject.toml`, `sed -n '1,200p' Makefile`

## System summary
Kalshi Alpha is a Python 3.11+ monorepo for research, pricing, scanning, and (guarded) execution of Kalshi ladder markets. Current hard scope is index ladders only (INX/INXU/NASDAQ100/NASDAQ100U hourly + close). The system is built to be fail-closed: quality gates, kill-switches, and broker safeguards must pass before any live behavior is armed. Default execution is paper/dry and all live paths require explicit acknowledgement and credentials.

## Top-level directory map
- `src/kalshi_alpha/`: Primary application code (drivers, strategies, execution, risk, scheduling).
- `configs/`: Risk/quality gate configs, fee tables, pilot caps, systemd/logrotate templates.
- `jobs/`: Calibration jobs for index models and execution curves.
- `scripts/`: Operational scripts (WS listeners, parity checks, fixtures, AWS jobs).
- `tools/`: Auxiliary CLIs (replay, settlement basis audit, failover smoke).
- `report/`: Reporting utilities (digest, honesty, agent logs).
- `reports/`: Generated reports, scoreboards, pilot readiness outputs, artifacts.
- `data/`: Raw + processed datasets, calibrations, ledgers, and state (gitignored).
- `tests/`: Pytest suite with fixtures for offline-safe validation.
- `docs/`: Runbooks, process docs, sprint tickets, and progress logs.
- `project_state/`: Repo self-description snapshot (this folder).

## Major components
- **Data ingestion**: `src/kalshi_alpha/drivers/*` pull Polygon indices and macro sources; `src/kalshi_alpha/datastore/*` writes snapshots to `data/raw` and processed tables to `data/proc`.
- **Strategy & pricing**: `src/kalshi_alpha/strategies/*` generate ladder PMFs; `src/kalshi_alpha/core/pricing/*` aligns PMFs to strike grids and computes EVs; `src/kalshi_alpha/models/pmf_index.py` handles index PMF utilities.
- **Risk & limits**: `src/kalshi_alpha/core/risk/*`, `src/kalshi_alpha/risk/*`, and `src/kalshi_alpha/exec/limits.py` enforce PAL, VaR, drawdown caps, and pilot constraints.
- **Execution & scanning**: `src/kalshi_alpha/exec/*` orchestrates scanners, runners, brokers, ledgers, and monitors. Index-specific entrypoints live in `src/kalshi_alpha/exec/supervisor_index.py`, `preflight_index.py`, and `exec/scanners/scan_index_*`.
- **Scheduling**: `src/kalshi_alpha/sched/windows.py` defines ET-aware hourly and close windows; `exec/pipelines/*` apply calendar-aware scheduling.
- **Monitoring & reporting**: `src/kalshi_alpha/exec/monitors/*`, `monitor/*`, and `report/*` generate freshness checks, sigma drift, fee/rule watch, scoreboards, and digests.

## Key entrypoints
- CLI scanner: `kalshi-scan` (script in `pyproject.toml`) → `kalshi_alpha.exec.runners.scan_ladders:main`.
- Index preflight: `python -m kalshi_alpha.exec.preflight_index`.
- Index supervisor: `python -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run`.
- Index scanners: `python -m kalshi_alpha.exec.scanners.scan_index_hourly` and `scan_index_close` (offline fixtures supported).
- Pipelines: `python -m kalshi_alpha.exec.pipelines.daily`, `today`, `week`.
- Scoreboard/reporting: `python -m kalshi_alpha.exec.scoreboard`, `python -m report.digest`.

## Default safety posture
- Live trading is feature-flagged and guarded by explicit ack flags, kill-switch, and broker boundary checks.
- Pilot constraints are enforced in config (`configs/pilot.yaml`) and broker boundary logic.
- Quality gates and freshness checks are required before GO decisions; index scope is enforced via `configs/quality_gates.index.yaml` and `configs/freshness.index.yaml`.

## Generated references
- Module inventory: `project_state/MODULE_SUMMARIES.md`
- Function-level index: `project_state/FUNCTION_INDEX.md`
- Internal import graph: `project_state/DEPENDENCY_GRAPH.md`
