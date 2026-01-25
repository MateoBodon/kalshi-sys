# Known Issues

## Metadata
- Generated: 2026-01-10T11:43:04Z
- Git SHA: 31316e59451269689f2da173d8a9c6d9049d3d5e
- Branch: codex/TICKET-111_project_state_refresh
- Commands: `python tools/project_state_build.py`, `python3 tools/agentic/project_state_refresh.py --zip`, `rg --files`, `sed -n '1,200p' README.md`, `sed -n '1,200p' docs/PROGRESS.md`, `sed -n '1,200p' CHANGELOG.md`, `sed -n '1,200p' pyproject.toml`, `sed -n '1,200p' Makefile`

## Evidence gaps
- Scoreboards report no ledger data in recent windows (`reports/scoreboard_7d.md`, `reports/scoreboard_30d.md`).
- Pilot readiness shows 0/4 GO with fills=0 and insufficient_data (`reports/pilot_readiness.md`).

## Operational signals
- Recent INXU report shows NO-GO and clock-skew exceeded (`reports/INXU/2025-12-22.md` monitor section).

## Data calibration gaps
- Calibration ages are now reported under `reports/calibration/calibration_ages_<ASOF_DATE>.md`, but recent artifacts are not committed by default (reports are gitignored); confirm the latest report in run logs.

## Scope risk
- Macro strategy code exists and could run if family scoping is misconfigured; index-only scope must be enforced by entrypoints (see `docs/PLAN_OF_RECORD.md`).
