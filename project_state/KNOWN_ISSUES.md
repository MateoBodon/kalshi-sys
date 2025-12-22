# Known Issues

## Metadata
- Generated: 2025-12-22T19:42:20Z
- Git SHA: a907a2eed87531d8178c3dc183d6f070182f9ebe
- Branch: codex/TICKET-000_project_state_rebuild
- Commands: `python tools/project_state_build.py`, `rg --files`, `sed -n '1,200p' README.md`, `sed -n '1,200p' docs/PROGRESS.md`, `sed -n '1,200p' CHANGELOG.md`, `sed -n '1,200p' pyproject.toml`, `sed -n '1,200p' Makefile`

## Evidence gaps
- Scoreboards report no ledger data in recent windows (`reports/scoreboard_7d.md`, `reports/scoreboard_30d.md`).
- Pilot readiness shows 0/4 GO with fills=0 and insufficient_data (`reports/pilot_readiness.md`).

## Operational signals
- Recent INXU report shows NO-GO and clock-skew exceeded (`reports/INXU/2025-12-22.md` monitor section).

## Data calibration gaps
- Calibration freshness is enforced, but the latest calibration ages are not surfaced in a single, committed summary artifact (see `data/proc/calib/index/*` for raw params).

## Scope risk
- Macro strategy code exists and could run if family scoping is misconfigured; index-only scope must be enforced by entrypoints (see `docs/PLAN_OF_RECORD.md`).
