# Current Results

## Metadata
- Generated: 2026-01-10T11:43:04Z
- Git SHA: 31316e59451269689f2da173d8a9c6d9049d3d5e
- Branch: codex/TICKET-111_project_state_refresh
- Commands: `python tools/project_state_build.py`, `python3 tools/agentic/project_state_refresh.py --zip`, `rg --files`, `sed -n '1,200p' README.md`, `sed -n '1,200p' docs/PROGRESS.md`, `sed -n '1,200p' CHANGELOG.md`, `sed -n '1,200p' pyproject.toml`, `sed -n '1,200p' Makefile`

## Scoreboards (latest committed artifacts)
- `reports/scoreboard_7d.md`: Data Freshness OK; no ledger data available for the window.
- `reports/scoreboard_30d.md`: Data Freshness OK; no ledger data available for the window.

## Pilot readiness (latest committed artifact)
- `reports/pilot_readiness.md` (14-day): GO series 0/4 (INXU, NASDAQ100U, INX, NASDAQ100 all NO-GO) due to insufficient_data with fills=0.

## Recent ladder report sample
- `reports/INXU/2025-12-22.md`: NO-GO; dry mode; outstanding orders reported; includes monitor snapshot and window metadata for the 10:00 ET window.

## Summary (as of 2025-12-22)
- No evidence of realized fills in committed artifacts; readiness remains NO-GO.
- Freshness checks are passing in the scoreboard artifacts, but fill evidence and calibration/ledger coverage remain insufficient for GO.

## Notes / caveats
- Results reflect local, committed artifacts only. Live trading is gated and not inferred from these reports.
