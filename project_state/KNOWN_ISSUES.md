# Known Issues

## Metadata
- Updated: 2026-01-26T00:04:48Z
- Git SHA: c78b933ec78e5a01a1b9e943de3dfd17ec5cd260
- Branch: codex/TICKET-000_project_state_refresh
- Sources: local `reports/` artifacts (gitignored), `docs/PROGRESS.md`, `README.md`

## Evidence gaps
- Scoreboard artifacts (`reports/scoreboard_7d.md`, `reports/scoreboard_30d.md`) are missing in this repo snapshot; regenerate with `python -m kalshi_alpha.exec.scoreboard`.
- Local pilot readiness report (2025-12-29) shows global NO-GO reasons `ledger_stale` and `monitors_stale`; fill evidence is insufficient.

## Operational signals
- Recent local INXU report (2025-12-30) shows `clock_skew_exceeded` and `polygon_ws_stale` among quality-gate reasons.
- Report artifacts embed absolute paths from the machine that generated them (e.g., `/Users/...`), so paths are not portable across hosts.

## Data calibration gaps
- Calibration ages are reported under `reports/calibration/calibration_ages_<ASOF_DATE>.md` when pilot readiness is generated; artifacts are gitignored, so confirm via local run logs or `data/proc/calib/index/*/*/params.json`.

## Scope risk
- Macro strategy code exists and could run if family scoping is misconfigured; index-only scope must be enforced by entrypoints (see `docs/PLAN_OF_RECORD.md`).
