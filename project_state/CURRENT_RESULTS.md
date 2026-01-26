# Current Results

## Metadata
- Updated: 2026-01-26T00:04:48Z
- Git SHA: c78b933ec78e5a01a1b9e943de3dfd17ec5cd260
- Branch: codex/TICKET-000_project_state_refresh
- Sources: local `reports/` artifacts (gitignored), `docs/PROGRESS.md`

## Scoreboards
- `reports/scoreboard_7d.md` and `reports/scoreboard_30d.md` are not present in this repo snapshot (gitignored artifacts). Regenerate via `python -m kalshi_alpha.exec.scoreboard` if needed.

## Pilot readiness (local artifact)
- `reports/pilot_readiness.md` exists locally and was generated 2025-12-29. It reports global NO-GO reasons `ledger_stale` and `monitors_stale` with no per-series fills recorded.

## Recent ladder report samples (local artifacts)
- `reports/INXU/2025-12-30.md`: NO-GO; monitors show `clock_skew_exceeded` and `polygon_ws_stale` reasons in the quality gate list.
- `reports/NASDAQ100U/2025-12-26.md`: NO-GO; dry mode; sample fill/EV tables included.

## Other artifacts present
- `reports/_artifacts/go_no_go.json` exists (latest GO/NO-GO status artifact).
- `reports/_artifacts/*_ledger.json|csv` sample ledger artifacts present (earliest dated 2025-12-21).

## Notes / caveats
- Results reflect local, gitignored artifacts only. Live trading is gated and not inferred from these reports.
