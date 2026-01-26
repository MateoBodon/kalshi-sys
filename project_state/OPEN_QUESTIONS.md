# Open Questions

## Metadata
- Updated: 2026-01-26T00:04:48Z
- Git SHA: c78b933ec78e5a01a1b9e943de3dfd17ec5cd260
- Branch: codex/TICKET-000_project_state_refresh
- Sources: local `reports/` artifacts (gitignored), `docs/PROGRESS.md`, `README.md`

## Execution evidence gaps
- What is the current empirical maker fill curve for index ladders, and how is it measured from TOB snapshots? (See `reports/fillcalib/README.md` and `data/proc/fillcalib/*`.)
- Do we have recent, per-window basis audits for INXU/NASDAQ100U and INX/NASDAQ100? (See `reports/settlement_basis/`.)
- Are scoreboard artifacts regenerated and archived (e.g., `reports/scoreboard_7d.md`, `reports/scoreboard_30d.md`) in recent run logs?

## Gating & readiness
- What are the most recent calibration ages for index PMFs and sigma_tod curves? (Check `reports/calibration/calibration_ages_<ASOF_DATE>.md` or `data/proc/calib/index/*/*/params.json`.)
- Are clock-skew warnings routinely triggered during scans? (See `reports/<SERIES>/<DATE>.md` monitor sections.)
- Are ledger and monitor freshness artifacts being updated regularly (see `reports/_artifacts/*` and `reports/pilot_readiness.md`)?

## Ops / deployment
- Which production environment is currently authoritative for supervisor runs (local vs AWS)? (See `docs/runbooks/` and `docs/PROGRESS.md`.)
- Are CloudWatch metrics and log shipping in place and verified on the latest deployment? (See `docs/PROGRESS.md` entries for 2025-12-22.)

## Configuration clarity
- Which config file is the canonical source of fee rules during scans (`configs/fees.json` vs `data/proc/state/fees.json`), and how is refresh enforced?
- Is `FAMILY=index` enforced by default in production entrypoints? If not, which runners still allow macro families to run?
