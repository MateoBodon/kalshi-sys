# Open Questions

## Metadata
- Generated: 2026-01-10T11:43:04Z
- Git SHA: 31316e59451269689f2da173d8a9c6d9049d3d5e
- Branch: codex/TICKET-111_project_state_refresh
- Commands: `python tools/project_state_build.py`, `python3 tools/agentic/project_state_refresh.py --zip`, `rg --files`, `sed -n '1,200p' README.md`, `sed -n '1,200p' docs/PROGRESS.md`, `sed -n '1,200p' CHANGELOG.md`, `sed -n '1,200p' pyproject.toml`, `sed -n '1,200p' Makefile`

## Execution evidence gaps
- What is the current empirical maker fill curve for index ladders, and how is it measured from TOB snapshots? (See `reports/fillcalib/README.md` and `data/proc/fillcalib/*`.)
- Do we have recent, per-window basis audits for INXU/NASDAQ100U and INX/NASDAQ100? (See `reports/settlement_basis/`.)

## Gating & readiness
- What are the most recent calibration ages for index PMFs and sigma_tod curves? (Check `reports/calibration/calibration_ages_<ASOF_DATE>.md` or `data/proc/calib/index/*/*/params.json`.)
- Are clock-skew warnings routinely triggered during scans? (See `reports/<SERIES>/<DATE>.md` monitor sections.)

## Ops / deployment
- Which production environment is currently authoritative for supervisor runs (local vs AWS)? (See `docs/runbooks/` and `docs/PROGRESS.md`.)
- Are CloudWatch metrics and log shipping in place and verified on the latest deployment? (See `docs/PROGRESS.md` entries for 2025-12-22.)

## Configuration clarity
- Which config file is the canonical source of fee rules during scans (`configs/fees.json` vs `data/proc/state/fees.json`), and how is refresh enforced?
- Is `FAMILY=index` enforced by default in production entrypoints? If not, which runners still allow macro families to run?
