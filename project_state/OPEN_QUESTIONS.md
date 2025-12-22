# Open Questions

## Metadata
- Generated: 2025-12-22T19:42:20Z
- Git SHA: a907a2eed87531d8178c3dc183d6f070182f9ebe
- Branch: codex/TICKET-000_project_state_rebuild
- Commands: `python tools/project_state_build.py`, `rg --files`, `sed -n '1,200p' README.md`, `sed -n '1,200p' docs/PROGRESS.md`, `sed -n '1,200p' CHANGELOG.md`, `sed -n '1,200p' pyproject.toml`, `sed -n '1,200p' Makefile`

## Execution evidence gaps
- What is the current empirical maker fill curve for index ladders, and how is it measured from TOB snapshots? (See `reports/fillcalib/README.md` and `data/proc/fillcalib/*`.)
- Do we have recent, per-window basis audits for INXU/NASDAQ100U and INX/NASDAQ100? (See `reports/settlement_basis/`.)

## Gating & readiness
- What are the most recent calibration ages for index PMFs and sigma_tod curves? (Check `data/proc/calib/index/*/*/params.json`.)
- Are clock-skew warnings routinely triggered during scans? (See `reports/<SERIES>/<DATE>.md` monitor sections.)

## Ops / deployment
- Which production environment is currently authoritative for supervisor runs (local vs AWS)? (See `docs/runbooks/` and `docs/PROGRESS.md`.)
- Are CloudWatch metrics and log shipping in place and verified on the latest deployment? (See `docs/PROGRESS.md` entries for 2025-12-22.)

## Configuration clarity
- Which config file is the canonical source of fee rules during scans (`configs/fees.json` vs `data/proc/state/fees.json`), and how is refresh enforced?
- Is `FAMILY=index` enforced by default in production entrypoints? If not, which runners still allow macro families to run?
