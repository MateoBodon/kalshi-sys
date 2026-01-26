# TICKET-122

## Goal
Make settlement basis audits routine and fail-closed for INX/INXU/NASDAQ100/NASDAQ100U by adding an all-series basis audit runner and enforcing fresh per-day artifacts in preflight with archived outputs.

## Scope
- Add a single command to generate basis audit JSON+MD artifacts for all four index series for a given ET date.
- Ensure artifacts land in `data/proc/basis/<SERIES>/<YYYY-MM-DD>.json` and `reports/basis/<SERIES>/<YYYY-MM-DD>.md` and can be copied into `--runlog/--archive-dir`.
- Tighten/verify preflight_index basis staleness rules (asof_date + generated_at + flip_risk) as needed.
- Add fixture-based tests covering missing/stale/flip-risk/valid cases.
- Do not change strategy math, pricing, or broker behavior.

## Acceptance Criteria
- One command generates basis audit artifacts for all four index series for a date.
- `preflight_index` blocks GO with `basis_audit_missing`, `basis_audit_stale`, and `basis_flip_risk` when appropriate.
- Tests cover missing -> NO_GO, stale -> NO_GO, flip_risk -> NO_GO, valid -> GO.
- Basis artifacts are portable and can be archived via existing runlog/archive flags.

## Plan
1. Review settlement basis audit tooling and preflight basis gate logic (`tools/settlement_basis_audit.py`, `src/kalshi_alpha/exec/preflight_index.py`).
2. Add an all-series audit runner with runlog/archive support and per-series outputs (`src/kalshi_alpha/exec/...` or `tools/settlement_basis_audit.py`).
3. Tighten basis audit staleness checks if gaps exist and update tests (`tests/exec/test_preflight_index.py`).
4. Add fixtures/tests for missing/stale/flip-risk cases and ensure valid pass.
5. Update docs (`CHANGELOG.md`, `docs/PROGRESS.md`, run log artifacts).

## Notes
- Keep diffs minimal and fail-closed.
