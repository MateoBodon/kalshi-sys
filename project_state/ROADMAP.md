# Roadmap

## Metadata
- Updated: 2026-01-26T00:04:48Z
- Git SHA: c78b933ec78e5a01a1b9e943de3dfd17ec5cd260
- Branch: codex/TICKET-000_project_state_refresh
- Sources: `docs/PLAN_OF_RECORD.md`, `docs/PROGRESS.md`, `kalshi_alpha_long_term_plan.md`

## Near-term (index ladders only)
- Regenerate scoreboards and pilot readiness artifacts and archive them in run logs (`python -m kalshi_alpha.exec.scoreboard`, `make pilot-readiness`).
- Finalize basis audit coverage for INX/INXU/NASDAQ100/NASDAQ100U windows (`tools/settlement_basis_audit.py`, `reports/settlement_basis/`).
- Build empirical fill curves from TOB logs and integrate into fill models (`tools/build_fillcalib_dataset.py`, `data/proc/fillcalib/*`).
- Validate 24/7 supervisor wiring and CloudWatch pipelines (see `docs/runbooks/` and `docs/PROGRESS.md` entries dated 2025-12-22).

## Mid-term
- Expand index window scheduling + AWS runbooks per `docs/PLAN_OF_RECORD.md` and `kalshi_alpha_long_term_plan.md`.
- Solidify replay parity and EV honesty workflows (`tools/replay.py`, `scripts/parity_gate.py`).

## Long-term
- Transition from PAPER to PILOT only after evidence from fills, basis audit, and ops stability.
- Maintain index-only focus; macro strategies remain paused unless explicitly re-scoped.

## Source documents
- `docs/PLAN_OF_RECORD.md`
- `kalshi_alpha_long_term_plan.md`
- `docs/CODEX_SPRINT_TICKETS.md`
