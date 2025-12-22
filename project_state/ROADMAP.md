# Roadmap

## Metadata
- Generated: 2025-12-22T19:42:20Z
- Git SHA: a907a2eed87531d8178c3dc183d6f070182f9ebe
- Branch: codex/TICKET-000_project_state_rebuild
- Commands: `python tools/project_state_build.py`, `rg --files`, `sed -n '1,200p' README.md`, `sed -n '1,200p' docs/PROGRESS.md`, `sed -n '1,200p' CHANGELOG.md`, `sed -n '1,200p' pyproject.toml`, `sed -n '1,200p' Makefile`

## Near-term (index ladders only)
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
