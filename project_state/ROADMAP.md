# Roadmap

## Metadata
- Generated: 2026-01-10T11:43:04Z
- Git SHA: 31316e59451269689f2da173d8a9c6d9049d3d5e
- Branch: codex/TICKET-111_project_state_refresh
- Commands: `python tools/project_state_build.py`, `python3 tools/agentic/project_state_refresh.py --zip`, `rg --files`, `sed -n '1,200p' README.md`, `sed -n '1,200p' docs/PROGRESS.md`, `sed -n '1,200p' CHANGELOG.md`, `sed -n '1,200p' pyproject.toml`, `sed -n '1,200p' Makefile`

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
