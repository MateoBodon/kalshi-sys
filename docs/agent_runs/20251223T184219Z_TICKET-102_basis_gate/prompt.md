# Prompt

TICKET-102 — Settlement basis audit as first-class gate + daily artifact + strike flip risk summary.

Key requirements (excerpt):
- Ensure basis audit tool emits daily JSON + MD artifacts per series.
- JSON must include: series, asof_date, generated_at, sample_count, basis_quantiles, per_window_deltas, flip_risk.
- Add flip-risk heuristic with conservative defaults and rationale.
- Wire basis audit gate into preflight (fail-closed on missing/stale/flip-risk).
- Add fixture-based tests for tool output and preflight gating.
- Run: `pytest -q`, `python tools/settlement_basis_audit.py --help`, `python -m kalshi_alpha.exec.preflight_index --offline`.
- Update docs: PLAN_OF_RECORD, DOCS_AND_LOGGING_SYSTEM, CODEX_SPRINT_TICKETS, PROGRESS, CHANGELOG.
- Create run log under docs/agent_runs/<RUN_NAME>/ and run `make gpt-bundle`.
