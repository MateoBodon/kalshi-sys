# RUN — TICKET-102

Goal: Basis audit becomes a required daily artifact and can fail-closed when missing/stale or risky.

Initial git sha: 699ca934ed4ae091f62fc787b44f0cecc535946b

Acceptance checklist (from docs/CODEX_SPRINT_TICKETS.md):
- New artifact written per series/day:
  - basis distribution (quantiles)
  - per-window deltas
  - “flip risk” flags for likely strike spacing / quote distances
- Preflight or quality gate fails closed if basis audit missing/stale for the series/day.
- Add fixture-based test that validates output schema on a synthetic window.
