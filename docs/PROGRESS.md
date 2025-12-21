# Progress

## 2025-12-20
- Ticket #1 — Index-only GO/NO-GO gates: scoped quality gates and freshness summaries to index feeds, and index runners now forward the index scope so macro staleness no longer blocks index scans.
  - Gate status: PAPER
  - Evidence: agent run log `docs/agent_runs/20251220_231551Z_TICKET-001_index_only_gates/README.md`

## 2025-12-21
- Ticket #2 — Settlement basis audit: added a reproducible Polygon-vs-Kalshi expiration audit tool, offline fixtures, and report outputs for index ladder windows.
  - Gate status: PAPER
  - Evidence: agent run log `docs/agent_runs/20251221_011149Z_TICKET-002_settlement_basis_audit/README.md`
- Ticket #3 — TOB snapshot logger + fill-calibration dataset skeleton: added bounded TOB + quote-intent logging for index ladders, dataset builder, and calibration README.
  - Gate status: PAPER
  - Evidence: agent run log `docs/agent_runs/20251221_033013Z_TICKET-003_tob_snapshot_logger/README.md`
- Ticket #4 — Pilot safety enforced at broker boundary: added live broker boundary guards for pilot caps/window/maker-only, kill-switch submit blocking, and new pilot safety tests.
  - Gate status: PAPER
  - Evidence: agent run log `docs/agent_runs/20251221_055147Z_TICKET-004_pilot_broker_boundary/README.md`
