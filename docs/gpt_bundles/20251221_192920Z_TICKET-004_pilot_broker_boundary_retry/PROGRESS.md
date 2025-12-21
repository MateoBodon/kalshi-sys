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
- Ticket #4 RETRY — Pilot safety enforced at broker boundary: tightened pilot config to index-only, enforced TOB staleness checks, added broker-boundary tests (crossing/stale/kill-switch queue), and refreshed run logs (supersedes 20251221_055147Z run).
  - Gate status: PAPER
  - Evidence: agent run log `docs/agent_runs/20251221_192920Z_TICKET-004_pilot_broker_boundary_retry/README.md`; tests `tests/test_broker_live_safety.py`, `src/kalshi_alpha/brokers/kalshi/live.py`
