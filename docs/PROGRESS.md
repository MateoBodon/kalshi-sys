# Progress

Note: run logs under `docs/agent_runs/` are local-only (gitignored). Use the per-ticket GPT bundle for review.

## 2025-12-22
- Ticket #6 — Fix index CLI smoke entrypoints: added preflight/supervisor summary lines, ensured go/no-go artifact writes on NO-GO, and added stdout fixture coverage for both CLIs.
  - Gate status: PAPER
  - Evidence: agent run log `docs/agent_runs/20251222_002818Z_TICKET-006_index_cli_smoke/README.md`
- Ticket #8 — AWS dry-run deployment verification: pinned the systemd unit to `--series INXU`, added CloudWatch agent + index monitor timer templates, expanded the AWS runbook with EC2/systemd/CloudWatch steps, and collected EC2 systemd + CloudWatch proof for two windows.
  - Gate status: PAPER
  - Evidence: agent run log `docs/agent_runs/20251222_030117Z_TICKET-008_aws_dry_run_deploy_verify/README.md`

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
- Ticket #5 — AWS / 24-7 supervisor wiring: added AWS runbook + on-call checks, a systemd supervisor template, supervisor_index CLI aliases for `--dry-run` + `--series` with tests, and stabilized the macro-stale index scanner fixture test.
  - Gate status: PAPER
  - Evidence: agent run log `docs/agent_runs/20251221_223847Z_TICKET-005_aws_supervisor_wiring/README.md`
- Ticket #7 — Bundle / diff hygiene stop-the-line: added GPT bundle verifier, hardened gpt-bundle diff generation, and tests for missing artifacts/placeholder diffs.
  - Gate status: PAPER
  - Evidence: agent run log `docs/agent_runs/20251221_204343Z_TICKET-007_bundle_diff_hygiene/README.md`; tests `tests/test_gpt_bundle_verifier.py`

## 2025-12-20
- Ticket #1 — Index-only GO/NO-GO gates: scoped quality gates and freshness summaries to index feeds, and index runners now forward the index scope so macro staleness no longer blocks index scans.
  - Gate status: PAPER
  - Evidence: agent run log `docs/agent_runs/20251220_231551Z_TICKET-001_index_only_gates/README.md`
