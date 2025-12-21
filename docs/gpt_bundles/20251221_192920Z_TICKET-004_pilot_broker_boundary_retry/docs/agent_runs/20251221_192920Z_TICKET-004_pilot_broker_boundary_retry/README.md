# Run README

Goal:
- Enforce pilot safety at the broker boundary for live submissions (index ladders only) and add required tests for Ticket #4 RETRY.

Summary:
- Tightened pilot config to index-only ladders in `configs/pilot.yaml`.
- Added TOB timestamp extraction + staleness checks in `src/kalshi_alpha/brokers/kalshi/live.py` (`_extract_snapshot_timestamp`, `_resolve_best_bid_ask`, `_enforce_pilot_boundary`) with fail-closed behavior.
- Stamped sniper book snapshots with `book_snapshot_ts` in `src/kalshi_alpha/exec/runners/scan_ladders.py::_evaluate_market`.
- Expanded broker safety tests in `tests/test_broker_live_safety.py` for stale TOB rejection and kill-switch queue suppression.

Exploration (key paths):
- Live broker submit/cancel/replace: `src/kalshi_alpha/brokers/kalshi/live.py` (`place`, `cancel`, `replace`).
- Cancel/replace queue: `src/kalshi_alpha/core/execution/order_queue.py` (`OrderQueue`).
- Proposal → broker calls: `src/kalshi_alpha/exec/runners/scan_ladders.py::execute_broker`.
- Pilot config loader/runtime: `src/kalshi_alpha/exec/pilot/config.py` and `configs/pilot.yaml`.
- Window guard: `src/kalshi_alpha/sched/windows.py::current_window` + freeze buffer checks.
- Kill switch sentinel: `src/kalshi_alpha/exec/heartbeat.py::kill_switch_engaged`.
- TOB source: proposal `book_snapshot` metadata or `KalshiPublicClient.get_orderbook` fetch.

Broker-boundary enforcement (pilot mode):
- Allowed series limited to INX/INXU/NASDAQ100/NASDAQ100U (`INDEX_SERIES`) plus pilot allowlist.
- Max contracts per order and max unique bins enforced across outstanding + new orders.
- Window guard blocks submits outside active window or inside freeze buffer.
- Maker-only guard rejects crossing orders; missing or stale TOB fails closed.
- Kill switch blocks submit/cancel/replace without queuing (prevents queue spam).

Commands:
- See `commands.log`.

Tests:
- `pytest -q` (117 passed, 732 skipped).

Artifacts:
- `docs/agent_runs/20251221_192920Z_TICKET-004_pilot_broker_boundary_retry/diff.patch`
- `docs/agent_runs/20251221_192920Z_TICKET-004_pilot_broker_boundary_retry/TESTS.md`
- `docs/agent_runs/20251221_192920Z_TICKET-004_pilot_broker_boundary_retry/README.md`
- `docs/gpt_bundles/gpt_bundle_TICKET-004_pilot_broker_boundary_20251221_192920Z_TICKET-004_pilot_broker_boundary_retry.zip`

Risks / gaps:
- Maker-only checks require TOB timestamps; missing timestamps force a fresh book fetch, and failures block submits (intentional fail-closed).
- Outstanding-orders metadata must include series/strike/side; missing fields fail closed and block pilot submits.
- Kill switch blocks live submits but does not auto-cancel existing live orders (operator action still required).
- Repo was already dirty (e.g., `AGENTS.md`, `reports/_artifacts/*`); left untouched.
