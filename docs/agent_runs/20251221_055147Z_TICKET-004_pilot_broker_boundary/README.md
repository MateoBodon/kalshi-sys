# Run README

Goal:
- Enforce pilot safety at the broker boundary for live submissions and add required tests for Ticket #4.

Summary:
- Added live broker boundary guards for pilot mode (index-series only, caps, window freeze, maker-only TOB crossing) plus kill-switch submit blocking and explicit ack/env gating.
- Wired execute_broker to pass pilot context, kill-switch path, and optional clock into the live broker.
- Expanded live broker tests with ack/env setup, crossing-order rejection, and kill-switch submit/cancel/replace blocking.

Broker-boundary enforcement (pilot mode):
- Allowed series limited to INX/INXU/NASDAQ100/NASDAQ100U (and pilot config allowlist).
- Max contracts per order and max unique bins enforced across outstanding + new orders.
- Window guard blocks submits outside active window or past freeze buffer.
- Maker-only guard rejects taker liquidity and crossing quotes using TOB snapshot or live orderbook fetch.
- Kill switch blocks submits and avoids cancel/replace queue spam.

Commands:
- See commands.log.

Tests:
- `pytest -q` (117 passed, 730 skipped).

Artifacts:
- `docs/agent_runs/20251221_055147Z_TICKET-004_pilot_broker_boundary/diff.patch`
- `docs/gpt_bundles/gpt_bundle_TICKET-004_pilot_broker_boundary_20251221_055147Z_TICKET-004_pilot_broker_boundary.zip`

Risks / gaps:
- Maker-only checks depend on TOB availability; missing or invalid TOB fails closed and blocks submit.
- Outstanding-orders metadata must include series/strike/side; missing fields fail closed and block pilot submits.
