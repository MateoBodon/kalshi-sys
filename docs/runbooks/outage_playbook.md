# Outage / DST / Maintenance Playbook

This playbook supplements the hourly/EOD runbooks with specific procedures for dual-feed data outages, DST transitions, and Kalshi maintenance windows.

## Dual-Feed Data Outage
1. **Detect:** `reports/_artifacts/monitors/freshness.json` and the new `reports/_artifacts/monitors/ev_gap.json` include `required_feeds_ok` plus ΔEV parity drift. Alert when `data_freshness.polygon_ws.stale` or `dual_feed.active=massive` persists for >2 windows.
2. **Failover:** The data plane now runs `kalshi_alpha.data.failover.DualFeedFailover` on the Massive (Polygon) indices feed. When `polygon` exceeds the configured freshness SLO, `active_feed` flips to `massive`. Confirm via monitor payloads or `python -m tools.failover_smoke --dry-run` to rehearse the path offline.
3. **Mitigation:** If both feeds degrade, stand down quoting, log the timestamp in `REPORT.md`, and open a Polygon ticket with the `active_feed` snapshots.
4. **Recovery:** Once `polygon` age < SLO for 5 seconds, the failover controller auto-recovers; confirm before rearming the scheduler.

## DST Transition Checklist
1. Run `python -m kalshi_alpha.sched.windows --series INXU --now "YYYY-MM-DDTHH:MM:SS-04:00"` for the first post-shift session to verify `scheduler_window.start_et` and `freeze_et` align with expectations.
2. Update `docs/runbooks/hourly.md` / `docs/runbooks/eod.md` with the DST-specific start offsets if the exchange publishes special hours.
3. Capture a hot-restart snapshot (`HotRestartManager.capture`) immediately after DST activation so operators can validate recovery paths.

## Kalshi Maintenance Windows
1. Maintenance windows are announced under Statuspage; add the slot to `REPORT.md` and arm `OutstandingOrdersState.mark_cancel_all("kalshi_maintenance")` 5 minutes prior.
2. Run `python -m kalshi_alpha.exec.housekeep --keep-days 30` before downtime so ledger/monitor artifacts remain compact.
3. After maintenance ends, run `python -m kalshi_alpha.exec.runners.scan_ladders --discover --today` to verify listings, then restart the scheduler. Use `HotRestartManager.restore()` (max age 5s) to validate that state snapshots remained fresh.

## Hot-Restart Drill
- Command: `python - <<'PY'` snippet invoking `HotRestartManager.capture(...)` and `restore()` is documented in `docs/runbooks/hourly.md`.
- Recovery target: restart quoting < 5 seconds using the persisted active-window metadata and outstanding-order summary.
