# On-Call Checks (supervisor_index)

Last updated: 2025-12-21

## Quick triage (0-5 minutes)
- Check service status:
  - `systemctl status kalshi-supervisor-index.service`
- Check recent logs for NO-GO / WS stale:
  - `journalctl -u kalshi-supervisor-index.service --since "15 min ago"`
- Check heartbeat age:
  - `data/proc/state/heartbeat.json` should be updated within 5 minutes.
- Check monitor artifacts:
  - newest file under `reports/_artifacts/monitors/*.json` should be < 30 minutes old.
- Check data freshness:
  - `reports/_artifacts/monitors/freshness.json` status should be OK and
    `polygon_index.websocket` age_seconds should be <= 10.

## Standard checks (every shift)
- Run runtime monitors (no secrets in output):
  - `python -m kalshi_alpha.exec.monitors.cli --no-report`
- Refresh data freshness (index-only):
  - `python -m kalshi_alpha.exec.monitors.freshness --config configs/freshness.index.yaml --print`
- Confirm kill switch is not engaged:
  - absence of `data/proc/state/kill_switch`.
- Confirm outstanding orders state is sane:
  - `data/proc/state/orders.json` should have no unexpected live entries.

## Response guide
- Heartbeat stale (>5 min):
  - Restart service, verify windows are active, check Polygon connectivity.
- Monitor artifacts stale (>30 min):
  - Run monitors CLI and check for filesystem write errors.
- Data freshness ALERT or polygon WS stale:
  - Validate POLYGON_API_KEY, network egress, and Polygon websocket logs.
- Repeated NO-GO with `missing_env`:
  - Validate EnvironmentFile content and permissions.
- Crash loop:
  - `systemctl status` for exit code; check recent deploys.

## Break-glass (only if necessary)
- Kill switch:
  - `touch data/proc/state/kill_switch` and restart supervisor.
- Cancel-all (live):
  - Use Kalshi UI/API to cancel orders; `data/proc/state/orders.json` records intent.
- Disable live broker:
  - Set broker to dry in EnvironmentFile and restart service.

## Escalation
- If issues persist past two windows, escalate to the incident lead and start a
  post-mortem draft using `docs/runbooks/postmortem_template.md`.
