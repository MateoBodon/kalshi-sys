# On-Call Checks (supervisor_index)

Last updated: 2025-12-30

## Quick triage (0-5 minutes: “is it alive?”)
- Check service status:
  - `systemctl status kalshi-index-supervisor-paper.service`
- Check recent logs for NO-GO / WS stale / heartbeat:
  - `journalctl -u kalshi-index-supervisor-paper.service --since "15 min ago"`
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

## Common failure modes (triage hints)
- Stale data:
  - `reports/_artifacts/monitors/freshness.json` shows stale Polygon WS.
- Stale calibration:
  - `reports/calibration/calibration_ages_<DATE>.md` flags stale files.
- Clock skew:
  - Monitor artifacts or logs show time alignment warnings; verify NTP/chrony.

## Safe restart procedure
1) Confirm no live mode is enabled (paper-only unit).
2) Restart the service:
   - `sudo systemctl restart kalshi-index-supervisor-paper.service`
3) Confirm logs show new `[heartbeat] updated` lines and no duplicate window runs.
4) Re-check heartbeat + monitors freshness.

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
