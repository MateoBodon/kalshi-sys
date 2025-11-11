# Post-Mortem Template

Fill out within 24 hours of any NO-GO, outage, or freeze violation. Include links to the relevant `REPORT.md`, monitor artifacts, and ledger snapshots.

```
## Summary
- **Window / Series:** (e.g., Hourly 13:00 ET — INXU)
- **Date:** YYYY-MM-DD
- **Impact:** (missed trades, capital at risk, etc.)

## Timeline (UTC)
- T-XXm — (Event)
- T-00m — (Freeze / outage starts)
- T+YYm — (Failover, cancel-all, kill-switch actions)

## Detection
- How was the issue detected? (monitor alert, manual check, CI failure)
- Which monitors fired? (include snippets from `reports/_artifacts/monitors/*.json`)

## Response
- Actions taken (cancel-all, HotRestart snapshot, dual-feed failover, maintenance coordination)
- Commands / scripts used (`kalshi-scan`, `tools.failover_smoke`, `HotRestartManager.restore`, etc.)

## Root Cause
- Technical root cause (data feed, scheduler bug, broker response, human error)
- Contributing factors

## Corrective Actions
- [ ] Immediate fix
- [ ] Follow-up ticket (link)
- [ ] Runbook update (link / PR)
- [ ] Monitoring / CI enhancement

## Attachments
- Links to `reports/_artifacts/*.json` (freshness, ev_gap, go_no_go)
- Ledger/parquet snippets
- Screenshots if applicable
```
