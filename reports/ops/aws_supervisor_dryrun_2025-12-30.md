# AWS Supervisor Dry-Run Proof (PAPER) — 2025-12-30

**Status:** DONE — systemd + CloudWatch verification captured on AWS host (sanitized).

## What was deployed (paper-only)
- Systemd unit: `configs/systemd/kalshi-index-supervisor-paper.service`
- Service name on host: `kalshi-index-supervisor-paper.service`
- ExecStart (paper-only):
  - `/opt/kalshi-sys/.venv/bin/python -m kalshi_alpha.exec.supervisor_index --loop --sleep-seconds 20 --heartbeat-seconds 60 --series INXU NASDAQ100U INX NASDAQ100 --dry-run`
- Env file (local, not committed): `/etc/kalshi/kalshi-supervisor.env`
- CloudWatch agent config: `configs/cloudwatch/kalshi-supervisor-index.json`
  - Log group: `/kalshi/kalshi-index-supervisor-paper`
  - Log stream: `{instance_id}/kalshi-index-supervisor-paper`

Deployment notes:
- AWS host required syncing telemetry modules (`run_metadata.py`, `tob_logger.py`, `telemetry/sink.py`) to match repo.
- CloudWatch CLI installed in the venv for log verification.

## Systemd evidence (sanitized)
Command: `sudo systemctl status kalshi-index-supervisor-paper.service --no-pager`

```
● kalshi-index-supervisor-paper.service - Kalshi Index Supervisor (paper-only)
     Loaded: loaded (/etc/systemd/system/kalshi-index-supervisor-paper.service; enabled; preset: enabled)
     Active: active (running) since Tue 2025-12-30 09:08:21 UTC; 4min 29s ago
   Main PID: 261992 (python)
      Tasks: 4 (limit: 4491)
     Memory: 106.8M (peak: 107.0M)
        CPU: 1.050s
     CGroup: /system.slice/kalshi-index-supervisor-paper.service
             └─261992 /opt/kalshi-sys/.venv/bin/python -m kalshi_alpha.exec.supervisor_index --loop --sleep-seconds 20 --heartbeat-seconds 60 --series INXU NASDAQ100U INX NASDAQ100 --dry-run

Dec 30 09:08:21 [REDACTED_HOST] systemd[1]: Started kalshi-index-supervisor-paper.service - Kalshi Index Supervisor (paper-only).
Dec 30 09:08:21 [REDACTED_HOST] python[261992]: [supervisor_index] 2025-12-30T09:08:21+00:00 [heartbeat] updated et=2025-12-30T04:08:21-05:00 ws_age=unknown kill=False
Dec 30 09:09:21 [REDACTED_HOST] python[261992]: [supervisor_index] 2025-12-30T09:09:21+00:00 [heartbeat] updated et=2025-12-30T04:09:21-05:00 ws_age=unknown kill=False
```

## CloudWatch ingestion evidence (sanitized)
Command (region from IMDSv2):

```bash
/opt/kalshi-sys/.venv/bin/aws --region <REGION> logs filter-log-events \
  --log-group-name "/kalshi/kalshi-index-supervisor-paper" \
  --start-time <UNIX_MS> \
  --filter-pattern "[heartbeat]"
```

Excerpt (sanitized):

```
{
    "events": [
        {
            "logStreamName": "[REDACTED_INSTANCE]/kalshi-index-supervisor-paper",
            "timestamp": 1767085578352,
            "message": "2025-12-30T09:06:13.812702+00:00 [REDACTED_HOST] python[261675]: [supervisor_index] 2025-12-30T09:06:13+00:00 [heartbeat] updated et=2025-12-30T04:06:13-05:00 ws_age=unknown kill=False",
            "ingestionTime": 1767085583368
        },
        {
            "logStreamName": "[REDACTED_INSTANCE]/kalshi-index-supervisor-paper",
            "timestamp": 1767085638351,
            "message": "2025-12-30T09:07:13.848527+00:00 [REDACTED_HOST] python[261675]: [supervisor_index] 2025-12-30T09:07:13+00:00 [heartbeat] updated et=2025-12-30T04:07:13-05:00 ws_age=unknown kill=False",
            "ingestionTime": 1767085639054
        }
    ]
}
```

## Crash recovery drill (sanitized)
Commands:

```bash
sudo systemctl stop kalshi-index-supervisor-paper.service
sudo systemctl start kalshi-index-supervisor-paper.service
journalctl -u kalshi-index-supervisor-paper.service --since "5 min ago"
```

Excerpt (sanitized):

```
Dec 30 09:08:08 [REDACTED_HOST] systemd[1]: Stopping kalshi-index-supervisor-paper.service - Kalshi Index Supervisor (paper-only)...
Dec 30 09:08:08 [REDACTED_HOST] systemd[1]: kalshi-index-supervisor-paper.service: Deactivated successfully.
Dec 30 09:08:08 [REDACTED_HOST] systemd[1]: Stopped kalshi-index-supervisor-paper.service - Kalshi Index Supervisor (paper-only).
Dec 30 09:08:21 [REDACTED_HOST] systemd[1]: Started kalshi-index-supervisor-paper.service - Kalshi Index Supervisor (paper-only).
Dec 30 09:08:21 [REDACTED_HOST] python[261992]: [supervisor_index] 2025-12-30T09:08:21+00:00 [heartbeat] updated et=2025-12-30T04:08:21-05:00 ws_age=unknown kill=False
Dec 30 09:09:21 [REDACTED_HOST] python[261992]: [supervisor_index] 2025-12-30T09:09:21+00:00 [heartbeat] updated et=2025-12-30T04:09:21-05:00 ws_age=unknown kill=False
```

Safe reconciliation (PAPER) confirmed:
- No live orders are placed (unit pins `--dry-run`).
- Restart occurs cleanly and heartbeats resume.
- No duplicate window execution observed in logs during the restart window (only heartbeats during off-window time).

## Local smoke run evidence (non-AWS)
This local paper-only run completed and emitted a heartbeat artifact:

```bash
python3 -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run --offline --now 2025-12-30T10:50:00-05:00
```

Sanitized excerpt:
- `SUPERVISOR preflight: GO reasons=0 series=INXU (broker=dry)`
- `[supervisor_index] ... running window hourly-1100 for series INXU`
- `[supervisor_index] ... [heartbeat] updated et=2025-12-30T03:40:38-05:00 ws_age=unknown kill=False`
