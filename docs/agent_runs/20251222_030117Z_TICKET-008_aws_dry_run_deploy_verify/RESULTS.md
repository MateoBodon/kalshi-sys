Status: COMPLETE (EC2 systemd + CloudWatch proof collected; dry-run only)

What changed:
- Systemd dry-run unit now pins `--series INXU`.
- Added CloudWatch agent config template and index monitor/freshness systemd templates.
- Expanded AWS runbook with EC2 setup, systemd, CloudWatch, redaction, and kill-switch steps.
- Added a heartbeat log line in `scan_ladders` for CloudWatch visibility.

Local verification (dry-run/offline):
- Preflight (offline): `PRECHECK index: NO-GO reasons=4 series=INX,NASDAQ100,INXU,NASDAQ100U`
- Supervisor (offline, required run):
  - `SUPERVISOR preflight: NO-GO reasons=4 series=INXU (broker=dry)`
  - `NO-GO hourly-1000: calibration_missing:...`
- Supervisor (offline + skip-preflight, proof lines):
  - `SUPERVISOR preflight: GO reasons=0 series=INXU (broker=dry)`
  - `running window hourly-1000 for series INXU`
  - `[heartbeat] updated .../data/proc/state/heartbeat.json`

AWS systemd proof (EC2, Ubuntu; sanitized):
- Window 1 (systemd unit + --now 2025-12-22T09:50:00-05:00):
  - `SUPERVISOR preflight: GO reasons=0 series=INXU (broker=dry)`
  - `[supervisor_index] ... running window hourly-1000 for series INXU`
  - `[heartbeat] updated /home/ubuntu/kalshi-sys/data/proc/state/heartbeat.json`
- Window 2 (systemd unit + --now 2025-12-22T10:50:00-05:00):
  - `SUPERVISOR preflight: GO reasons=0 series=INXU (broker=dry)`
  - `[supervisor_index] ... running window hourly-1100 for series INXU`
  - `[heartbeat] updated /home/ubuntu/kalshi-sys/data/proc/state/heartbeat.json`
- Heartbeat timestamp updated: `Modify: 2025-12-22 03:59:59 +0000`
- Monitor artifacts refreshed: `reports/_artifacts/monitors/freshness.json` and related JSONs at `2025-12-22 04:00`.

CloudWatch proof (sanitized excerpts; log group `/kalshi/kalshi-supervisor-index`):
- `... SUPERVISOR preflight: GO reasons=0 series=INXU (broker=dry)` (x2)
- `... running window hourly-1000 for series INXU`
- `... running window hourly-1100 for series INXU`
- `... [heartbeat] updated /home/ubuntu/kalshi-sys/data/proc/state/heartbeat.json` (x2)

Notes:
- CloudWatch agent required an instance role; once attached, `aws sts get-caller-identity` succeeded.
- On this Ubuntu host, the journald CloudWatch config failed schema validation; log shipping used `/var/log/syslog`.

Bundle:
- docs/gpt_bundles/gpt_bundle_TICKET-008_aws_dry_run_deploy_verify_20251222_030117Z_TICKET-008_aws_dry_run_deploy_verify.zip
