# Results

Status: PASS

## Summary
- EC2 systemd service started cleanly using venv Python and paper-only defaults; no PYTHONPATH hacks.
- Verified ExecStart path and unit file contents; service running after restart.

## Acceptance Evidence (sanitized)

Command: `systemctl cat kalshi-supervisor-index.service`
```
[Service]
WorkingDirectory=/opt/kalshi-sys
User=kalshi
Group=kalshi
Environment=SUPERVISOR_BROKER=dry
EnvironmentFile=/etc/kalshi/kalshi-supervisor.env
ExecStart=/opt/kalshi-sys/.venv/bin/python -m kalshi_alpha.exec.supervisor_index --loop --sleep-seconds 20 --series INXU --dry-run --broker ${SUPERVISOR_BROKER}
```

Command: `systemctl show kalshi-supervisor-index.service -p ExecStart -p FragmentPath -p User --no-pager`
```
ExecStart={ path=/opt/kalshi-sys/.venv/bin/python ; argv[]=/opt/kalshi-sys/.venv/bin/python -m kalshi_alpha.exec.supervisor_index --loop --sleep-seconds 20 --series INXU --dry-run --broker ${SUPERVISOR_BROKER} ; ... }
User=kalshi
FragmentPath=/etc/systemd/system/kalshi-supervisor-index.service
```

Command: `systemctl status kalshi-supervisor-index.service --no-pager`
```
Active: active (running) since Mon 2025-12-22 18:33:35 UTC
Main PID: 191440 (python)
CGroup: /system.slice/kalshi-supervisor-index.service
└─191440 /opt/kalshi-sys/.venv/bin/python -m kalshi_alpha.exec.supervisor_index --loop --sleep-seconds 20 --series INXU --dry-run --broker dry
Dec 22 18:33:36 [REDACTED_HOST] python[191440]: [supervisor_index] 2025-12-22T18:33:36+00:00 skip hourly-1000: past cancel buffer (2025-12-22T09:59:58-05:00)
```

Command: `journalctl -u kalshi-supervisor-index.service -n 200 --no-pager`
```
Dec 22 18:33:36 [REDACTED_HOST] python[191440]: [supervisor_index] 2025-12-22T18:33:36+00:00 skip hourly-1100: past cancel buffer (2025-12-22T10:59:58-05:00)
Dec 22 18:33:36 [REDACTED_HOST] python[191440]: [supervisor_index] 2025-12-22T18:33:36+00:00 skip hourly-1200: past cancel buffer (2025-12-22T11:59:58-05:00)
```

Notes:
- `/etc/kalshi/kalshi-supervisor.env` was created with empty key placeholders (no secrets logged); service still runs in dry-run mode.
- Earlier log entries show prior `/home/ubuntu/kalshi-sys` runs missing scipy/pandas; resolved by venv install on `/opt/kalshi-sys` and service restart.

## Bundle
- docs/gpt_bundles/gpt_bundle_ticket-009_20251222_181604Z_TICKET-009_packaging_systemd_hardening_ec2_verify.zip

## Follow-ups
- Populate real secrets in `/etc/kalshi/kalshi-supervisor.env` for online connectivity when approved.
- Ticket #10 CloudWatch validation remains pending after EC2 proof.
