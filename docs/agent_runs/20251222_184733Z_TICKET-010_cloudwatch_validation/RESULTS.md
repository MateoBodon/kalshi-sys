# Results

Status: PASS

## Summary
- CloudWatch agent config validated on Ubuntu EC2 and agent is running.
- Logs are shipping to `/kalshi/kalshi-supervisor-index` via syslog tailing, with aws logs proof.
- AWS access docs made prominent in runbook + ACCESS.md.
- Updated .gitignore to ignore the entire `docs/` directory per request (tracked docs remain tracked).

## Acceptance Evidence (sanitized)

Command: `amazon-cloudwatch-agent-ctl -a fetch-config -m ec2 -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json -s`
```
Start configuration validation...
... Valid Json input schema.
Configuration validation first phase succeeded
Configuration validation second phase succeeded
Configuration validation succeeded
```

Command: `amazon-cloudwatch-agent-ctl -a status`
```
{"status":"running","configstatus":"configured","version":"1.300062.0b1304"}
```

Command: `aws logs filter-log-events --log-group-name /kalshi/kalshi-supervisor-index --filter-pattern '"kalshi-supervisor-index"' --region us-east-1 --limit 5`
```
"logStreamName": "[REDACTED_INSTANCE_ID]",
"message": "2025-12-22T03:53:36.040613+00:00 [REDACTED_HOST] systemd[1]: Started kalshi-supervisor-index.service - Kalshi Index Supervisor (paper-only default)."
"message": "2025-12-22T03:53:36.047940+00:00 [REDACTED_HOST] systemd[1]: kalshi-supervisor-index.service: Main process exited, code=exited, status=127/n/a"
```

Notes:
- CloudWatch agent schema on Ubuntu only supports file log collection; config tails `/var/log/syslog` (systemd logs forwarded by default).

## Bundle
- docs/gpt_bundles/gpt_bundle_ticket-010_20251222_184733Z_TICKET-010_cloudwatch_validation.zip

## Follow-ups
- None for Ticket #10.
