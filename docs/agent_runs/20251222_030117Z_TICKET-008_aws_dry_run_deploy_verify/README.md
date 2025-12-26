# Agent Run README

Goal: Ticket #8 — AWS dry-run deployment verification (EC2 + systemd + CloudWatch proof).

Summary:
- Updated the dry-run systemd unit to pin `--series INXU`.
- Added CloudWatch agent config template plus index monitor/freshness systemd templates.
- Expanded the AWS runbook with copy/paste EC2 setup, systemd, CloudWatch, and redaction steps.
- Added a heartbeat log line in `scan_ladders` for CloudWatch visibility.
- EC2 systemd proof completed for two windows (offline + --now).
- CloudWatch proof captured via log group `/kalshi/kalshi-supervisor-index` after attaching instance role.

Commands:
- `rg -n "kalshi-supervisor-index|systemd|CloudWatch|journalctl|heartbeat.json" -S`
- `pytest -q`
- `PYTHONPATH=src python -m kalshi_alpha.exec.preflight_index --offline --now "2025-12-22T09:50:00-05:00"`
- `PYTHONPATH=src python -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run --offline --now "2025-12-22T09:50:00-05:00"`
- `PYTHONPATH=src python -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run --offline --skip-preflight --no-ws-listen --now "2025-12-22T09:50:00-05:00"`
- `ssh kalshi-aws ...` (systemd install, journals, monitors, CloudWatch agent config)
- `aws sts get-caller-identity` (on EC2 via instance role)
- `aws logs filter-log-events` (CloudWatch proof excerpts)

Tests:
- `pytest -q` (pass; see TESTS.md)

Artifacts:
- `data/proc/state/heartbeat.json` (updated by dry-run supervisor)
- `reports/_artifacts/go_no_go.json` (written by preflight)
- `configs/cloudwatch/kalshi-supervisor-index.json` (template)
