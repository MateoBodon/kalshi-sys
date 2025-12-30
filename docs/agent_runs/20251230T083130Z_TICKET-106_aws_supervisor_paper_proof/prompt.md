# Prompt

Ticket: TICKET-106 — AWS supervisor wiring proof (PAPER): systemd + CloudWatch + crash recovery drill artifact

Primary requirements:
- Add paper-only systemd unit for supervisor_index (INXU/NASDAQ100U at minimum).
- CloudWatch agent config for supervisor logs (group/stream naming documented).
- Runbooks updated (AWS supervisor + on-call checks).
- Supervisor heartbeat evidence (log line and/or artifact) in dry-run.
- Proof report: reports/ops/aws_supervisor_dryrun_<DATE>.md with systemd/CloudWatch/drill evidence or BLOCKED.
- Run tests (pytest -q) and local paper supervisor smoke run.
- Update docs/PROGRESS.md, CHANGELOG.md, and run log directory.
- Make gpt-bundle with the run log + proof report included.
