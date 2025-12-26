# Run README

Goal: Ticket #10 — CloudWatch agent config validation + deterministic log shipping.

Summary:
- Updated CloudWatch config to tail /var/log/syslog (Ubuntu default for systemd logs).
- Validated CloudWatch agent config on EC2 and confirmed log shipping via aws logs filter-log-events.
- Added prominent AWS access notes in docs/ACCESS.md and runbook.
- Updated .gitignore to ignore docs/ per request (tracked docs remain tracked).
- Local tests: pytest -q (117 passed, 740 skipped).

Commands:
- See COMMANDS.md and commands.log.

Tests:
- See TESTS.md.
