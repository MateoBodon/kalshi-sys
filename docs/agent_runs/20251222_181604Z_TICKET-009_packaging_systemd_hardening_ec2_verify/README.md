# Run README

Goal: Ticket #9 RETRY — EC2 verification of venv-based systemd unit (no PYTHONPATH hacks).

Summary:
- Verified EC2 systemd unit uses /opt/kalshi-sys/.venv/bin/python and runs in dry-run.
- Captured sanitized systemd/journalctl proof in RESULTS.md.
- Local tests: pytest -q (117 passed, 740 skipped).

Commands:
- See COMMANDS.md and commands.log.

Tests:
- See TESTS.md.
