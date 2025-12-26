# Run README

Goal:
- Ticket #5: AWS-ready runbook + minimal wiring templates for 24/7 supervisor_index with alerts and break-glass steps.

Summary:
- Added AWS supervisor runbook and on-call checks covering secrets handling, health checks, alerting, and break-glass.
- Added systemd supervisor template with dry-run default and restart policy.
- Added supervisor_index CLI aliases for --dry-run and --series filter with tests.
- Stabilized macro-stale index scanner fixture test by pinning a fixed --now and neutralizing clock skew.

Commands:
- See commands.log for full list.

Tests:
- pytest -q
- python -m kalshi_alpha.exec.supervisor_index --help (failed: ModuleNotFoundError)
- PYTHONPATH=src python -m kalshi_alpha.exec.supervisor_index --help

Artifacts:
- docs/runbooks/aws_supervisor_index.md
- docs/runbooks/oncall_checks.md
- deploy/systemd/supervisor_index.service
- src/kalshi_alpha/exec/supervisor_index.py
- tests/exec/test_supervisor_index.py
- tests/test_index_scanner_fixtures.py
- docs/PROGRESS.md
- CHANGELOG.md
