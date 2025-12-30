# Run Summary

Goal: Produce PAPER-safe AWS supervisor wiring proof (systemd + CloudWatch) with heartbeat visibility and a documented crash recovery drill.

Approach:
- Added a paper-only systemd unit template under `configs/systemd/` with venv ExecStart and explicit INX/NDX series scope.
- Added supervisor_index heartbeat cadence/logging + artifact writes for dry-run supervision.
- Updated CloudWatch log config to a dedicated paper log group/stream and refreshed AWS/on-call runbooks.
- Deployed the unit + updated Python modules on the AWS host and restarted the service.
- Installed AWS CLI in the venv to capture CloudWatch log ingestion evidence.
- Captured systemd status, journal heartbeats, CloudWatch log excerpts, and an explicit stop/start recovery drill.
- Extended GPT bundle staging to include `reports/ops/aws_supervisor_dryrun_*.md`.

Key Decisions:
- Keep the systemd unit pinned to `--dry-run` (paper-only) to prevent live trading by default.
- Use syslog tailing for CloudWatch ingestion (journald -> /var/log/syslog on Ubuntu).
- Emit heartbeat logs every 60s via `--heartbeat-seconds` for CloudWatch filtering.

Risks / Notes:
- AWS host repo was behind local changes; synced telemetry modules (`run_metadata.py`, `tob_logger.py`, `telemetry/sink.py`) to avoid import errors.
- CloudWatch log group now uses `/kalshi/kalshi-index-supervisor-paper`; update any downstream alarms accordingly.

Validation:
- `pytest -q` (pass; 124 passed, 746 skipped).
- Local paper smoke run: `python3 -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run --offline --now 2025-12-30T10:50:00-05:00`.
- AWS proof captured in `reports/ops/aws_supervisor_dryrun_2025-12-30.md` (systemd + CloudWatch + crash recovery drill).
