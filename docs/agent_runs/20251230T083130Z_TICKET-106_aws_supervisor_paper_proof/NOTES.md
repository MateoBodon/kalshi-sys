# Notes

- Existing AWS runbook referenced `deploy/systemd/supervisor_index.service` and syslog tailing for CloudWatch.
- AWS host repo lagged local changes; supervisor_index import failed until telemetry modules were synced.
- AWS CLI was installed in `/opt/kalshi-sys/.venv` to capture CloudWatch log evidence.
