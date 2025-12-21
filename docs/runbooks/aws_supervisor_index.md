# AWS Supervisor Index Runbook (paper-default)

This runbook describes an AWS-ready, fail-closed deployment for the 24/7
`kalshi_alpha.exec.supervisor_index` loop. Default posture is PAPER/dry-run.

Last updated: 2025-12-21

## Scope and defaults
- Scope: index ladder windows only (INX/INXU/NASDAQ100/NASDAQ100U).
- Default broker: dry (paper). Live must be explicitly armed and is out of scope
  for this runbook unless a separate approval ticket exists.
- Optional: `--series` can restrict the supervisor to a subset for smoke checks.

## Prerequisites
- Repo installed or checked out on the host (example: `/opt/kalshi-sys`).
- Python 3.11+ environment activated.
- Either:
  - `pip install -e .`, or
  - set `PYTHONPATH=src` in the EnvironmentFile.

## Environment variables and secrets (SSM / Secrets Manager)
Store secrets in AWS SSM Parameter Store or Secrets Manager and render into a
local EnvironmentFile (example: `/etc/kalshi/kalshi-supervisor.env`).
Do NOT store secrets in the repo.

Required for online supervisor (preflight + discovery):
- `POLYGON_API_KEY`
  - Used by Polygon REST ping and websocket readers.
  - Loaded via `kalshi_alpha.utils.keys.load_polygon_api_key()`.
- `KALSHI_API_KEY_ID`
  - Used by `kalshi_alpha.brokers.kalshi.http_client.KalshiHttpClient`.
- `KALSHI_PRIVATE_KEY_PEM_PATH`
  - Path to RSA private key PEM for Kalshi auth.

Optional / ops:
- `SUPERVISOR_BROKER`
  - Used by `deploy/systemd/supervisor_index.service` to set broker mode.
  - Default is `dry`; override only with explicit approval.
- `KALSHI_MONITOR_SLACK_WEBHOOK`
  - Used by `kalshi_alpha.exec.monitors.cli` for Slack alerts.
- `KALSHI_INDEX_PAPER_LEDGER_PATH`
  - Overrides default paper ledger path.

## Configs and artifact paths
Core configs:
- `configs/index_ops.yaml` (window definitions + cancel buffers).
- `configs/quality_gates.index.yaml` (index-only freshness thresholds).
- `configs/freshness.index.yaml` (index-only data freshness monitor).
- `configs/pilot.yaml` (pilot safety caps used by index scanners).

Runtime artifacts:
- Heartbeat: `data/proc/state/heartbeat.json`
- Kill switch: `data/proc/state/kill_switch`
- Outstanding orders: `data/proc/state/orders.json`
- Monitor artifacts: `reports/_artifacts/monitors/*.json`
- Freshness monitor: `reports/_artifacts/monitors/freshness.json`
- GO/NO-GO summary: `reports/_artifacts/go_no_go.json`

## Deployment (systemd on EC2)
1) Copy the systemd unit template:
   - `deploy/systemd/supervisor_index.service`
2) Set WorkingDirectory + User/Group placeholders.
3) Create an EnvironmentFile (example: `/etc/kalshi/kalshi-supervisor.env`).
   - Include `SUPERVISOR_BROKER=dry` (default) and any required keys.
4) Install and enable the service:
   - `sudo cp deploy/systemd/supervisor_index.service /etc/systemd/system/kalshi-supervisor-index.service`
   - `sudo systemctl daemon-reload`
   - `sudo systemctl enable --now kalshi-supervisor-index.service`

Watchdog behavior:
- The service uses `Restart=always` and will self-heal after failures.
- The supervisor exits after the final window; systemd restarts it and keeps a
  24/7 posture.

Monitoring jobs (recommended):
- Reuse the existing templates under `configs/systemd/` for runtime monitors.
  - `kalshi-alpha-monitors.service` runs `python -m kalshi_alpha.exec.monitors.cli`.
  - `kalshi-alpha-monitors.timer` runs every 5 minutes.
- Add an index-only freshness cron/timer:
  - `python -m kalshi_alpha.exec.monitors.freshness --config configs/freshness.index.yaml`

## Log routing (journald -> CloudWatch)
- Configure the CloudWatch Agent to collect systemd/journald logs for
  `kalshi-supervisor-index.service`.
- Example signals to index in CloudWatch:
  - `NO-GO` lines from supervisor logs.
  - `skip ... polygon WS stale` lines.
  - `fresh WS ok age=` lines for freshness trending.

## Health checks and staleness thresholds
Heartbeat:
- Path: `data/proc/state/heartbeat.json`
- Threshold: 5 minutes (see `heartbeat_stale(threshold=timedelta(minutes=5))` in
  `src/kalshi_alpha/exec/runners/scan_ladders.py`).
- Alarm if last heartbeat is older than 5 minutes.

Monitor artifacts:
- Path: `reports/_artifacts/monitors/*.json`
- Staleness threshold: 30 minutes (DEFAULT_MONITOR_MAX_AGE_MINUTES in
  `src/kalshi_alpha/exec/monitors/summary.py`).
- Alarm if the newest monitor artifact is older than 30 minutes.

Data freshness (index scope):
- Config: `configs/freshness.index.yaml`
- Polygon index websocket age_seconds threshold: 10 seconds.
- Use `python -m kalshi_alpha.exec.monitors.freshness --config configs/freshness.index.yaml`
  on a 5-minute cadence to refresh `freshness.json`.

Supervisor WS gating:
- Default thresholds: soft 1500 ms, strict 800 ms
  (`DEFAULT_WS_SOFT_MS`, `DEFAULT_WS_STRICT_MS` in
  `src/kalshi_alpha/exec/supervisor_index.py`).
- Alarm if repeated `skip <window>: polygon WS stale` messages appear.

Preflight calibration freshness:
- Max age: 14 days (`MAX_CALIBRATION_AGE_DAYS` in
  `src/kalshi_alpha/exec/preflight_index.py`).
- Alarm if calibration files under `data/proc/calib/index/*/*/params.json`
  exceed 14 days.

## Alert conditions (minimum set)
- Heartbeat stale > 5 minutes.
- Monitor artifacts stale > 30 minutes.
- Freshness artifact reports `status=ALERT` or `polygon_index.websocket` age > 10s.
- Supervisor logs contain `NO-GO` or `polygon_ws_stale` for more than 2 windows.
- systemd restarts > 3 within 10 minutes (crash loop).

## Break-glass procedures
Kill switch (immediate halt):
1) Create the sentinel file: `touch data/proc/state/kill_switch`.
2) Confirm `kill_switch_engaged` appears in `reports/_artifacts/go_no_go.json`.
3) Remove file to resume (after incident review).

Cancel-all (safe cleanup):
- The kill switch triggers `cancel_all` intent in `data/proc/state/orders.json`.
- For live orders, manually cancel via Kalshi UI/API (no built-in CLI yet).
- For dry-run cleanup: `python -m kalshi_alpha.exec.runners.orders_doctor --reconcile --show`.

Disable live broker quickly:
- If running with a live override, set broker back to dry in the EnvironmentFile
  (for example, `SUPERVISOR_BROKER=dry`), then restart the service:
  - `sudo systemctl restart kalshi-supervisor-index.service`
- Live should never be enabled without explicit approval and acknowledgment.

## Local smoke commands
- Dry-run (requires keys):
  - `python -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run`
- No-keys smoke:
  - `python -m kalshi_alpha.exec.supervisor_index --help`

## Related docs
- `docs/runbooks/oncall_checks.md`
- `docs/runbooks/hourly.md`
- `docs/index_ladders/aws_supervisor_index.md`
