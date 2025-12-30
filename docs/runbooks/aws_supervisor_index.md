# AWS Supervisor Index Runbook (paper-default)

This runbook describes an AWS-ready, fail-closed deployment for the 24/7
`kalshi_alpha.exec.supervisor_index` loop. Default posture is PAPER/dry-run.

Last updated: 2025-12-30

## AWS access (SSH)
- Quick connect: `ssh kalshi-aws`
- Access details (key path + SSH config host alias) live in `docs/ACCESS.md`.

## Scope and defaults
- Scope: index ladder windows only (INX/INXU/NASDAQ100/NASDAQ100U).
- Default broker: dry (paper). Live must be explicitly armed and is out of scope
  for this runbook unless a separate approval ticket exists.
- Optional: `--series` can restrict the supervisor to a subset for smoke checks.
- Canonical systemd unit: `configs/systemd/kalshi-index-supervisor-paper.service`
  (paper-only default).
  - The template pins `--series INXU NASDAQ100U INX NASDAQ100` for index-only
    scope; adjust only with explicit approval and updated verification evidence.

## Prerequisites
- Repo installed or checked out on the host (example: `/opt/kalshi-sys`).
- Python 3.11+ venv created (example: `/opt/kalshi-sys/.venv`).
- Editable install: `python -m pip install -e .` (if imports fail with pip 25,
  use `--config-settings editable_mode=compat` as noted in `README.md`).

## EC2 bootstrap (copy/paste)
> Use a dedicated non-root user (example: `kalshi`). Run commands as that user
> unless `sudo` is required. Replace placeholders in <>. Reminder: do not echo
> secrets into logs; render them into `/etc/kalshi/kalshi-supervisor.env`.

```bash
# Reminder: never echo secrets into logs; render them into /etc/kalshi/kalshi-supervisor.env.
sudo useradd --create-home --shell /bin/bash kalshi
sudo mkdir -p /opt/kalshi-sys /etc/kalshi
sudo chown -R kalshi:kalshi /opt/kalshi-sys

sudo -u kalshi bash -lc '\
  set -euo pipefail\
  cd /opt/kalshi-sys\
  git clone <REPO_URL> .\
  python -m venv .venv\
  source .venv/bin/activate\
  python -m pip install -U pip wheel\
  python -m pip install -e .\
'

sudo tee /etc/kalshi/kalshi-supervisor.env >/dev/null <<'EOF'
SUPERVISOR_BROKER=dry
POLYGON_API_KEY=<SSM_OR_SECRETS_MANAGER>
KALSHI_API_KEY_ID=<SSM_OR_SECRETS_MANAGER>
KALSHI_PRIVATE_KEY_PEM_PATH=/etc/kalshi/kalshi.pem
EOF
sudo chmod 600 /etc/kalshi/kalshi-supervisor.env

sudo cp /opt/kalshi-sys/configs/systemd/kalshi-index-supervisor-paper.service \
  /etc/systemd/system/kalshi-index-supervisor-paper.service
sudo systemctl daemon-reload
sudo systemctl enable --now kalshi-index-supervisor-paper.service

# Status + (optional) stop/start checks
sudo systemctl status kalshi-index-supervisor-paper.service
sudo systemctl stop kalshi-index-supervisor-paper.service
sudo systemctl start kalshi-index-supervisor-paper.service
```

Notes:
- Keep secrets in SSM/Secrets Manager and render locally; never commit them.

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
  - Optional compatibility flag; paper supervisor is pinned to `--dry-run` in
    `configs/systemd/kalshi-index-supervisor-paper.service` and ignores live
    overrides unless the unit is explicitly edited.
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
   - `configs/systemd/kalshi-index-supervisor-paper.service`
2) Set WorkingDirectory + User/Group placeholders in the unit file.
3) Create an EnvironmentFile (example: `/etc/kalshi/kalshi-supervisor.env`).
   - Include `SUPERVISOR_BROKER=dry` (default) and any required keys.
4) Install and enable the service:
   - `sudo cp configs/systemd/kalshi-index-supervisor-paper.service /etc/systemd/system/kalshi-index-supervisor-paper.service`
   - `sudo systemctl daemon-reload`
   - `sudo systemctl enable --now kalshi-index-supervisor-paper.service`
5) Verify status + logs:
   - `systemctl status kalshi-index-supervisor-paper.service`
   - `journalctl -u kalshi-index-supervisor-paper.service --since "15 min ago"`

Watchdog behavior:
- The service uses `Restart=always` and will self-heal after failures.
- The supervisor exits after the final window; systemd restarts it and keeps a
  24/7 posture.

Monitoring jobs (recommended):
- Use the index-specific templates under `configs/systemd/`:
  - `kalshi-index-monitors.service` + `kalshi-index-monitors.timer`
  - `kalshi-index-freshness.service` + `kalshi-index-freshness.timer`
- Install + enable:
  - `sudo cp configs/systemd/kalshi-index-monitors.service /etc/systemd/system/`
  - `sudo cp configs/systemd/kalshi-index-monitors.timer /etc/systemd/system/`
  - `sudo cp configs/systemd/kalshi-index-freshness.service /etc/systemd/system/`
  - `sudo cp configs/systemd/kalshi-index-freshness.timer /etc/systemd/system/`
  - `sudo systemctl daemon-reload`
  - `sudo systemctl enable --now kalshi-index-monitors.timer`
  - `sudo systemctl enable --now kalshi-index-freshness.timer`

## Log routing (syslog -> CloudWatch)
On Ubuntu, systemd unit logs are forwarded to `/var/log/syslog` by default.
The CloudWatch Agent config in this repo tails that file and ships entries
to the index supervisor log group/stream.

```bash
sudo cp /opt/kalshi-sys/configs/cloudwatch/kalshi-supervisor-index.json \
  /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
  -a fetch-config -m ec2 -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json -s
```

CloudWatch destinations (default template):
- Log group: `/kalshi/kalshi-index-supervisor-paper`
- Log stream: `{instance_id}/kalshi-index-supervisor-paper`

Example signals to index in CloudWatch:
- `NO-GO` lines from supervisor logs.
- `skip ... polygon WS stale` lines.
- `fresh WS ok age=` lines for freshness trending.
- `[heartbeat] updated` lines for heartbeat confirmation.

If the agent is not installed, install it first (example for Ubuntu):
```bash
sudo apt-get update
sudo apt-get install -y amazon-cloudwatch-agent
```

## Health checks and staleness thresholds
Heartbeat:
- Path: `data/proc/state/heartbeat.json`
- Threshold: 5 minutes (see `heartbeat_stale(threshold=timedelta(minutes=5))` in
  `src/kalshi_alpha/exec/runners/scan_ladders.py`).
- Alarm if last heartbeat is older than 5 minutes.
- Supervisor logs `[heartbeat] updated` every ~60s (configurable via
  `--heartbeat-seconds` in the systemd unit).

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

## Crash recovery drill (PAPER)
Goal: prove systemd restart + safe reconciliation for the paper supervisor.

1) Pick a window that has completed and is past the cancel buffer.
2) Stop the service (or kill the process) and confirm it is down:
   - `sudo systemctl stop kalshi-index-supervisor-paper.service`
3) Start the service and confirm it restarts:
   - `sudo systemctl start kalshi-index-supervisor-paper.service`
4) Verify recent logs:
   - `journalctl -u kalshi-index-supervisor-paper.service --since "10 min ago"`
   - Expect new `[heartbeat] updated` lines after restart.
5) Confirm no duplicate window execution:
   - After restart, logs should show either `skip <window>: past cancel buffer`
     or only one `running window ...` per window.
6) Confirm heartbeat artifact advances:
   - `data/proc/state/heartbeat.json` has a newer timestamp.

Safe reconciliation (PAPER) means:
- No live orders are placed (systemd unit pins `--dry-run`).
- Completed windows are not re-executed after the cancel buffer.
- No stale state reuse: the supervisor recomputes window state from current time.

## Break-glass procedures
Kill switch (immediate halt):
1) Create the sentinel file: `touch data/proc/state/kill_switch`.
2) Confirm `kill_switch_engaged` appears in `reports/_artifacts/go_no_go.json`.
3) Remove file to resume (after incident review).

Stop/disable the service:
- `sudo systemctl stop kalshi-index-supervisor-paper.service`
- `sudo systemctl disable kalshi-index-supervisor-paper.service`

Cancel-all (safe cleanup):
- The kill switch triggers `cancel_all` intent in `data/proc/state/orders.json`.
- For live orders, manually cancel via Kalshi UI/API (no built-in CLI yet).
- For dry-run cleanup: `python -m kalshi_alpha.exec.runners.orders_doctor --reconcile --show`.

Disable live broker quickly:
- If running with a live override, set broker back to dry in the EnvironmentFile
  (for example, `SUPERVISOR_BROKER=dry`), then restart the service:
  - `sudo systemctl restart kalshi-index-supervisor-paper.service`
- Live should never be enabled without explicit approval and acknowledgment.

## Local smoke commands
- Dry-run (requires keys):
  - `/opt/kalshi-sys/.venv/bin/python -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run`
- No-keys smoke:
  - `/opt/kalshi-sys/.venv/bin/python -m kalshi_alpha.exec.supervisor_index --help`

## Redaction policy (do not paste into logs/docs)
- AWS account IDs, instance IDs, private IPs, or AMI IDs.
- API keys, PEM contents, or signed request payloads.
- Full CloudWatch log streams; use short sanitized excerpts only.

## Related docs
- `docs/runbooks/oncall_checks.md`
- `docs/runbooks/hourly.md`
- `docs/index_ladders/aws_supervisor_index.md`
