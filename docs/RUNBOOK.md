# Kalshi Alpha Runbook

## Environment Setup
- Install Python 3.11 and create a virtual environment (recommended via `uv` or `python -m venv`).
- Install project dependencies:
  ```bash
  uv pip install -e .[dev]
  ```
  (If `uv` is unavailable, use `pip install -e .[dev]`.)
- Populate `.env.local` with required credentials. For live Kalshi trading set `KALSHI_API_KEY_ID` and `KALSHI_PRIVATE_KEY_PEM_PATH` (absolute path to the RSA private key used for signing). Never commit secrets; `.env.local` is git-ignored.
- Copy any local credentials into `.env.local` (never commit). The application automatically loads `.env.local` and `.env` via `kalshi_alpha.utils.env`.

### macOS Keychain bootstrap (preferred for local live access)
For long-lived local shells on macOS, store the Kalshi credentials in Keychain and have your shell read them automatically:

```bash
# one-time: move PEM out of the repo and lock permissions
mkdir -p ~/Secure/kalshi
mv ~/Downloads/MyKey.txt ~/Secure/kalshi/private.pem
chmod 600 ~/Secure/kalshi/private.pem

# one-time: record API key ID and PEM path in Keychain
security add-generic-password -a "$USER" -s KalshiAPIKeyID \
  -w 5abb0c59-0ad5-4c5c-bf02-3bbdc0b6f49b
security add-generic-password -a "$USER" -s KalshiPrivateKeyPath \
  -w /Users/<mac-username>/Secure/kalshi/private.pem

# append to ~/.zshrc so new shells auto-export the vars
echo '# Kalshi credentials from macOS Keychain' >> ~/.zshrc
echo 'export KALSHI_API_KEY_ID="$(security find-generic-password -a "$USER" -s KalshiAPIKeyID -w 2>/dev/null)"' >> ~/.zshrc
echo 'export KALSHI_PRIVATE_KEY_PEM_PATH="$(security find-generic-password -a "$USER" -s KalshiPrivateKeyPath -w 2>/dev/null)"' >> ~/.zshrc
echo 'export KALSHI_ENV=prod' >> ~/.zshrc
source ~/.zshrc
```

Replace the placeholder PEM path with your actual location, approve the Keychain prompts with “Always Allow”, and future shells will expose the correct environment variables without re-entering secrets. Rotate keys by updating the two Keychain entries (repeat the `security add-generic-password ... -w NEW_VALUE` commands).

## Offline vs. Online Data
- **Offline** mode uses fixtures under `tests/fixtures` (driver data) and `tests/data_fixtures` (public Kalshi payloads). This is the default for CI and testing.
- **Online** mode hits live endpoints. Use `--online` flags for scanner or pipeline commands.
- Ingest CLI examples:
  ```bash
  python -m kalshi_alpha.datastore.ingest --all --offline --fixtures tests/fixtures
  python -m kalshi_alpha.datastore.ingest --all --online --force-refresh
  ```

## Telemetry Capture
- Live execution emits JSONL telemetry under `data/raw/kalshi/YYYY/MM/DD/exec.jsonl`. Each line contains an UTC timestamp, event type (`sent`, `fill`, `partial_fill`, `reject`, heartbeat events, etc.), and a sanitized payload (signatures/keys masked to the last four characters).
- Use `make telemetry-smoke` to append a synthetic telemetry sequence without touching Kalshi—useful for verifying filesystem permissions and log shipping.
- The sink auto-rotates daily; batch jobs can glob directories by date to collect execution traces for analytics.
- `python -m kalshi_alpha.exec.telemetry.shipper --day yesterday --compress` packages the latest JSONL into `reports/_artifacts/telemetry/`. A systemd unit template is available under `configs/systemd/kalshi-alpha-telemetry.*`.

## Runtime Monitors & Alerts
- Run `make monitors` (or `python -m kalshi_alpha.exec.monitors.cli`) to evaluate EV gap, fill realism, drawdown, websocket health, and auth streaks. JSON artifacts are written to `reports/_artifacts/monitors/` and the summary block in `REPORT.md` is refreshed automatically.
- Configure optional Slack notifications by exporting `KALSHI_MONITOR_SLACK_WEBHOOK`; the CLI posts a compact alert message when any monitor breaches.
- Install the provided systemd unit/timer (`configs/systemd/kalshi-alpha-monitors.*`) to execute monitors every five minutes. Adjust thresholds via CLI flags as needed.
- The monitor suite now includes:
  - `ev_seq_guard`: one-sided CuSum detector on Δbps streams. Any ALERT forces a ramp `NO-GO` and trips the panic-backoff gate.
  - `freeze_window`: enforces the pre-event freeze schedule (CPI T-24h @ 06:00 ET, Claims Wednesday 18:00 ET, TenY 13:30 ET, Weather cycle start). ALERT means submissions must halt until the freeze lifts.
  - Kill-switch and drawdown state are surfaced directly in monitor payloads so the final broker gate has a single source of truth.
- When three or more distinct monitors alert within a 30-minute window the CLI emits a `panic_backoff=true` flag; the ramp policy treats this as a day-long `NO-GO` unless an ops override is signed off.

## Pilot Ramp Policy
- `make pilot-readiness` (wrapper around `python -m kalshi_alpha.exec.reports.ramp`) evaluates ledger performance per series, enforces ramp guardrails, and emits:
- `reports/pilot_ready.json` — machine-readable GO/NO-GO + size multipliers, staleness fields (`freshness.ledger_age_minutes`, `freshness.monitors_age_minutes`), and per-bin EV honesty (`series[*].ev_honesty_bins`) including the recommended weights/caps.
- `reports/pilot_readiness.md` — operator-facing summary table with fills, Δbps, t-stat, guardrail breaches, drawdown state, and a rendered per-bin EV honesty table.
- `python -m kalshi_alpha.exec.runners.pilot --series <ticker> [--pilot-config configs/pilot.yaml] --broker live --i-understand-the-risks` launches the minimal maker-only pilot run. The wrapper snaps args to online mode, clamps contracts/bins per config, respects kill-switches, and writes `reports/_artifacts/pilot_session.json` for ramp review.
- `pilot_session.json` now surfaces `family`, `cusum_state`, `fill_realism_gap`, and the full `alerts_summary` alongside trade counts and Δbps metrics—treat these as the single source of truth for the latest pilot run.
- Pilot scans automatically import the latest per-bin EV honesty weights/caps from `reports/pilot_ready.json` and apply them before placing proposals, even when the family-level decision is GO.
- Thresholds follow the default policy (fills ≥300, Δbps ≥ +6, t-stat ≥ 2.0, zero guardrail breaches, drawdown intact). Override via CLI flags when testing new families.
- For unattended runs, enable the `kalshi-alpha-runner.*` timer; it executes `make report` followed by `make pilot-readiness` each weekday at 05:30 ET.
- Ramp readiness now incorporates freshness and safety gates:
  - Ledger and monitor artifacts must be ≤2 hours old; stale inputs automatically produce `ledger_stale` / `monitors_stale` reasons.
  - Panic backoff (`panic_backoff`) mirrors the runtime monitor aggregation and blocks submission for the trading day.
  - The sequential EV guard adds a per-series CuSum detector; any `sequential_alert` appears in JSON/Markdown and overrides legacy t-stat logic.
  - Freeze windows appear as `freeze_window` reasons when Ops is inside the T-24h/T-18h pre-event freezes. See [Freeze Windows & Pre-Event Policy](#freeze-windows--pre-event-policy).

### Data Freshness Monitor

- `python -m kalshi_alpha.exec.monitors.freshness` (or `make freshness-smoke`) computes the per-feed freshness table and writes `reports/_artifacts/monitors/freshness.json`.
- Required feeds and thresholds:
  - `bls_cpi.latest_release` — release ≤35 days old.
  - `dol_claims.latest_report` — week-ending date ≤8 days old.
  - `treasury_10y.daily` — DGS10 snapshot within 3 business days; series mismatch raises `TENY_SERIES_MISMATCH`.
  - `cleveland_nowcast.monthly` — nowcast ≤35 days old.
  - `aaa_gas.daily` — spot ≤2 days old and price between $2.00 and $6.00.
  - `nws_daily_climate` — ≤2-day-old DCRs for active trading stations only (declare stations in `configs/freshness.yaml`).
- Readiness JSON/Markdown include a “Data Freshness” table sourced from the monitor. Any stale, missing, or out-of-range feed appends `STALE_FEEDS` to the ramp policy and forces `NO-GO`.
- The pre-submit gate (`scan_ladders`) short-circuits before arming proposals whenever `required_feeds_ok` is false, mirroring the readiness decision.

## Ops Glue & Log Rotation
- Systemd unit templates under `configs/systemd/` cover the daily runner, recurring monitors sweep, and telemetry shipper. Deploy with `systemctl enable --now kalshi-alpha-monitors.timer` (and analogous commands for runner/telemetry).
- `configs/logrotate/kalshi-alpha` rotates telemetry JSONL files daily, retaining two weeks of history. Drop the file into `/etc/logrotate.d/` and adjust the absolute paths if the repo lives outside `/opt/kalshi-alpha`.

## Calibration
- Calibrate Tier-1 strategies from fixtures:
  ```bash
  python - <<'PY'
  from pathlib import Path
  import json
  from kalshi_alpha.strategies import cpi, claims, teny, weather
  fixtures = Path("tests/fixtures")
  cpi.calibrate(json.loads((fixtures / "cpi" / "history.json").read_text())["history"])
  claims.calibrate(json.loads((fixtures / "claims" / "history.json").read_text())["history"])
  teny.calibrate(json.loads((fixtures / "teny" / "history.json").read_text())["history"])
  weather.calibrate(json.loads((fixtures / "weather" / "history.json").read_text())["history"])
  PY
  ```
- Calibration artifacts are written under `data/proc/*_calib.parquet` and feed drift monitors.
- Live fill ratios and slippage curves are derived from the aggregated ledger; see [Fill Ratio & Slippage Calibration](#fill-ratio--slippage-calibration) for refresh commands and state locations.

## Daily Pipeline Modes
Run the orchestrator in offline or online mode:
```bash
python -m kalshi_alpha.exec.pipelines.daily \
  --mode pre_cpi \
  --offline \
  --driver-fixtures tests/fixtures \
  --scanner-fixtures tests/data_fixtures \
  --report --paper-ledger
```

Available modes:
- `pre_cpi`: CPI ladder run (T-24h freeze, morning scan).
- `pre_claims`: Weekly jobless claims run (Wednesday freeze at 18:00 ET, Thursday scan window).
- `teny_close`: 10Y Treasury close window (14:30–15:25 ET).
- `weather_cycle`: NOAA/NWS 00/06/12/18Z cycles with settlement the following morning.
- `full`: executes all modes sequentially.

Pipeline steps per run:
1. Datastore ingest (`--offline` fixtures or `--online`).
2. Strategy calibration for CPI, CLAIMS, TNEY, WEATHER.
3. Freeze window validation (logs `scan_notes` if out of window).
4. Ladder scan (maker-only EV, Kelly sizing with VaR/PAL caps).
5. Paper ledger simulation (configurable slippage model, CSV/JSON artifacts under `reports/_artifacts`).
6. Markdown report + JSON log in `data/proc/logs/YYYY-MM-DD/`.

## TENY Close Ops Checklist
- Window: 14:30–15:25 ET. Confirm `--snap-to-window wait` or live clock alignment before moving to maker-only clips. Outside the window run dry scans only.
- Kill switch: ensure `data/proc/state/kill_switch` does **not** exist; if present, halt immediately and page ops.
- Data freshness: `data/proc/treasury_yields/latest.parquet` should carry today’s close timestamp; `data/proc/treasury_yields/daily/*.parquet` must include the current trade date; `data/proc/macro_calendar/latest.parquet` must contain today’s row with any `is_fomc/is_cpi/is_jobs/is_claims` markers.
- Orderbook imbalance: run the websocket smoke (`python -m kalshi_alpha.dev.ws_smoke --tickers TNEY-<contract>`) and confirm new JSON snapshots land under `data/raw/kalshi/orderbook/<ticker>/` and `data/proc/kalshi/orderbook_imbalance/<ticker>.json` updates within two minutes.
- Report ↔ gate: regenerate offline (`python -m kalshi_alpha.exec.pipelines.daily --mode teny_close --offline --report`) and verify the GO/NO-GO badge in `reports/TNEY/.../REPORT.md` matches `reports/_artifacts/go_no_go.json`.
- EV honesty: default shrink is 0.9; override via `--ev-honesty-shrink` on `scan_ladders` or the daily pipeline when ops requests. Monitor block should include `ev_honesty_shrink` plus `ev_shrink` metadata per proposal.

### First Microlive Clip
1. Run the offline pipeline with `--report --paper-ledger`; confirm the markdown summarizes GO with clean EV honesty metrics.
2. Execute `python -m kalshi_alpha.exec.scoreboard --window 7 --window 30` and review trend/fill stats for TNEY.
3. Launch `python -m kalshi_alpha.dev.ws_smoke --tickers TNEY-<today>` for at least five minutes; ensure imbalance JSON updates and no websocket drops are logged.
4. Validate monitors: `python -m kalshi_alpha.exec.monitors.cli --series TNEY` (or `make monitors`) should return all green; heartbeat file `data/proc/state/heartbeat.json` must be <5 min old.
5. Confirm PAL exposure room and set Kelly cap ≤0.25; keep `--maker-only` and 1-lot sizing until ramp sign-off.
6. Flip to live only after repeating the report run with `--online --broker live --paper-ledger` and ensuring kill switch absent.

### Quick Verify Commands
- `python -m kalshi_alpha.exec.pipelines.daily --mode teny_close --offline --report`
- `python -m kalshi_alpha.exec.scoreboard --window 7 --window 30`
- `python -m kalshi_alpha.dev.ws_smoke --tickers TNEY-<contract>`
- `python -m kalshi_alpha.exec.runners.scan_ladders --series TNEY --offline --fixtures-root tests/data_fixtures --ev-honesty-shrink 0.9 --quiet`

## TENY Smoke Harness
1. `python -m kalshi_alpha.dev.ws_smoke --tickers TNEY-<today> --run-seconds 600`
   - Pass: JSONL snapshots appended under `data/raw/kalshi/orderbook/TNEY-<today>/` and `data/proc/kalshi/orderbook_imbalance/TNEY-<today>.json` timestamp refreshes within two minutes.
   - Fail: websocket disconnects, no new snapshot rows, or stale imbalance metric.
2. `python -m kalshi_alpha.exec.pipelines.daily --mode teny_close --online --report --paper-ledger --ev-honesty-shrink 0.9`
   - Pass: GO/NO-GO badge in `reports/TNEY/*.md` matches `reports/_artifacts/go_no_go.json` and monitor block lists `ev_honesty_shrink`.
   - Fail: pipeline exits with NO-GO, missing report, or monitors omit shrink metadata.
3. `python -m kalshi_alpha.exec.scoreboard --window 7 --window 30`
   - Pass: `reports/scoreboard_7d.md`, `reports/scoreboard_30d.md`, and `reports/pilot_readiness.md` regenerate without errors.
   - Fail: scoreboard markdown missing or stale metrics.
4. `python scripts/pilot_teny.py`
   - Pass: script prints paths for the latest TENY report, GO/NO-GO artifact, and scoreboard outputs after running the full smoke sequence.
   - Fail: missing path output or any subprocess non-zero exit.

## Fill Ratio & Slippage Calibration
- Fill alpha values are auto-tuned from the aggregated ledger (`data/proc/ledger_all.parquet`) and persisted to `data/proc/state/fill_alpha.json`. The scanner and daily pipelines automatically load the most recent alpha when `--fill-alpha` is omitted; `--fill-alpha auto` forces a refresh before scanning.
- Depth slippage curves are fitted from the same ledger and stored in `data/proc/state/slippage.json`. When `--slippage-mode depth` is selected the pipelines will hydrate a `SlippageModel` from this state; pass `--impact-cap` to override at runtime.
- Manual refresh example:
  ```bash
  python - <<'PY'
  from kalshi_alpha.core.execution import fillratio, slippage
  fillratio.tune_alpha("CPI", lookback_days=14)
  slippage.fit_slippage("CPI", lookback_days=14)
  PY
  ```

## Scanner Usage
```bash
python -m kalshi_alpha.exec.runners.scan_ladders \
  --series CPI \
  --offline \
  --fixtures-root tests/data_fixtures \
  --output-dir exec/proposals \
  --report --paper-ledger
```
- `--online` switches to live Kalshi market data.
- `--sizing kelly --kelly-cap 0.25` enables truncated Kelly sizing.
- Reports include:
  - Proposal table (maker EV after fees).
  - Paper ledger summary.
  - Monitors block (drift, TZ mismatch, non-monotone ladders).
  - Strategy metadata (AAA gas MTD, suspicious deltas, etc.).

## Common Make Targets
- `make report` regenerates scoreboards from `data/proc/ledger_all.parquet` and writes Markdown to `reports/`.
- `make telemetry-smoke` appends a synthetic execution trace to the telemetry sink for end-to-end log validation.
- `make live-smoke` runs the read-only REST/WebSocket health check (`kalshi_alpha.dev.sanity_check --live-smoke`).

## Live Trading Connectivity
- **Endpoints**: REST `https://api.elections.kalshi.com/trade-api/v2`, WebSocket `wss://api.elections.kalshi.com/trade-api/ws/v2`.
- **Header signing**: every REST call signs `timestamp_ms + METHOD + PATH` (RSA-PSS, SHA-256). Only the path component participates in the signature—query strings are excluded by definition.
- **Headers**: attach `KALSHI-ACCESS-KEY`, `KALSHI-ACCESS-TIMESTAMP` (milliseconds), and `KALSHI-ACCESS-SIGNATURE`; no bearer tokens are exchanged. HTTP 401 responses bubble to operators for manual remediation.
- **Clock guard**: system clock drift greater than 5 s raises a `KalshiClockSkewError` locally—sync with NTP before enabling `--broker live`.
- **Logging**: structured log entries mask sensitive values (only last 4 characters of access keys/Idempotency-Key) and capture retry counts/status codes. Audit artifacts remain under `data/proc/audit/live_orders_*.jsonl`.

## Live Smoke Check
- Run `python -m kalshi_alpha.dev.sanity_check --live-smoke --env demo` to hit `/portfolio/balance` and `/markets` with authenticated headers (no submissions).
- The command reports basic health information and surfaces non-JSON responses or HTTP errors before arming live pipelines.

## Interpreting Reports
- **Proposals**: strike, side, contracts, and maker EV per contract.
- **Paper Ledger Summary**: aggregated expected PnL, max loss, trade count using the configured slippage model.
- **EV Honesty**: scoreboard and ladder reports include expected vs. realized EV bars plus a confidence badge (`✓`, `△`, `✗`) based on sample size and t-statistic.
- **Monitors**:
  - `model_drift`: `True` when rolling CRPS/Brier worsens by >10% vs baseline.
  - `tz_not_et`: `True` if local clock is not ET.
  - `non_monotone_ladders`: count of survival curves that violate monotonicity.
- **Metadata**: includes driver notes (AAA gas deltas, etc.).

## Common Failures & Fixes
- **Missing fixtures**: ensure `--driver-fixtures`/`--scanner-fixtures` paths contain data.
- **Secret detection**: logs are sanitized via `ensure_safe_payload`. A `ValueError` indicates a secret leaked into the payload—redact before retrying.
- **Drift alerts**: refresh calibrations with up-to-date history and validate driver inputs.
- **Non-monotone ladders**: inspect the strategy PMF; adjust variance/grid if needed.
- **TZ mismatch**: verify system timezone or run pipeline on an ET-aligned host.

## Risk Guardrails & Configs
- `configs/pal_policy.yaml` caps CPI ladder exposure (default USD 3k, strike overrides down to USD 2.5k).
- `configs/portfolio.yaml` holds factor volatilities and betas used by the VaR guardrail. Tighten limits by increasing these weights; never lower them without coordination.
- `configs/quality_gates.yaml` encodes minimum CRPS/Brier advantages and sets all monitor tolerances (`tz_not_et`, `non_monotone_ladders`, `negative_ev_after_fees`) to zero.
- CI executes `tests/test_config_guardrails.py` so any attempt to weaken the baselines will fail fast.

## CI Nightly Pipeline
GitHub Actions triggers the offline pipeline nightly:
- Runs `daily.py --mode full --offline` using fixtures.
- Uploads Markdown reports and ledger artifacts as workflow artifacts (`offline-reports`).
- See `.github/workflows/ci.yml` for configuration.

## Freeze Windows & Pre-Event Policy
- **CPI**: Freeze begins T-24h at 06:00 ET the day before release; scans run between 06:00 and release minus 10 minutes.
- **Claims**: Freeze begins Wednesday 18:00 ET even on holiday shifts; Thursday morning scans close 5 minutes before 08:30 ET release.
- **TenY**: Freeze begins 13:30 ET on trading day; window closes at 15:25 ET.
- **Weather**: Freeze aligns with the four-cycle hours (00Z/06Z/12Z/18Z); submissions must start and end inside the 45-minute window.
- The `freeze_window` monitor and ramp report surface active freezes; brokers must respect `NO-GO` until the window clears or Ops issues an explicit override memo.

## Pilot Bundle Workflow & Review Checklist
1. **Pre-flight**
   - `make monitors` and confirm no ALERTs besides known issues.
   - `make pilot-readiness` to refresh `reports/pilot_ready.json` / `pilot_readiness.md`.
   - `make report` (optional) to refresh scoreboards.
2. **Bundle creation**
   - Run `make pilot-bundle` (wrapper for `python -m kalshi_alpha.exec.pilot_bundle`).
   - Default output: `reports/pilot_bundle_YYYYMMDD_HHMMSS.tar.gz` containing pilot readiness JSON/Markdown, `reports/_artifacts/pilot_session.json`, the generated `README_pilot.md` checklist, scoreboards, latest ladder report(s), monitor JSON artifacts (including go/no-go state), and a telemetry slice (≤3 newest `data/raw/kalshi/.../exec.jsonl` files).
   - Inspect `manifest.json` inside the tarball for a machine-readable file list and timestamps.
3. **Review checklist**
   - Confirm `overall.go` vs `overall.no_go`, and scan `overall.global_reasons`, `sequential_alert_series`, and `freeze_violation_series`.
  - Review `README_pilot.md` for the automated checklist—now including the final GO/NO-GO verdict with rationale—plus EV honesty flags, CuSum status, freeze violations, drawdown, WS/auth health, and freshness. Cross-check any `ev_honesty_bins` entries in `pilot_ready.json` for per-bin caps/weights before arming ramps.
   - Review monitor JSON for `panic_backoff`, `ev_seq_guard`, `freeze_window`, and `kill_switch` statuses. Any ALERT requires Ops sign-off before proceeding.
   - Verify ladder Markdown includes latest freeze window notes and sequential stats for the traded series.
   - Check telemetry tail for rejects/auth streaks; if missing, rerun `make monitors` to regenerate artifacts.
4. **Rollback procedure**
   - Touch the kill-switch sentinel (`python -m kalshi_alpha.exec.heartbeat --kill-switch on`) or create `data/proc/state/kill_switch` to hard-stop submissions.
   - rerun `make monitors` and `make pilot-bundle` to capture the stop decision in artifacts.
   - Document the rollback rationale in Ops notes and archive the bundle for audit.
