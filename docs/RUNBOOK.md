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
For long-lived local shells on macOS, store the Kalshi credentials in Keychain and have your shell read them automatically. Keep the canonical copy in 1Password (create an “API Credential” item with `KALSHI_API_KEY_ID`, attach the PEM, and restrict it to the trading vault). When you rotate keys:

```bash
# load from 1Password and refresh local cache (example)
op signin <your-signin-shorthand> >/dev/null
op item get "Kalshi – Prod" --field KALSHI_API_KEY_ID --vault "Trading Ops" --reveal |
  xargs -I{} security add-generic-password -a "$USER" -s kalshi-sys:KALSHI_API_KEY_ID -w '{}' -U
op item get "Kalshi – Prod" --field private_key --vault "Trading Ops" --reveal > ~/.kalshi/kalshi_private_key.pem
chmod 600 ~/.kalshi/kalshi_private_key.pem
```

Then make the shell exports persistent:

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

#### FinFeed API credentials (Polygon/Kalshi replay support)
- API key is stored under Keychain service `finfeed`, account `API_KEY`.

  ```bash
  security add-generic-password \
    -s "finfeed" \
    -a "API_KEY" \
    -w "6f0b172f-a985-4d6c-abd4-a247ab915638" \
    -U
  ```

- HMAC signing secret for JWT auth is stored under the same service with account `JWT_SECRET`.

  ```bash
  security add-generic-password \
    -s "finfeed" \
    -a "JWT_SECRET" \
    -w "0BDD700E6155C5D59A3C1E2F238D80923802D29697C36A38D80188677B3453AE" \
    -U
  ```

Retrieve them programmatically with `security find-generic-password -s finfeed -a <ACCOUNT> -w` before generating short-lived Bearer tokens for FinFeed’s REST calls. Never commit the raw values; rotate by re-running the `add-generic-password` command with the new secret.

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

## Live Smoke Check
- Before lining up a live pilot, run the read-only smoke test (requires `KALSHI_API_KEY_ID`, `KALSHI_PRIVATE_KEY_PEM_PATH`, and `KALSHI_ENV=prod`):
  ```bash
  python -m kalshi_alpha.exec.live_smoke
  ```
  Sample success output:
  ```
  [live_smoke] OK 2025-11-04T12:55:00-05:00
    INXU: target H1300 found
    NASDAQ100U: target H1300 found
    outstanding orders: 0 ({'dry': 0, 'live': 0})
  ```
- Use `--json` to emit machine-readable diagnostics. A non-zero exit code highlights missing U-series events, authentication failures, or outstanding orders that still need cancel acknowledgements before requesting live proposals.

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

## Index Ladder Ops Checklist (SPX/NDX)
- Scope: intraday hourly (12:00 ET; `INXU`, `NASDAQ100U`) and daily close (`INX`, `NASDAQ100`). Maker-only, 1-lot sizing, ≤2 bins per market. Runs remain DRY/paper until pilot promotion.
- Secrets: Polygon API key loads from macOS Keychain `kalshi-sys:POLYGON_API_KEY`; fallback env `POLYGON_API_KEY` (CI only). Never commit keys.
- Windows & cancel timers (single source of truth: `configs/index_ops.yaml`):
  - Hourly ladders quote **11:45:00–12:00:00 ET**; cancel/replace all resting quotes by **11:59:58 ET (T−2 s)** per cancel buffer.
  - Close ladders quote **15:50:00–16:00:00 ET**; cancel/replace all resting quotes by **15:59:58 ET (T−2 s)**.
  - Scanners and microlive load the same config; any proposal outside the window is operator error and must be cancelled immediately.
- Fixtures: `scripts/make_index_fixtures.sh` wraps `scripts/polygon_dump.py` to capture Polygon minute bars for `I:SPX`/`I:NDX` over 11:45–12:00 ET and 15:50–16:00 ET windows. Generated Parquets + metadata live under `tests/data_fixtures/index/` and back the math/scanner tests.
- Calibrations (refresh ≤14 days):
  ```bash
  python -m jobs.calibrate_hourly --series INXU NASDAQ100U --days 35
  python -m jobs.calibrate_close  --series INX  NASDAQ100   --days 55
  ```
  Outputs land in `data/proc/calib/index/<symbol>/{hourly,close}/params.json` (legacy `noon` directories remain read-only); raw Polygon payloads are snapshot to `data/raw/polygon_index/`.
- Websocket collector: keep the Massive indices feed (`A.I:SPX`, `A.I:NDX`) running before any window so freshness stays ≤15 s. Options:
  - `make collect-polygon-ws` (foreground burner — exits with Ctrl+C).
  - Install `configs/launchd/kalshi_polygon_ws.plist` (macOS): update `WorkingDirectory`, `PYTHONPATH`, and `ProgramArguments[0]` if your repo or Python path differs, then:
    ```bash
    cp configs/launchd/kalshi_polygon_ws.plist ~/Library/LaunchAgents/
    launchctl unload ~/Library/LaunchAgents/kalshi_polygon_ws.plist 2>/dev/null || true
    launchctl load ~/Library/LaunchAgents/kalshi_polygon_ws.plist
    launchctl start com.kalshi.polygon-ws
    ```
    Logs land in `~/Library/Logs/kalshi_polygon_ws*.log`; use `launchctl stop com.kalshi.polygon-ws` to halt.
  - For Linux, run under systemd using the same entry point (`python -m kalshi_alpha.exec.collectors.polygon_ws`) inside a service unit.
  Use `--max-runtime 300` for smoke tests; omit the flag for production.
- Websocket operations (indices cluster):
  - Trading hours: Massive only streams index aggregates while U.S. cash equities trade (≈09:30–16:00 ET). Outside that window the socket stays open but no ticks post, so the quality gate will read `polygon_ws_stale` even if you’re authenticated. Holidays follow the NYSE schedule.
  - Run cadence: start the collector ~5 minutes before each ops window (11:45–12:00 ET, 15:50–16:00 ET) and shut it down once the window ends. This keeps the single Massive slot free and avoids overnight stale alerts.
  - launchd automation: extend `~/Library/LaunchAgents/kalshi_polygon_ws.plist` with two `StartCalendarInterval` blocks (e.g. `{ "Hour": 11, "Minute": 40 }` and `{ "Hour": 15, "Minute": 45 }`) so macOS brings the feed up ahead of each window. Add matching stop triggers using macOS 15’s `StopCalendarInterval` (set to `12:02` and `16:02`) or a companion plist that runs `launchctl stop com.kalshi.polygon-ws`. After editing, reload with `launchctl bootout gui/$(id -u)/com.kalshi.polygon-ws && launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/kalshi_polygon_ws.plist`.
  - systemd automation: create `kalshi_polygon_ws.timer` with `OnCalendar=Mon-Fri 09:25`, `OnCalendar=Mon-Fri 11:40`, and `OnCalendar=Mon-Fri 15:45`, plus sibling stop timers at `09:35`, `12:02`, and `16:02`. Mask both timers (`systemctl mask --now ...`) on exchange holidays to keep the API key free.
  - Background agent: if you keep a launchd/systemd unit, point it at `.venv/bin/python`, and disable it after the close (`launchctl bootout gui/…` or `systemctl stop`) so ad-hoc smoketests aren’t blocked by `max_connections`.
  - Monitoring: check https://massive.com/system before connecting; a yellow/red “indices” cluster means no point in rebooting the collector. Locally, tail `~/Library/Logs/kalshi_polygon_ws.log` for `auth_success`/`subscribed` and confirm `reports/_artifacts/monitors/freshness.json` shows `age_seconds ≤ 2` once ticks resume.
  - Troubleshooting: `max_connections` means another client is still attached—shut it down and restart. `nodename nor servname provided` usually indicates DNS/connectivity issues; retry after the Massive status page goes green.
- Massive only allows one websocket connection per asset class; keep the launch agent as the canonical feed. If you see `max_connections` or `policy violation` in the log, another client is connected—shut it down, then `launchctl bootout/gui bootstrap` the agent. Our collector now fails fast and logs `[polygon-ws] error: ... max_connections ...` instead of spinning forever.
- Freshness gate: `configs/quality_gates.index.yaml` requires the Massive snapshot age ≤2 s. Any gap longer than that surfaces `polygon_ws_stale` in `reports/_artifacts/go_no_go.json` and the scanners short-circuit proposals. Cross-check `reports/_artifacts/monitors/freshness.json` for `age_seconds` when debugging.
- Quality gates: pass `--quality-gates-config configs/quality_gates.index.yaml` to every scanner and microlive run. The index-only gates enforce Polygon websocket freshness (`max_age_seconds=2`) while ignoring stale macro feeds; treat `polygon_ws_stale` as a hard NO-GO and cancel all orders by the T−2 s buffer baked into `configs/index_ops.yaml`.

### Backtest & Replay Harness
- `make backtest-build START=YYYY-MM-DD END=YYYY-MM-DD` snapshots Polygon minute bars into `data/backtest/index_minutes.parquet`. Each row tags the target ladder with the “on/before” settlement value—if the exact 12:00 ET or 16:00 ET print is missing, we fall back to the most recent prior minute per the exchange rule.
- `make backtest-hourly` and `make backtest-close` load the latest calibrations (`data/proc/calib/index/**/params.json`) and write `reports/backtests/{hourly,close}/{metrics.md,ev_table.csv}` with CRPS/Brier summaries, PIT histograms, and EV-after-fees tables.
- `make replay-yesterday` expects a Massive capture at `data/replay/<DATE>_spx_ndx.json` and replays yesterday’s 11:40–12:05 and 15:45–16:05 ET windows at 10× speed. The command refreshes `reports/_artifacts/monitors/freshness.json` and drops window-specific summaries under `reports/_artifacts/replay/`.
- Index fee truth: makers in `INX*` / `NASDAQ100*` pay $0.00. Takers owe `ceil(0.035 × contracts × price × (1 − price) × 100) / 100`, so 100 contracts at p=0.50 cost $0.88.
- Ops config is centralized in `configs/index_ops.yaml`; the 2 s cancel buffers there are the only supported knobs—never override them in per-run configs.
- Pre-flight:
 1. `python -m kalshi_alpha.exec.heartbeat` → Polygon minute latency ≤30 s (hourly) / ≤20 s (close); websocket tick age ≤2 s; kill-switch absent.
  2. Confirm calibration parquet mtimes ≤14 days.
  3. Dry run (`--offline --broker dry --contracts 1 --min-ev 0.05 --maker-only --report --quality-gates-config configs/quality_gates.index.yaml`) for the target series; inspect markdown for Polygon `snapshot_last_price`, `minutes_to_target`, EV after fees, and confirm GO/NO-GO reasons exclude `polygon_ws_stale`.
- U-series rotation: scans at :40/:55 discover the **next** hour market and emit `[u-roll] ROLLED U-SERIES: HHHMM -> HHHMM`. When the boundary is within two seconds, the runner marks cancel-all before replaying proposals.
- Fees: Indices maker fees are $0.00; indices taker fees use `0.035 × contracts × price × (1 − price)` (see [`docs/kalshi-fee-schedule.pdf`](kalshi-fee-schedule.pdf) for the underlying Polygon-derived curve). EV_after_fees, Δbps, and ledger golden rows must reflect these series-specific curves.
- Target logging: scanner and microlive monitors now emit `ops_timezone=America/New_York`, `ops_target_et`, `ops_target_unix`, and `data_timestamp_used` so operators can confirm DST alignment. When the `'on/before'` fallback from the index rules applies, `ops_target_fallback` is recorded as `"on_before"` and should be cited in the ops log.
- Freshness & clocks (gated by `reports/_artifacts/monitors/freshness.json`): abort when Polygon websocket heartbeat age >2 s, minute aggregates age >30 s (hourly) or >20 s (close), or ET clock skew exceeds 1.5 s. Monitor fields `clock_skew_seconds`/`clock_skew_exceeded` surface in GO/NO-GO artifacts.
- Execution metrics: reports include a `Fill & Slippage` section (mean fill ratio, α target, slippage ticks/USD, EV bps). Defaults seed from `data/reference/index_execution_defaults.json` until the ledger-based stores (`data/proc/state/fill_alpha.json`, `data/proc/state/slippage.json`) are populated.
- First-fill checklist (paper → live ramp):
  1. Immediately after the first live fill, run `python -m kalshi_alpha.exec.scoreboard --window 7 --window 30` to refresh fill counts and EV deltas.
  2. Inspect the newest `reports/_artifacts/*_ledger.json` record for `fills_live`, `max_loss`, and fee lines; archive the JSON alongside operations notes.
  3. Regenerate `python -m kalshi_alpha.exec.pilot_readiness --window 14` and confirm the readiness table reflects the new fill (Δbps ≥6, fills ≥1, freshness OK).
  4. Update the team logbook with timestamp, series, strike, side, broker confirmation ID, and monitor snapshot link.
- Post-run: archive proposals under `exec/proposals/index_*`, keep paper ledger CSV/JSON, regenerate readiness via `python -m kalshi_alpha.exec.scoreboard --window 7 --window 30`.

### First Microlive Clip
1. Run the offline pipeline with `--report --paper-ledger` (add `--quality-gates-config configs/quality_gates.index.yaml` when prepping index microlive); confirm the markdown summarizes GO with clean EV honesty metrics and no `polygon_ws_stale` reason.
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

### Hourly Index Run (13:00 ET — 2025-11-06)
- `PYTHONPATH=src python -m kalshi_alpha.exec.collectors.polygon_ws --symbols I:SPX,I:NDX --freshness-config configs/freshness.index.yaml --freshness-output reports/_artifacts/monitors/freshness.json --max-runtime 120`  
  Captured Massive indices websocket for ~2 min using the index-only freshness profile; wrote `reports/_artifacts/monitors/freshness.json` (status `OK`, polygon feed within 1 s) and parquet heartbeat `data/proc/polygon_index/snapshot_2025-11-04.parquet`. Raw snapshots landed under `data/raw/2025/11/06/polygon_index/20251106T*.json.json` for SPX/NDX + ladder aliases.
- `PYTHONPATH=src:. python -m jobs.calibrate_hourly --symbols I:SPX I:NDX --months 12`  
  Rebuilt hourly calibrations at `data/proc/calib/index/spx/hourly/params.json` and `data/proc/calib/index/ndx/hourly/params.json`.
- `PYTHONPATH=src:. python -m jobs.calibrate_close --symbols I:SPX I:NDX --months 12`  
  Updated close calibrations at `data/proc/calib/index/spx/close/params.json` and `data/proc/calib/index/ndx/close/params.json`.
- `PYTHONPATH=src:. python -m kalshi_alpha.exec.scanners.scan_index_hourly --online --series INXU NASDAQ100U --report --paper-ledger --quality-gates-config configs/quality_gates.index.yaml --target-hour 13`  
  Generated fresh reports `reports/INXU/2025-11-06.md`, `reports/NASDAQ100U/2025-11-06.md` plus proposal bundles in `exec/proposals/{INXU,NASDAQ100U}/2025-11-06*.json` and EV honesty CSVs under `reports/_artifacts/`.
- `PYTHONPATH=src:. python -m kalshi_alpha.exec.ledger.aggregate`  
  Aggregated paper ledger rows into `data/proc/ledger_all.parquet`.
- `PYTHONPATH=src:. python -m kalshi_alpha.exec.scoreboard --window 7 --window 30`  
  Regenerated scoreboards (`reports/scoreboard_7d.md`, `reports/scoreboard_30d.md`), both reflecting `NO-GO` because macro feeds remain stale per freshness monitor.
- `PYTHONPATH=src:. python -m kalshi_alpha.exec.pilot_readiness`  
  Updated `reports/pilot_readiness.md` (all index series `NO-GO`, reasons `insufficient_data` + freshness alert). Retain the latest `reports/_artifacts/pilot_session.json` for auditing.

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
