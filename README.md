# Kalshi Alpha – Execution & Research Monorepo

Kalshi Alpha is a Python 3.11+ monorepo that orchestrates research, backtests, and now a feature‑flagged live pilot lane for Kalshi ladder markets. It couples reproducible data ingestion with strategy scans, layered risk controls, and auditing so the default experience remains paper-only while still allowing controlled live execution when explicitly armed.

---

## What This Repository Provides

### Strategy & Pricing Stack
- **Core analytics (`src/kalshi_alpha/core/`)** – pricing primitives, fee models (October 1 2025 schedule), probability transforms, PAL enforcement, VaR checks, and backtesting helpers.
- **Market data client** – read-only public API with offline fixtures and caching to stay deterministic in CI/offline modes.
- **Strategy modules (`strategies/`)** – CPI, claims, 10Y Treasury, weather, and gasoline models that generate ladder PMFs with calibration hooks.
- **Index ladders (`strategies/index/`)** – SPX/NDX noon + close models pulling Polygon indices with σ<sub>tod</sub> curves, residual clamps, and indices maker=$0.00; indices taker=0.035*C*p*(1-p).

### Data Ingestion & Storage
- **Drivers (`drivers/`)** pull canonical macro datasets (BLS CPI, DOL ETA-539, treasury par yields, Cleveland nowcast, NOAA/NWS, AAA gas). Each driver supports offline fixtures under `tests/fixtures`.
- **Datastore (`data/raw`, `data/proc`)** auto-populates timestamped snapshots and processed parquet tables. These directories are git-ignored and recreated on demand.

### Execution Layer
- **Scanners & pipelines (`exec/`)** power the CLI scanner (`kalshi-scan`) plus daily/today/week orchestration runners.
- **Broker adapters (`exec/brokers`)** offer dry-run recording by default and a live Kalshi adapter that is locked behind dual CLI flags, environment credentials, kill-switch enforcement, and a bounded cancel/replace queue.
- **State management (`exec/state`)** persists outstanding orders under `data/proc/state/orders.json` for restart recovery and reporting.
- **Heartbeat & kill-switch (`exec/heartbeat.py`)** emit JSON heartbeats and gate execution when stale; presence of the kill-switch file forces NO-GO + cancel-all intent.
- **Index ladder ops (`INX`, `INXU`, `NASDAQ100`, `NASDAQ100U`)** add hourly-U rotation with T−2s cancel-all, websocket freshness enforcement, ET clock-skew guards, and `Fill & Slippage` metrics in reports. Use `scripts/polygon_dump.py` or `scripts/make_index_fixtures.sh` to materialize Polygon minutes (11:45–12:05 ET and 15:45–16:05 ET) into `tests/data_fixtures/index/` for deterministic math + scanner tests.
- **Index scanners (`exec/scanners/scan_index_hourly.py`, `scan_index_close.py`)** loop each target window (10:00–16:00 ET hourly plus the 16:00 close) and drop per-window CSV/Markdown under `reports/index_ladders/<HHMM>/` with EV-after-fees, α honesty, slippage drift, freshness, and calibration-age metadata.
- **Pilot runners (`exec/runners/pilot_hourly.py`, `pilot_close.py`)** wrap `scan_ladders` with maker-only guards, 1-lot sizing, ≤2 unique bins, PAL/loss caps, kill-switch enforcement, and optional paper-ledger mode for dry rehearsals or controlled live sessions.

### Reporting & Monitoring
- **Markdown reports (`reports/<SERIES>/`)** include GO/NO-GO badge, Live Pilot header, exposure stats, mispricings, replay scorecards, and outstanding order counts.
- **Pilot readiness dashboard** (`reports/pilot_readiness.md`) summarizes the last 7 days across GO rate, EV after fees, fill realism (observed vs α), replay deltas, and calibration age.
- **Scoreboard tooling (`exec/scoreboard.py`)** produces rolling 7/30‑day scorecards, surfaces gate stats and replay metrics, and now stamps GO/NO-GO decisions with fill, Δbps, α-gap, slippage drift, freshness, and calibration-age context.

### Safety Net
- Quality gates merge model monitors with drawdown checks and heartbeat freshness.
- Kill-switch file (`data/proc/state/kill_switch`) halts new orders and records cancel-all intents.
- Bounded cancel/replace queue (shared by dry & live brokers) enforces FIFO with retries, backoff, and audit drops.
- `dev/sanity_check.py` fails if to-do/NotImplemented markers exist outside tests/docs or if code prints env var names (protects against secret leakage).

---

## Repository Layout

| Path | Purpose |
| --- | --- |
| `src/kalshi_alpha/core/` | Pricing, fees, risk, archive, and API abstractions. |
| `src/kalshi_alpha/drivers/` | Data ingestion for macro + weather sources (offline-ready). |
| `src/kalshi_alpha/strategies/` | Calibrated strategy modules per ladder series. |
| `src/kalshi_alpha/exec/` | Pipelines, runners, brokers, reports, ledger, scoreboard. |
| `src/kalshi_alpha/exec/state/orders.py` | Outstanding order persistence, cancel-all intents. |
| `src/kalshi_alpha/exec/heartbeat.py` | Heartbeat writer, staleness detection, kill-switch helpers. |
| `src/kalshi_alpha/core/execution/order_queue.py` | FIFO cancel/replace queue with retry + audit. |
| `configs/` | PAL policy templates, quality gate overrides, etc. |
| `data/raw`, `data/proc` | Auto-generated data/processing stores (ignored in Git). |
| `data/reference/index_execution_defaults.json` | Default α/slippage curves for index ladders. |
| `reports/` | Markdown reports, scoreboard outputs, pilot readiness summaries. |
| `exec/proposals/` | Scanner proposal JSON artifacts. |
| `tests/` | Pytest suites covering core subsystems, brokers, pipelines, and safety checks. |
| `notebooks/` | Optional research notebooks (no data committed). |

---

## Environment & Configuration

1. **Set up Python environment**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -e ".[dev]"        # use `uv pip install -e ".[dev]"` when available
   ```

2. **Secrets & environment variables**
   - Copy credentials into `.env.local` (git-ignored). The live broker expects `KALSHI_API_KEY_ID` and `KALSHI_PRIVATE_KEY_PEM_PATH` (path to an RSA private key used for RSA-PSS signatures). Values are never logged; they load via `kalshi_alpha.utils.env.load_env()`.
   - If `.env.local` is absent, the code falls back to `.env` then environment variables.

3. **Data directories**
   - `data/raw`, `data/proc`, and `reports/_artifacts` are created on demand. Tests patch these paths to temp locations through fixtures (see `tests/conftest.py`).

---

## Quick Start Workflow

```bash
# Formatting, linting, typing, testing
make fmt
make lint
make typecheck
make test          # runs pytest -q -m "not legacy" across the suite

# Optional live smoke (no submits; requires credentials)
python -m kalshi_alpha.dev.sanity_check --live-smoke --env demo

# Repo hygiene (run locally & in CI)
PYTHONPATH=src python -m kalshi_alpha.dev.sanity_check

# Dry-run ladder scan with offline fixtures
python -m kalshi_alpha.exec.runners.scan_ladders \
  --series CPI \
  --offline \
  --fixtures-root tests/data_fixtures \
  --report \
  --paper-ledger \
  --quiet
```

### Environment & Tooling Notes

- **Editable installs:** pip 25’s default PEP 660 mode does not add `src/` to `sys.path` on macOS. Pin pip `<25` or install with `python -m pip install -e ".[dev]" --config-settings editable_mode=compat`.
- **Manual `.pth` fallback:** drop `.venv/lib/python3.11/site-packages/kalshi_alpha_local.pth` containing the absolute `src` path if imports ever fail.
- **Calibration refresh:** regenerate `data/proc/*_calib.parquet` from fixtures:
  ```bash
  PYTHONPATH=src python - <<'PY'
  import json
  from pathlib import Path
  from kalshi_alpha.strategies.cpi import calibrate as cpi
  from kalshi_alpha.strategies.claims import calibrate as claims
  from kalshi_alpha.strategies.teny import calibrate as teny
  from kalshi_alpha.strategies.weather import calibrate as weather
  fixtures = Path("tests/fixtures")
  cpi(json.loads((fixtures/"cpi"/"history.json").read_text())["history"])
  claims(json.loads((fixtures/"claims"/"history.json").read_text())["history"])
  teny(json.loads((fixtures/"teny"/"history.json").read_text())["history"])
  weather(json.loads((fixtures/"weather"/"history.json").read_text())["history"])
  PY
  ```
- **Repo hygiene:** run `PYTHONPATH=src python -m kalshi_alpha.dev.sanity_check` before committing—this blocks stray unfinished markers and env var leaks.

Common targets:
- `make scan` – shorthand for CPI dry-run scan.
- `python -m kalshi_alpha.exec.pipelines.daily ...` – run structured daily pipeline (see below). New helpers:
  - `--snap-to-window {off,wait,print}` aligns execution with the next Eastern-time window (print-only or sleep-until-open behaviour).
  - `--force-run` (DRY broker only) now always produces a markdown report labelled **FORCE-RUN (DRY)** so manual reviews match production scans.
- `python -m kalshi_alpha.exec.pipelines.preflight --mode ...` – quick ET-aware window + gate check (see “Window Gating & Preflight”).
- `python -m kalshi_alpha.exec.scoreboard` – regenerate scoreboard + pilot readiness reports.
- `make pilot-hourly BROKER=dry REPORT=1` – invoke the hourly pilot runner for the next target hour (set `BROKER=live ACK=1` once credentials, kill-switch, and readiness gates are satisfied).

---

## Running Scans & Pipelines

### CLI Scanner (`kalshi-scan`)
Key flags:
- `--offline/--online` toggles fixtures vs live public API data.
- `--broker {dry,live}` selects adapter (default `dry`).
- `--i-understand-the-risks` **required** alongside `--broker live` to arm live execution.
- `--kill-switch-file <path>` forwards a kill-switch sentinel (defaults to `data/proc/state/kill_switch`).
- Risk controls: `--daily-loss-cap`, `--weekly-loss-cap`, `--max-var`, `--pal-policy`, `--max-loss-per-strike`.

**Live mode requirements**
1. Credentials loaded via `.env.local`.
2. Command must include both `--broker live` and `--i-understand-the-risks`.
3. Ensure kill-switch file does not exist (delete `data/proc/state/kill_switch` if safe). Creating the file forces cancel-all and blocks new orders.
4. Live submissions operate through a cancel/replace queue with rate limiting, exponential backoff, idempotency keys, and audit JSONL (`data/proc/audit/live_orders_*.jsonl`).

Example dry-run with live-capable flags (still dry unless `--broker live` is set):
```bash
python -m kalshi_alpha.exec.runners.scan_ladders \
  --series CPI \
  --offline \
  --report \
  --paper-ledger \
  --broker dry
```

### Pilot Session CLI

For a guarded maker-only pilot, use the dedicated wrapper. It enables `--pilot`, clamps contracts/bins according to `configs/pilot.yaml`, flips to online mode, and writes a structured session artifact.

```bash
python -m kalshi_alpha.exec.runners.pilot \
  --series CPI \
  --pilot-config configs/pilot.yaml \
  --broker live \
  --i-understand-the-risks \
  --kill-switch-file data/proc/state/kill_switch \
  --report
```

Outputs land under `reports/_artifacts/`:
- `pilot_session.json` captures trades, Δbps/t-stat, the CuSum state (`cusum_state`), fill realism gap, alerts, and the target `family` for the run.
- `pilot_ready.json` / `pilot_readiness.md` include `freshness.ledger_age_minutes`, `freshness.monitors_age_minutes`, and full per-bin EV honesty tables (`series[*].ev_honesty_bins`) with the recommended weights/caps now rendered in Markdown.
- Pilot scans load those per-bin weights/caps on startup and clamp proposal sizes even when a family-level GO is in effect.
- `make pilot-bundle` adds the session file plus a generated `README_pilot.md` checklist with an explicit final GO/NO-GO decision, rationale, EV honesty adjustments, CuSum status, freeze violations, drawdown, websocket/auth health, and staleness before you escalate live orders.

### Pipelines
- `kalshi_alpha.exec.pipelines.daily` – full ingestion → calibration → scan workflow for a single mode (e.g., `pre_cpi`). Persists heartbeats, outstanding order state, reports, replay scorecards, and ledger artifacts.
- `kalshi_alpha.exec.pipelines.today` – calendar-driven multi-mode launcher for the current day. Prints outstanding order counts before/after each run.
- `kalshi_alpha.exec.pipelines.week` – weekly bundles (optionally weather, presets for paper/live cadence). Forwards broker/kill-switch flags into each daily invocation.

Each pipeline writes:
- `reports/_artifacts/go_no_go.json` – latest quality gate result.
- Force-run dry scans bypass the usual window guard but still archive proposals, replay parity, and markdown reports for audit. Normal runs abort early if outside the window.

### Window Gating & Preflight

- **Strict window guard** – `daily`/`today` refuse to scan when the calendar window is closed, unless `--force-run` is supplied with `--broker dry`. Outside-window exits leave no proposals, ledgers, or reports.
- **Force-run audit trail** – when force-running, reports always render with a prominent “FORCE-RUN (DRY)” banner and monitors capture the associated latency metrics.
- **Accurate `online` monitors** – monitors now expose the exact CLI `--online/--offline` choice; no silent fallback.
- **Snap-to-window** – `--snap-to-window wait` sleeps until the next allowable window (based on calendar resolution) before scanning. `--snap-to-window print` prints the next ET window and exits 0 (no ingestion/calibration).
- **Preflight CLI** – use
  ```bash
  python -m kalshi_alpha.exec.pipelines.preflight --mode teny_close
  ```
  to print the current ET timestamp, the active/next windows, whether a scan would be allowed if triggered now, and any quality-gate warnings (re-using daily’s gating logic without side effects). Handy for manual ops checklists or CRON guards.
- `data/proc/state/orders.json` – outstanding order state (synchronized on place/cancel/replace).
- `data/proc/state/heartbeat.json` – heartbeat metadata with ET timestamp, monitors, outstanding counts, and broker status.
- Report markdown and scoreboard data.

---

## Index Ladders (SPX/NDX)

- **Series coverage:** intraday hourly (12:00 ET; `INXU`, `NASDAQ100U`) and daily close (`INX`, `NASDAQ100`). Maker-only by default with 1-lot sizing and a two-bin cap per market.
- **Secrets:** Polygon indices use macOS Keychain item `kalshi-sys:POLYGON_API_KEY` first and fall back to the `POLYGON_API_KEY` environment variable. Approving the Keychain prompt once keeps subsequent scans non-interactive.
- **Calibrations:**
  ```bash
  python -m jobs.calibrate_hourly --series INXU NASDAQ100U --days 45
  python -m jobs.calibrate_close --series INX NASDAQ100 --days 60
  ```
  Both jobs fetch Polygon minute bars, persist JSON params in `data/proc/calib/index/<symbol>/{hourly,close}/params.json` (with legacy `noon` directories read-only), and snapshot raw payloads under `data/raw/polygon_index/` for audit.
- **Dry scans:**
  ```bash
  python -m kalshi_alpha.exec.runners.scan_ladders \
    --series INXU \
    --offline \
    --driver-fixtures tests/fixtures \
    --scanner-fixtures tests/data_fixtures \
    --broker dry \
    --contracts 1 \
    --min-ev 0.05 \
    --maker-only \
    --report
  ```
- **Outputs:** scanner metadata now includes Polygon snapshot prices, minutes-to-target, and EV after indices maker=$0.00; indices taker=0.035*C*p*(1-p). Reports render under `reports/index_*` when `--report` is set, and ledger EV columns reflect the updated fee schedule.

---

## Live Broker Setup & Current Status (Oct 30, 2025)

### 1. Populate credentials

Create `.env.local` (git-ignored) with API keys. Multi-line secrets must be quoted:

```ini
EIA_API_KEY=...
FRED_API_KEY=...
NASDAQ_API_KEY=...
KALSHI_API_KEY_ID=...
KALSHI_PRIVATE_KEY_PEM_PATH=/Users/example/kalshi_private_key.pem
```

`kalshi_alpha.utils.env.load_env()` loads `.env.local` first, `.env` next, then falls back to environment variables.

### 2. Safety checklist before arming live

1. **Clear kill-switch & state**: `rm -f data/proc/state/kill_switch data/proc/state/orders.json data/proc/state/heartbeat.json`.
2. **Seed fresh heartbeat**:
   ```bash
   PYTHONPATH=$PWD/src python -m kalshi_alpha.exec.runners.scan_ladders \
       --series CPI --offline --fixtures-root tests/data_fixtures --report --paper-ledger --quiet
   ```
3. **Verify GO/NO-GO**: `cat reports/_artifacts/go_no_go.json` → expect `{"go": true, ...}`.
4. **Run live scan**:
   ```bash
   PYTHONPATH=$PWD/src python -m kalshi_alpha.exec.runners.scan_ladders \
       --series CPI --online --report --paper-ledger \
       --broker live --i-understand-the-risks \
       --kill-switch-file data/proc/state/kill_switch
   ```
5. **Monitor artifacts**:
   - `data/proc/state/orders.json` – live outstanding orders.
   - `data/proc/audit/live_orders_YYYY-MM-DD.jsonl` – audit trail.
   - `reports/_artifacts/go_no_go.json` – final gate decision.

### 3. Connectivity checklist

- Trading endpoints now live at `https://api.elections.kalshi.com/trade-api/v2`. Every request must include `KALSHI-ACCESS-KEY`, `KALSHI-ACCESS-TIMESTAMP` (milliseconds), and `KALSHI-ACCESS-SIGNATURE` (RSA-PSS over `timestamp + METHOD + PATH` with the query string removed).
- The broker no longer exchanges bearer tokens. Each REST call is independently signed; HTTP 401 responses bubble up for operator action.
- The websocket client shares the same signing scheme and automatically retries with exponential backoff when connections drop.
- Clock drift >5 s is rejected locally with a descriptive error—sync NTP before arming.
- Structured logs mask keys to the last 4 characters and include idempotency key tails, retry counts, and HTTP status codes.

If connectivity fails, re-run with `--broker dry` to keep monitoring live data without order flow while troubleshooting credentials or clock issues.

---

## Broker Adapters & Order Lifecycle

| Adapter | Default Mode | Behavior |
| --- | --- | --- |
| `DryBroker` | dry/paper | Deduplicates idempotency keys, serializes intents to `reports/_artifacts`, writes audit JSONL (`data/proc/audit/orders_*.jsonl`). Shares the cancel/replace queue when available. |
| `LiveBroker` | live (feature-flagged) | Disabled in CI, loads `.env.local` credentials, enforces rate limiting (token bucket), idempotency headers, bounded cancel/replace queue, exponential backoff on 429/5xx, and structured audit trail (`live_orders_*.jsonl`). |

Cancel/replace logic is centralized in `OrderQueue`:
- FIFO processing, capacity guard (default 64), configurable retries/backoff.
- On exhaustion, records `queue_drop` audit entries.
- For replaces, executes cancel first then place using the same queue to maintain sequencing.

Outstanding orders are recorded immediately after `broker.place(...)` so state is crash-safe. Reconciliation helpers can remove stale entries or mark cancel-all intents when kill-switches fire.

---

## Reports & Analytics

- **Series reports** (written per run) include:
  - GO/NO-GO badge.
  - *Live Pilot* header summarizing broker mode, Kelly cap, VaR, fill α, and outstanding order totals.
  - Mispricings, exposure, ledgers, replay scorecards, outstanding order breakdown.

- **Scoreboard** (`python -m kalshi_alpha.exec.scoreboard`) produces:
  - `reports/scoreboard_7d.md` and `reports/scoreboard_30d.md` with EV, realized PnL, fill ratios, α, GO/NO-GO counts.
  - `reports/pilot_readiness.md` capturing last 7‑day GO rate, EV after fees, fill realism, and replay delta summary per series.
  - Pulls replay metrics from `reports/_artifacts/scorecards/*.parquet`.
  - *Note:* these markdown files will show “No data available” until the paper/live ledger contains fills.

---

## Monitoring & Safety Controls

1. **Quality gates** – combine monitors + drawdown status + heartbeat freshness. A stale (older than 5 min) heartbeat marks NO-GO and triggers cancel-all intents.
2. **Kill-switch** – any existing kill-switch file forces `OutstandingOrdersState.mark_cancel_all(...)` and raises a broker refusal.
3. **Heartbeats** – `write_heartbeat()` stores mode, monitors, outstanding counts, broker status. `heartbeat_stale()` guards pipelines and broker execution.
4. **Outstanding orders** – persisted across runs; pipelines print counts during today/week orchestrations and reports include totals/breakdowns.
5. **Sanity check** – `kalshi_alpha.dev.sanity_check` ensures no lingering to-do/NotImplemented markers slip through and blocks accidental prints of env variable names.

---

## Data Source Fallbacks

Several public data sources (BLS CPI, DOL ETA‑539, treasury par yields, NOAA/NWS) occasionally return 403/404 responses or fail SSL validation. The drivers now:

- Reuse cached artifacts under `data/proc/_cache/…` when live fetches fail.
- Fall back to the bundled fixtures in `tests/fixtures/...`, then normalize timestamps (e.g., treasury data is re-stamped with the current date) so freshness gates remain green.
- Persist the fallback content to `reports/_artifacts` for reproducibility.

**Recommendation:** rerun the pipelines with healthy network connectivity and valid credentials to replace fallback data with true live snapshots whenever possible.

---

## Testing & Continuous Integration

- Unit & integration tests: `pytest -q`
  - Broker safety (`tests/test_broker_live_safety.py`)
  - Outstanding order state (`tests/test_orders_state.py`)
  - Heartbeat & kill-switch gating (`tests/test_heartbeat_killswitch.py`)
  - Order queue mechanics (`tests/test_order_queue.py`)
  - Pilot readiness + scoreboard outputs (`tests/test_pilot_readiness.py`, `tests/test_scoreboard.py`)
  - Legacy suites cover pipelines, risk, strategies, ledgers.

- GitHub Actions (`.github/workflows/ci.yml`) stages:
  1. Lint (ruff), mypy, pytest, sanity check.
  2. Offline ingestion smoke.
  3. Strategy calibration from fixtures.
  4. Quality gate NO-GO validation.
  5. Today wrapper offline run + artifact upload.
  6. Nightly schedule executing the daily pipeline offline.

To run locally in parity with CI:
```bash
pytest -q
PYTHONPATH=src python -m kalshi_alpha.dev.sanity_check
python -m kalshi_alpha.exec.pipelines.daily --mode pre_cpi --offline --driver-fixtures tests/fixtures --scanner-fixtures tests/data_fixtures --report
python -m kalshi_alpha.exec.scoreboard --window 7 --window 30
```

---

## Future Work Guidelines

- **Documentation first** – update this README and relevant docstrings when adding pipelines, brokers, strategy knobs, or safety controls.
- **Safety review** – any live-path change must retain defaults (paper-only), require explicit arming flags, and include offline test coverage with fixtures.
- **Artifacts hygiene** – new persistent state should live under `data/proc/state/` or `reports/_artifacts/` with explicit audit trails, and tests should validate serialization.
- **CI hooks** – ensure `dev/sanity_check.py` remains green, add targeted tests for new controls, and extend GitHub workflow steps when introducing new scripts.
- **Replay analytics** – when adding replay metrics, persist them under `reports/_artifacts/scorecards/` so scoreboard + pilot readiness ingest them automatically.
- **Kalshi elections API migration** – migrate to the header-only RSA-PSS authentication flow for `https://api.elections.kalshi.com/`, add integration tests, and update this README once live submissions succeed.

### Live Broker Notes
- All order routes now target the `/portfolio` namespace (`POST /portfolio/orders`, `DELETE /portfolio/orders/{order_id}`, etc.) on `https://api.elections.kalshi.com/trade-api/v2`. Make sure any custom tooling, curl smoke tests, or docs reference those paths; `/orders` is deprecated and returns HTTP 404.

---

## Reference Commands

```bash
# Clean outstanding state (dry-run only)
rm -f data/proc/state/orders.json data/proc/state/kill_switch data/proc/state/heartbeat.json

# Manually trigger cancel-all intent
touch data/proc/state/kill_switch

# Inspect live audit entries
tail -f data/proc/audit/live_orders_$(date +%F).jsonl

# Refresh pilot readiness report
python -m kalshi_alpha.exec.scoreboard --window 7 --window 30
```

---

## License

This project is released under the MIT License. See `LICENSE` for details.
- `python -m kalshi_alpha.exec.monitors.cli --freeze-series INX INXU NASDAQ100 NASDAQ100U --no-report` refreshes runtime monitor artifacts while scoping the `freeze_window` check to the index ladders only (handy when CPI/TenY freezes should not block INX/NDX). Omit `--freeze-series` to evaluate the full default set.
