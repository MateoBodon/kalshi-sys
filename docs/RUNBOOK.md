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
