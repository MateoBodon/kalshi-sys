# Kalshi Alpha Runbook

## Environment Setup
- Install Python 3.11 and create a virtual environment (recommended via `uv` or `python -m venv`).
- Install project dependencies:
  ```bash
  uv pip install -e .[dev]
  ```
  (If `uv` is unavailable, use `pip install -e .[dev]`.)
- Copy any local credentials into `.env.local` (never commit). The application automatically loads `.env.local` and `.env` via `kalshi_alpha.utils.env`.

## Offline vs. Online Data
- **Offline** mode uses fixtures under `tests/fixtures` (driver data) and `tests/data_fixtures` (public Kalshi payloads). This is the default for CI and testing.
- **Online** mode hits live endpoints. Use `--online` flags for scanner or pipeline commands.
- Ingest CLI examples:
  ```bash
  python -m kalshi_alpha.datastore.ingest --all --offline --fixtures tests/fixtures
  python -m kalshi_alpha.datastore.ingest --all --online --force-refresh
  ```

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

## Interpreting Reports
- **Proposals**: strike, side, contracts, and maker EV per contract.
- **Paper Ledger Summary**: aggregated expected PnL, max loss, trade count using the configured slippage model.
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

## CI Nightly Pipeline
GitHub Actions triggers the offline pipeline nightly:
- Runs `daily.py --mode full --offline` using fixtures.
- Uploads Markdown reports and ledger artifacts as workflow artifacts (`offline-reports`).
- See `.github/workflows/ci.yml` for configuration.
