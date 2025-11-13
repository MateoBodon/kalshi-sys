# Live Ops Runbook

This runbook captures the current setup for INX/NDX hourly + close ladder trading on Kalshi.

## 1. Data + Calibration

1. **Polygon websocket collector** (Massive indices WS)
   ```bash
   PYTHONPATH=src .venv/bin/python -m kalshi_alpha.exec.collectors.polygon_ws \
       > logs/polygon_ws.log 2>&1 &
   ```
   * Streams snapshots into `data/raw/<YYYY>/<MM>/<DD>/polygon_index/…`.
   * Monitor the process (`ps`/`tail logs/polygon_ws.log`). Restart if it stops; the model otherwise falls back to stale fixtures.

2. **Rebuild calibrations** whenever the websocket restarts:
   ```bash
   PYTHONPATH=src .venv/bin/python -m jobs.calib_hourly --skip-plots
   PYTHONPATH=src .venv/bin/python -m jobs.calib_eod    --skip-plots
   ```
   Output: `data/proc/calib/index_pmf/**` and plots in `reports/calib/**`.

## 2. Continuous Live Scanning

A supervisor loop (`scripts/live_hourly_loop.py`) fires both scan profiles every ~3 minutes:

```bash
nohup .venv/bin/python scripts/live_hourly_loop.py --interval 180 \
    > logs/live_loop.log 2>&1 &
```

* The script injects `<repo>/src` into `PYTHONPATH`, so it works outside the repo root.
* It targets the next hourly window (with a 55-minute cutoff) and automatically adds the 16:00 ET close for INX/NASDAQ100.
* Commands per window:
  - **Wide-net path**: 1× lot, up to 4 bins (`configs/pilot_bins4.yaml`), `impact_cap=0.05`.
  - **Size path**: 2× lots, 2 bins, `impact_cap=0.02`.
  - Both run for INXU and NASDAQ100U hourly; close runs use 2× lots for INX/NASDAQ100 at 16:00.
* Logs: `logs/live_loop.log`. Tail it to confirm scans are succeeding.

### Pilot configs

* Default pilot config (1 lot, 2 bins) already in repository.
* Extra config for 4-bin sweeps: `configs/pilot_bins4.yaml`.

## 3. Guardrails + Overrides

* `--force-gate-pass` is in use only to bypass **macro freshness** alerts. All other fail-closed guards still apply:
  - PAL / drawdown budgets (daily $50, weekly $250).
  - Kill switch file.
  - Polygon WS final-minute freshness (if stale, scans auto-cancel).
* Maker-only and 1–2 lots per quote; `impact_cap` limits help keep submissions away from huge jumps.

## 4. Monitoring & Cancels

1. After each scan cycle, check outstanding orders:
   ```bash
   PYTHONPATH=src .venv/bin/python - <<'PY'
   from kalshi_alpha.exec.state.orders import OutstandingOrdersState
   print(OutstandingOrdersState.load().summary())
   PY
   ```
2. Ensure the exchange has zero resting orders:
   ```bash
   PYTHONPATH=src .venv/bin/python - <<'PY'
   from kalshi_alpha.brokers.kalshi.http_client import KalshiHttpClient
   client = KalshiHttpClient()
   orders = client.get('/portfolio/orders', params={'status': 'resting'}).json().get('orders', [])
   print('resting:', len(orders))
   PY
   ```
3. If anything is still resting before T−2, cancel directly:
   ```bash
   PYTHONPATH=src .venv/bin/python - <<'PY'
   from kalshi_alpha.brokers.kalshi.http_client import KalshiHttpClient
   client = KalshiHttpClient()
   orders = client.get('/portfolio/orders', params={'status': 'resting'}).json().get('orders', [])
   for order in orders:
       client.request('DELETE', f"/portfolio/orders/{order['order_id']}", json_body={})
   PY
   ```
4. Clean the local ledger state (optional) by removing keys from `OutstandingOrdersState` once cancellations succeed.

## 5. Artifacts + Logging

* **Manifests** per scan: `data/raw/kalshi/<date>/<timestamp>/<SERIES>/manifest.json`.
* **Reports**: `reports/<SERIES>/<date>.md`.
* **Live order audit**: `data/proc/audit/live_orders_<YYYY-MM-DD>.jsonl`.
* **Pilot session summary**: `reports/_artifacts/pilot_session.json`.
* **Monitors**: `reports/_artifacts/monitors/*.json`, `go_no_go.json` (will stay NO-GO until macro feeds refresh).
* **Loop + collector logs**: `logs/live_loop.log`, `logs/polygon_ws.log`.
* **Digest**: once there are fills, run `report.digest --date today --engine polars` (add `--skip-slo` if you need a fast pass).

## 6. Common Failure Modes

| Symptom | Fix |
| --- | --- |
| `ModuleNotFoundError: kalshi_alpha` in loop log | Restart `scripts/live_hourly_loop.py` (fixed PYTHONPATH logic). |
| Polygon feed stops writing snapshots | Relaunch `kalshi_alpha.exec.collectors.polygon_ws`. Then rerun `jobs.calib_hourly` and `jobs.calib_eod`. |
| Outstanding orders still resting near T−2 | Use the delete snippet above and confirm `/portfolio/orders?status=resting` returns 0. |
| Macro GO/NO-GO stays red | Expected with current override; to clear it, refresh CPI/claims/AAA feeds or relax the monitors. |

## 7. Scaling knobs

* Raise `--contracts` when PAL limits allow (remember to update pilot config / PAL). Current safe defaults: up to 2 contracts per bin.
* Increase `max_unique_bins` via pilot config (e.g., `configs/pilot_bins4.yaml`).
* Tighten/loosen `--impact-cap` based on how aggressively you want to lean into wide quotes.
* Restore a positive `--min-ev` once the market calms down so we’re not over-trading noise.

## 8. How to stop

* Kill the live loop: find PID via `ps` and `kill <pid>`.
* Stop the Polygon collector similarly.
* Always ensure outstanding orders are canceled before shutting down processes.
