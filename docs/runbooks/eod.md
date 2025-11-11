# EOD Index Ladder Runbook

## Window Template
- Target: **16:00 ET close**. Scheduler opens quoting at **15:50 ET** (configurable via `configs/index_ops.yaml`) and freezes **2 seconds before** the bell.
- Series: `INX`, `NASDAQ100`.
- `kalshi_alpha.sched.windows.current_window` exposes start/target timestamps; monitors expose the active window under `scheduler_window`.

## Pre-Window Checklist (15:40–15:50 ET)
- [ ] `.env.local` present + loaded (see `.env.local.example`).
- [ ] `reports/_artifacts/monitors/freshness.json` generated within 5 minutes; `polygon_index.websocket` entry present.
- [ ] Review scoreboard freshness metrics (`Polygon WS age`) and ensure data plane GO.
- [ ] Confirm `OutstandingOrdersState` empty and kill switch **not** armed.
- [ ] `python -m kalshi_alpha.exec.runners.scan_ladders --discover --today --offline --series INX` verifies the close ladder is listed before you arm quoting.
- [ ] Run the honesty calc (`python -m report.honesty --window 7 --window 30`) so the EOD clamp in `honesty_window30.json` reflects the latest fills.

## Active Window (15:50 ET → 15:59:58 ET)
- [ ] Run `kalshi-scan --series INX --maker-only --online` (repeat for `NASDAQ100`).
- [ ] Maker-only enforced; `--allow-taker` should remain unset.
- [ ] Monitor `ws_freshness_age_ms` and `ws_final_minute_guard` in monitors. Normal latency < 400 ms; investigate if it spikes.
- [ ] Respect `scheduler_t_minus_2s` — scanner now auto issues cancel-all once freeze time hits.

## Final Minute Guard (15:59:00 ET → close)
- [ ] `ws_final_minute_guard.strict` = `true`.
- [ ] If `ws_freshness_age_ms > 700`, scanner tags `polygon_ws_final_minute_stale`, wipes proposals, and marks `cancel_reason`. Validate the alert, then stand down until feed recovers.
- [ ] Keep Massive dashboard open for real-time visibility; log incidents in `REPORT.md`.

## After Close
- [ ] Verify cancel-all succeeded (`OutstandingOrdersState.total() == 0`).
- [ ] Tag ledger/report with `scheduler_window` metadata for auditing.
- [ ] Regenerate scoreboard + pilot readiness (`make scoreboard`) so Polygon freshness metrics appear in the header.
- [ ] Use the shimmed automation when large calibration/replay runs are needed: `make aws-calib` (hourly σ job) and `make aws-replay FILE=data/replay/YYYY-MM-DD_spx_ndx.json` capture artifacts + metrics under `reports/_artifacts/aws_jobs/`.
