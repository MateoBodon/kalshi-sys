# EOD Index Ladder Runbook

## Window Template
- Target: **16:00 ET close**. Scheduler opens quoting at **15:50 ET** (configurable via `configs/index_ops.yaml`) and freezes **2 seconds before** the bell.
- Series: `INX`, `NASDAQ100`.
- `kalshi_alpha.sched.windows.current_window` exposes start/target timestamps; monitors expose the active window under `scheduler_window`.

## Pre-Window Checklist (15:40–15:50 ET)
- [ ] `.env.local` present + loaded (see `.env.local.example`).
- [ ] `reports/_artifacts/monitors/freshness.json` generated within 5 minutes; `polygon_index.websocket` entry present.
- [ ] Run `python monitor/fee_rules_watch.py` (no flags) to ensure it reports `status=OK`; acknowledge changes only after reviewing the official docs.
- [ ] Review scoreboard freshness metrics (`Polygon WS age`) and ensure data plane GO.
- [ ] Hot-restart snapshot age `< 5s` (`python - <<'PY'\nfrom kalshi_alpha.sched.hotrestart import HotRestartManager\nsnap = HotRestartManager().restore()\nprint(snap.age_seconds() if snap else 'missing')\nPY`). If missing/stale, capture before arming quoting.
- [ ] Confirm `OutstandingOrdersState` empty and kill switch **not** armed.
- [ ] `python -m kalshi_alpha.exec.runners.scan_ladders --discover --today --offline --series INX` verifies the close ladder is listed before you arm quoting.
- [ ] Run the honesty calc (`python -m report.honesty --window 7 --window 30`) so the EOD clamp in `honesty_window30.json` reflects the latest fills.
- [ ] Refresh sigma drift monitor (`python monitor/drift_sigma_tod.py --lookback-days 7`); if shrink < 1.0, log the plan before arming quoting.

## Active Window (15:50 ET → 15:59:58 ET)
- [ ] Run `kalshi-scan --series INX --maker-only --online` (repeat for `NASDAQ100`).
- [ ] Maker-only enforced; `--allow-taker` should remain unset.
- [ ] Monitor `ws_freshness_age_ms` and `ws_final_minute_guard` in monitors. Normal latency < 400 ms; investigate if it spikes.
- [ ] Respect `scheduler_t_minus_2s` — scanner now auto issues cancel-all once freeze time hits.

## Final Minute Guard (15:59:00 ET → close)
- [ ] `ws_final_minute_guard.strict` = `true`.
- [ ] If `ws_freshness_age_ms > 700`, scanner tags `polygon_ws_final_minute_stale`, wipes proposals, and marks `cancel_reason`. Validate the alert, then stand down until feed recovers.
- [ ] Keep Massive dashboard open for real-time visibility; log incidents in `REPORT.md`.

## Outages / DST / Maintenance
- Reference `docs/runbooks/outage_playbook.md` for DST tweaks, maintenance windows, and dual-feed failover (run `python -m tools.failover_smoke --dry-run` after status incidents).
- Capture a post-close hot-restart snapshot so overnight recovery stays <5s.
- File a [Post-Mortem](docs/runbooks/postmortem_template.md) for any NO-GO, freeze miss, or extended data outage.

## After Close
- [ ] Verify cancel-all succeeded (`OutstandingOrdersState.total() == 0`).
- [ ] Tag ledger/report with `scheduler_window` metadata for auditing.
- [ ] Replay parity: `python -m tools.replay --families SPX,NDX --date YYYY-MM-DD --hours 16 --epsilon 0.15 --out reports/_artifacts/` followed by `python scripts/parity_gate.py --threshold 0.15 --path reports/_artifacts/replay_ev.parquet --output-json reports/_artifacts/monitors/ev_gap.json`. Review the JSON summary before signing off.
- [ ] Regenerate scoreboard + pilot readiness (`make scoreboard`) so Polygon freshness metrics appear in the header.
- [ ] Generate the daily digest (`python -m report.digest --date YYYY-MM-DD --write --s3 s3://<bucket>/kalshi-sys/reports/`) and include the link in `REPORT.md`.
- [ ] Use the shimmed automation when large calibration/replay runs are needed: `make aws-calib` (hourly σ job) and `make aws-replay FILE=data/replay/YYYY-MM-DD_spx_ndx.json` capture artifacts + metrics under `reports/_artifacts/aws_jobs/`.
