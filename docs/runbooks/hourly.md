# Hourly Index Ladder Runbook

## Window Template
- Targets: **10:00 → 15:00 ET** (top-of-hour). Scheduler opens quoting **15 minutes before** each target and freezes at **T−2s**.
- Series: `INXU`, `NASDAQ100U`.
- Use `kalshi_alpha.sched.windows.current_window(series, now)` to confirm the active window; metadata is emitted to `scheduler_window` in scanner monitors.

## Pre-Window Checklist (T−15m)
- [ ] Ensure `.env.local` exists (see `.env.local.example`) and `python -m kalshi_alpha.exec.runners.scan_ladders --series INXU --maker-only` loads secrets without prompting.
- [ ] `python -m kalshi_alpha.exec.runners.scan_ladders --discover --today --offline --series INXU` confirms Kalshi listed today’s hourly markets before you arm the scheduler.
- [ ] Verify `reports/_artifacts/monitors/freshness.json` was refreshed in the last 5 min. `polygon_index.websocket` `age_seconds` should be `< 2` before arming quoting.
- [ ] Hot-restart snapshot current: `python - <<'PY'\nfrom kalshi_alpha.sched.hotrestart import HotRestartManager\nprint(HotRestartManager().restore())\nPY` should report `age_seconds < 5`. Re-run capture if stale.
- [ ] Regenerate honesty metrics (`python -m report.honesty --window 7 --window 30`) and glance at `reports/_artifacts/honesty/honesty_window7.json`; if `clamp < 0.75`, expect reduced EV sizing.
- [ ] Confirm `OutstandingOrdersState` has **0 live orders** (`python -m kalshi_alpha.exec.state.orders --summary`).
- [ ] Pilot checks: scoreboard GO, `pilot_readiness` GO, loss caps reset.

## Active Window (start → T−2s)
- [ ] Run `kalshi-scan --series INXU --maker-only --online --paper-ledger`.
- [ ] Inspect `scheduler_window` + `ops_window_*` metadata in monitors; ensure `ops_seconds_to_cancel > 2` before allowing broker submission.
- [ ] Maker-only is enforced by default; **never** pass `--allow-taker` during hourly pilots.
- [ ] Confirm `fee_path.maker` points to `configs/fees.json`. If the file is unreadable, abort (scanner will now fail closed).

## Final Minute (T−60s → T−2s)
- [ ] Freshness sentry: `ws_final_minute_guard.strict` will flip to `true`. If `ws_freshness_age_ms > 700`, scanner forces `polygon_ws_final_minute_stale` and issues `cancel_all`.
- [ ] Manually verify the Massive WS dashboard (Polygon portal) if freeze triggers twice in a session.
- [ ] Ensure cancel-all hits before `scheduler_t_minus_2s`. Outstanding orders should read zero at target.

## Failover & Maintenance Hooks
- Dual-feed failover is handled by `kalshi_alpha.data.failover.DualFeedFailover`. Monitor `reports/_artifacts/monitors/ev_gap.json` for `active_feed=massive`. Dry-run via `python -m tools.failover_smoke --dry-run`.
- DST / maintenance windows follow the [Outage / DST / Maintenance Playbook](docs/runbooks/outage_playbook.md). Log any incidents in the new [Post-Mortem Template](docs/runbooks/postmortem_template.md).
- If Kalshi schedules maintenance, issue `OutstandingOrdersState.mark_cancel_all("kalshi_maintenance")`, stop the scheduler, and rely on the hot-restart snapshot after the window.

## Post-Window
- [ ] Confirm `reports/INXU/YYYY-MM-DD.md` captures the `fee_path`, `scheduler_window`, and `ws_freshness` entries.
- [ ] Archive ledger/telemetry via `make ladders-archive` if orders were sent.
- [ ] Update `docs/runbooks/hourly.md` with any anomalies.
- [ ] When calibrations drift past SLA, kick `make aws-calib` (runs the new TOB-aware hourly calibration job in the AWS-compatible shim) and archive the resulting `reports/_artifacts/aws_jobs/*` metrics for the session log.
