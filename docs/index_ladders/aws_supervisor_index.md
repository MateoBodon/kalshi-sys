# supervisor_index on AWS (paper runs)

Goal: loop over SPX/NDX windows during U.S. cash hours, enforce preflight + WS freshness, and place **maker-only dry** quotes that log to `data/proc/ledger/index_paper.jsonl`.

## Single-window invocation (cron/EventBridge)
```bash
PYTHONPATH=src \
FAMILY=index \
python -m kalshi_alpha.exec.supervisor_index \
  --now "2025-11-21T09:50:00-05:00"   # any time inside the window
```
- Picks the active or next window and runs `micro_index` for its series.
- Skips trading if preflight fails or Polygon index WS is stale.
- Use `--offline --no-ws-listen` for fixture-only tests.

## Long-lived loop (cash hours on an EC2 box)
```bash
PYTHONPATH=src FAMILY=index \
python -m kalshi_alpha.exec.supervisor_index --loop --sleep-seconds 30
```
- Starts a shared Polygon index websocket listener for freshness gating.
- Re-checks preflight per window; skips if NO-GO or WS age exceeds thresholds (soft 1500ms / strict 800ms by default).
- Broker defaults to `dry`; **do not set `--broker live` without human approval.**
- Example EventBridge rule (UTC, covers EST 09:45–16:05): `cron(0/15 14-21 ? * MON-FRI *)` invoking a Lambda/RunCommand wrapper that calls the supervisor with `--loop --sleep-seconds 30`.

## Operational notes
- Preflight (see `kalshi_alpha.exec.preflight_index.run_preflight`) checks env/keys, kill-switch, calibration freshness, and Polygon REST reachability.
- Kill-switch: pass `--kill-switch-file` to point at a sentinel (default is `data/proc/state/kill_switch`).
- WS gating: when `--offline` or `--no-ws-listen` is set, WS checks are bypassed; otherwise the supervisor opens a single Massive index WS and requires fresh ticks.
- Paper ledger writes can be redirected with `KALSHI_INDEX_PAPER_LEDGER_PATH` for sandbox runs.
- Use `--quiet` to reduce stdout noise in AWS logs.
- Logs emit WS freshness on each window: `fresh WS ok age=XYZms strict=<bool> window=<label>`; grep for `fresh WS ok` in CloudWatch to confirm feed health. Preflight NO-GO reasons are emitted once per window unless transient (`polygon_unreachable`), in which case the supervisor will retry within the same window (default 60s interval).
