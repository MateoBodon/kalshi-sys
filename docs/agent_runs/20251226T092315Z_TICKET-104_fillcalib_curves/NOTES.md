# NOTES

## Exploration
- Telemetry format: TOB snapshots include `record_type`, `run_id`, `series`, `market_ticker`, `best_bid_price`, `best_ask_price`, `ts_utc`, `window_id` (gzipped JSONL under `data/proc/telemetry/tob/`).
- Quote intents include `record_type`, `series`, `market_ticker`, `quote_side`, `quote_price`, `quote_size`, `tob_ts`, `window_ts_utc`, `window_id` (gzipped JSONL under `data/proc/telemetry/quote_intents/`).
- Existing fill tooling: `tools/build_fillcalib_dataset.py` was a placeholder; `replay/fill_model.py` writes legacy `data/proc/fill/index_fill_curve.json` with series-level probabilities.
- Runtime usage: `fillprob.adjust_alpha` clamps fill alpha in `scan_ladders`; no existing `uncalibrated` flag in scanner output.
