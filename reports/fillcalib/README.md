# Fill-Calibration Dataset (TOB + Quote Intents)

## Schema (parquet columns)
- run_id
- ts_utc
- series
- window_label
- market_ticker
- bid
- ask
- mid
- spread
- bid_size
- ask_size
- bid_depth (sum of top-level bid sizes, when available)
- ask_depth (sum of top-level ask sizes, when available)
- quote_price
- quote_size
- quote_side
- time_to_expiry_seconds

## Record TOB snapshots (dry-run)

Example (offline fixtures, safe dry-run):

```
PYTHONPATH=src python -m kalshi_alpha.exec.supervisor_index \
  --offline \
  --skip-preflight \
  --record-tob \
  --tob-run-id <RUN_ID> \
  --now <ISO-8601>
```

Outputs:
- `data/raw/kalshi/tob/<RUN_ID>/tob.jsonl`
- `data/raw/kalshi/tob/<RUN_ID>/quote_intents.jsonl`

## Build dataset

```
PYTHONPATH=src python tools/build_fillcalib_dataset.py \
  --in data/raw/kalshi/tob/<RUN_ID> \
  --out data/proc/fillcalib/<RUN_ID>.parquet
```

## Note
This dataset captures inputs for later fill calibration. It does **not** claim realized fills or profitability.
