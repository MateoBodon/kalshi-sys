# Agent Run README

## Goal
- Implement TOB snapshot logging + quote-intent capture for index ladders, plus a fill-calibration dataset builder and docs.

## Summary
- Added bounded TOB snapshot + quote-intent logging wired through `supervisor_index` → `micro_index` → `scan_ladders`, with depth/size caps and window labeling.
- Added `tools/build_fillcalib_dataset.py` to convert snapshots/intents into a calibration-ready parquet table.
- Added fixture-based tests for dataset schema/calcs and TOB bounds, and updated fill-calibration README + progress/changelog.

## Commands
- See `commands.log` for full command history (including failed dry-run attempts and reruns).

## Tests
- `pytest -q` (117 passed, 730 skipped)

## Artifacts
- `data/raw/kalshi/tob/20251221_033013Z_TICKET-003_tob_snapshot_logger/tob.jsonl`
- `data/raw/kalshi/tob/20251221_033013Z_TICKET-003_tob_snapshot_logger/quote_intents.jsonl`
- `data/proc/fillcalib/20251221_033013Z_TICKET-003_tob_snapshot_logger.parquet`
- `reports/fillcalib/README.md`

## Risks
- `--skip-preflight` is required for offline dry-run smoke when calibrations are missing; do **not** use for live runs.
- Offline runs skip pilot enforcement to allow fixture-based smoke; live defaults remain unchanged.
