# Notes

Exploration highlights
- Existing TOB logger wrote `tob.jsonl`/`quote_intents.jsonl` under `data/raw/kalshi/tob/<RUN_ID>/` without compression.
- `scan_ladders` already emits TOB snapshots + quote intents when `--record-tob` is set.
- Housekeeping only pruned top-level roots; telemetry needed recursive pruning.
- Tests were skipped by default unless in `INDEX_ACTIVE_PATTERNS`; telemetry tests now included.

Environment/credentials references (redacted)
- `.env.local` is preferred by `load_env` and contains names only: `KALSHI_API_KEY_ID`, `KALSHI_PRIVATE_KEY_PEM_PATH`, `POLYGON_API_KEY`.
- Kalshi private key PEM path: `~/.kalshi/keys/kalshi_private_key.pem` (chmod 600).
- Polygon API key lookup: Keychain label `kalshi-sys:POLYGON_API_KEY` or `POLYGON_API_KEY` env var.

Run blockers
- Settlement basis audit now uses authenticated trade-api/v2 with KX series mapping; 2025-12-23 audit artifact generated successfully.
- `make calibrate-index` still points at `kalshi_alpha.jobs.*` (module missing); direct jobs under top-level `jobs/` remain the workaround.
- Built `data/proc/index_panel_polygon.parquet` and wrote `data/proc/calib/index_polygon/*/params.json` via `jobs.calibrate_index_polygon_model`.
- Preflight now blocks on `basis_flip_risk:INXU` (fail-closed).
