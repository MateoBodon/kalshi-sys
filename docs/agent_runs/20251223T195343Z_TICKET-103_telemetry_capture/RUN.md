# RUN — TICKET-103

Goal
- Add bounded TOB + quote-intent telemetry capture for index ladders with gzip outputs and retention proof.

Summary
- Added a bounded JSONL telemetry sink with per-window byte caps and gzip output keyed by run_id.
- Updated TOB/quote-intent logging to emit required fields (`run_id`, `window_id`, `series`, `market_ticker`, `ts`) into `data/proc/telemetry/{tob,quote_intents}/<RUN_ID>.jsonl.gz`.
- Added telemetry retention pruning in housekeeping and tests for bounded logging + retention.
- Updated tracked docs for logging/retention + progress/changelog; untracked docs updated locally (PLAN_OF_RECORD, CODEX_SPRINT_TICKETS).
- Patched settlement basis audit to use authenticated Kalshi trade-api/v2 with KX series mapping; generated 2025-12-23 basis artifacts.
- Built `data/proc/index_panel_polygon.parquet` and ran `jobs.calibrate_index_polygon_model` to populate `data/proc/calib/index_polygon/*/params.json`.

Credential placement (redacted; no secrets)
- Environment loading uses `kalshi_alpha.utils.env.load_env`, which reads `.env.local` first, then `.env` (if present).
- `.env.local` contains env var names only (values are redacted): `KALSHI_API_KEY_ID`, `KALSHI_PRIVATE_KEY_PEM_PATH`, `POLYGON_API_KEY`.
- Kalshi private key PEM is stored at `~/.kalshi/keys/kalshi_private_key.pem` (permissions `600`).
- Polygon API key can also be provided via macOS Keychain label `kalshi-sys:POLYGON_API_KEY` as a fallback.

Run attempts
- `tools/settlement_basis_audit.py` originally failed with 401 against Kalshi public markets; updated to authenticated trade-api/v2 and reran successfully.
- `make calibrate-index` failed (module path `kalshi_alpha.jobs` missing); ran `jobs.calibrate_hourly` and `jobs.calibrate_close` directly.
- `polygon_ws` collector ran with `--max-runtime 10`, produced fresh snapshot artifacts; REST fallback returned 404s.
- `supervisor_index --dry-run --record-tob` still NO-GO; latest reason is `basis_flip_risk:INXU` (fail-closed).

Key decisions
- Keep telemetry capture gated behind `--record-tob` (dry-run safe) with default base path `data/proc/telemetry`.
- Enforce per-window caps (256KB per stream) and per-record caps (TOB 10KB, intents 2KB) at write time.

Risks / follow-ups
- Basis flip-risk gate still blocks dry-run; review basis thresholds/inputs if GO is required.
