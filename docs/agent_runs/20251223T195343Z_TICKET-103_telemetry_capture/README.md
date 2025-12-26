# RUN — TICKET-103

Goal
- Add bounded TOB + quote-intent telemetry capture for index ladders with gzip outputs and retention proof.

Summary
- Added a bounded JSONL telemetry sink with per-window byte caps and gzip output keyed by run_id.
- Updated TOB/quote-intent logging to emit required fields (`run_id`, `window_id`, `series`, `market_ticker`, `ts`) into `data/proc/telemetry/{tob,quote_intents}/<RUN_ID>.jsonl.gz`.
- Added telemetry retention pruning in housekeeping and tests for bounded logging + retention.
- Updated tracked docs for logging/retention + progress/changelog; untracked docs updated locally (PLAN_OF_RECORD, CODEX_SPRINT_TICKETS).

Run attempts
- `supervisor_index --series INXU --dry-run --record-tob` returned NO-GO due to missing calibration artifacts and missing basis audit for INXU, so no TOB/intent telemetry was emitted.
- `collect-polygon-ws` ran via `PYTHON=python3 make collect-polygon-ws` and was interrupted after confirming output (no max runtime in target).

Key decisions
- Keep telemetry capture gated behind `--record-tob` (dry-run safe) with default base path `data/proc/telemetry`.
- Enforce per-window caps (256KB per stream) and per-record caps (TOB 10KB, intents 2KB) at write time.

Risks / follow-ups
- Telemetry pruning relies on periodic `housekeep` runs; add scheduling/monitoring if not already in ops.
- `window_id` is derived from available window metadata; if monitors are missing, fallback uses event timestamp.
