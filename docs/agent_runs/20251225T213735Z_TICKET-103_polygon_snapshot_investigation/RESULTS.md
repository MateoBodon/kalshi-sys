# Results

## Findings
- The REST fallback path in `PolygonIndicesClient.fetch_snapshot` was updated from the undocumented v2 indices path to the documented v3 indices snapshot endpoint.
- Recent run logs on **2025-12-23** recorded REST fallback warnings with 404s; no local evidence of a successful fallback response was found.
- Polygon/Massive vendor docs list **indices snapshots** under **v3** (`GET /v3/snapshot/indices`) and include `market_status` + `timeframe` fields for closed/real‑time/delayed state.
- Vendor docs also list market-status endpoints (`/v1/marketstatus/now`, `/v1/marketstatus/upcoming`) and include `serverTime` plus exchange/index-group status fields.
- V2 snapshot docs only reference **stocks** (`/v2/snapshot/locale/us/markets/stocks/tickers/{stocksTicker}`), so the previous v2 indices path is undocumented.
- Added a market-status guard in `collectors/polygon_ws` so closed/extended hours skip REST fallback and update the cadence tracker to avoid repeated stale warnings.
- Freshness monitor now consults `/v1/marketstatus/now` and treats closed/extended hours as inactive for `polygon_index.websocket` staleness.
- Added `python -m kalshi_alpha.exec.market_status` CLI for ops to print market status + server time.
- Telemetry artifacts produced for INXU run_id `20251226_074420Z`:
  - `data/proc/telemetry/tob/20251226_074420Z.jsonl.gz`
  - `data/proc/telemetry/quote_intents/20251226_074420Z.jsonl.gz`
  - `data/proc/telemetry/runs/20251226_074420Z.json`
- Telemetry artifacts produced for NASDAQ100U run_id `20251226_074517Z` (preflight NO-GO override):
  - `data/proc/telemetry/tob/20251226_074517Z.jsonl.gz`
  - `data/proc/telemetry/quote_intents/20251226_074517Z.jsonl.gz`
  - `data/proc/telemetry/runs/20251226_074517Z.json`
- Ops volume report written: `reports/ops/telemetry_volume_2025-12-26.md` (caps + retention documented).
- Spot-check: telemetry rows include required keys (`run_id`, `window_id`, `series`, `market_ticker`, `ts`).
- Housekeeping proof: synthetic old telemetry file deleted by `python -m kalshi_alpha.exec.housekeep --keep-days 10000`.
- `make collect-polygon-ws` equivalent run stopped with `max_connections` (Massive WS limit); recorded in run log.
- Preflight NO-GO reasons during telemetry-only runs: `basis_audit_missing:INXU` and `basis_audit_missing:NASDAQ100U`.
- GPT bundle produced (verified): `docs/gpt_bundles/gpt_bundle_TICKET-103_20251225T213735Z_TICKET-103_polygon_snapshot_investigation.zip`.
- Updated `tools/verify_gpt_bundle.py` to accept RUN.md/COMMANDS.md/ARTIFACTS.md run-log format; bundle verification now passes.

## Recommended verification (requires API key approval)
## Live verification
- `GET /v3/snapshot/indices?ticker.any_of=I:SPX,I:NDX` returned status OK with 2 results.
- Both tickers returned `market_status=closed` and `timeframe=REAL-TIME` during the check on 2025-12-25; no entitlement errors surfaced.
- Example fields captured: `I:SPX value=6932.05`, `I:NDX value=25656.14642200707` (values are informational only).
- Ops CLI sample (2025-12-26): `market=closed`, `indicesGroups: {s_and_p: closed, nasdaq: closed}`, `serverTime=2025-12-26T02:04:59-05:00`.

## Optional follow-ups
- If needed, test the unified snapshot endpoint: `GET /v3/snapshot?type=indices&ticker.any_of=I:SPX,I:NDX`.

## Commits (this run)
- `47af5dd` Use v3 indices snapshot endpoint
- `f950419` Guard polygon WS fallback with market status
- `c1f5722` Update docs for polygon snapshot guard
- `6963f50` Freshness: skip closed market staleness
- `3258f22` Add ops market status CLI
- `b986cb2` Track docs + run logs
- `6f67bdf` Finalize run log metadata
- `62fcd45` Telemetry proof + bundle artifacts
- `cd7849b` Update bundle verification + gpt bundle
- `dbcb25e` Finalize bundle log updates
- `0f54d82` Finalize run log metadata (post-bundle)
- `HEAD` Merge/push command log + final diff refresh (current tip)
