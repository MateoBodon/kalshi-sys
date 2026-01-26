# Notes

## Local code findings
- `PolygonIndicesClient.fetch_snapshot` now targets `/v3/snapshot/indices` with `ticker.any_of`, matching the vendor docs.
- Added `fetch_market_status()` on the Polygon client to call `/v1/marketstatus/now`.
- REST fallback in `src/kalshi_alpha/exec/collectors/polygon_ws.py` calls `fetch_snapshot` when websocket ticks are stale.
- `_load_index_snapshot` in `src/kalshi_alpha/exec/runners/scan_ladders.py` also calls `fetch_snapshot` for live snapshots.
- The prior v2 snapshot path dated back to the initial index ladder commit (`git blame` showed 2025-11-03) and has now been replaced with v3.
- No local evidence of a successful REST fallback; recent run logs (2025-12-23) recorded 404 fallback warnings.
- Added a market-status guard in the websocket fallback loop: when `/v1/marketstatus/now` reports closed/extended hours for the indices group, we skip REST fallback and update the cadence tracker to avoid repeated stale warnings.
- Freshness monitor now calls `/v1/marketstatus/now` and treats closed/extended hours as inactive for `polygon_index.websocket` staleness.
- Added `python -m kalshi_alpha.exec.market_status` for ops to print market status + serverTime (raw JSON available via `--json`).
- Telemetry-only supervisor runs used `--no-ws-listen` + `--now 2025-12-26T19:50:00Z` to stay inside an hourly window while markets were closed.
- Preflight results: `basis_audit_missing:INXU` (run_id 20251226_074420Z) and `basis_audit_missing:NASDAQ100U` (run_id 20251226_074517Z).
- `make gpt-bundle` now copies `data/proc/telemetry/*` and `reports/ops/telemetry_volume_*.md` into the bundle when present.

## External docs (Polygon/Massive)
- Indices snapshot endpoint documented as `GET /v3/snapshot/indices` (query param `ticker`).
- Unified snapshot endpoint documented as `GET /v3/snapshot` with `type` supporting `indices`.
- Plan access for indices snapshots: Indices Basic not included; Indices Starter/Advanced are 15-minute delayed; Indices Business is real-time.
- V2 snapshot docs are for stocks (e.g., `/v2/snapshot/locale/us/markets/stocks/tickers/{stocksTicker}`), highlighting the code path’s `market` (singular) and `indices` segment is undocumented.
- Indices snapshot payloads include `market_status` and `timeframe`, which can be used to detect closed/delayed states.
- Market status endpoint `GET /v1/marketstatus/now` returns exchange/index-group status plus `serverTime`.

## Live verification
- `GET /v3/snapshot/indices?ticker.any_of=I:SPX,I:NDX` returned status OK with both tickers.
- Both tickers showed `market_status=closed` and `timeframe=REAL-TIME` at the time of the check (2025-12-25).
