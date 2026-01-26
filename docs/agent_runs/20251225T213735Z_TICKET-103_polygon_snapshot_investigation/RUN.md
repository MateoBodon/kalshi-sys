# Run Summary
- Investigated REST fallback 404s by inspecting `PolygonIndicesClient.fetch_snapshot` and `polygon_ws` fallback path plus git history.
- Reviewed Polygon/Massive vendor docs to confirm indices snapshot endpoints and market-status semantics.
- Updated the REST snapshot call to use `/v3/snapshot/indices` and parse v3 payloads, with explicit error handling for not-entitled responses.
- Added a market-status guard to the websocket REST fallback so closed/extended hours skip fallback and avoid stale warnings.
- Live-verified `/v3/snapshot/indices?ticker.any_of=I:SPX,I:NDX` using the configured API key (market_status=closed at time of check).
- Wired `/v1/marketstatus/now` into the freshness monitor so closed/extended hours do not flag `polygon_index.websocket` as stale.
- Added `python -m kalshi_alpha.exec.market_status` CLI for ops to print market status + server time.
- Removed the docs ignore so vendor/prompts/gpt outputs can be tracked in git.
- Ran telemetry-only supervisor dry-runs (INXU + NASDAQ100U) with bounded TOB/quote-intent capture and recorded preflight NO-GO reasons.
- Generated the ops telemetry volume report and verified telemetry rows contain required keys.
- Proved retention pruning with a synthetic old telemetry file deletion.
- Updated `make gpt-bundle` to include telemetry artifacts + ops volume reports when present.
- Tests: `pytest -q`.

## Decisions
- Replace the undocumented v2 indices snapshot path with the documented v3 indices snapshot endpoint.
- Keep fail-closed behavior when the v3 response carries an error (e.g., NOT_ENTITLED).
- Use `/v1/marketstatus/now` to suppress REST fallback during inactive index hours.
- Use `/v1/marketstatus/now` in freshness gating so inactive index hours do not trip stale alarms.
- Use `--telemetry-only` with `--dry-run` to capture telemetry even when preflight is NO-GO.
- Use `--no-ws-listen` and `--now` to run a bounded dry-run window while markets are closed.
- Include telemetry artifacts in GPT bundles to satisfy reviewability.

## Risks / Open Questions
- Runtime verification still requires a live API key and entitlement check.
- If the API key lacks index snapshot entitlements, v3 calls will still fail with NOT_ENTITLED.
- Freshness stays fail-closed if market-status calls fail (no inactive override applied).
- Preflight remains NO-GO until basis audits exist (`basis_audit_missing`); telemetry-only runs do not change live safety.
