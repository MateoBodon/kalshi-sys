# Exploration Notes

## Current behavior (index runs)
- `preflight_index` blocks on missing env/keys, kill switch, stale/missing calibration params, and Polygon REST ping; it writes `reports/_artifacts/go_no_go.json` without any scope metadata.
- `scan_ladders` computes GO/NO-GO via quality gates + monitor freshness summary; it reads `reports/_artifacts/monitors/freshness.json` and applies `summarize_artifact(..., scope=...)` to decide `required_feeds_ok`.
- Freshness scoping currently relies on feed-id prefix heuristics (`polygon*`/`index*`) and quality gate scoping relies on namespace prefix heuristics; a macro-only freshness artifact (or a macro feed id that matches the prefix heuristic) can pollute index-only GO/NO-GO.

## Index config loading today
- Quality gates: `scan_ladders` chooses `configs/quality_gates.index.yaml` when scope resolves to `index` (else falls back to `configs/quality_gates.yaml`).
- Freshness: index runs read the shared freshness artifact at `reports/_artifacts/monitors/freshness.json`, whose contents depend on the config used by the freshness monitor/collector that last wrote it (currently `configs/freshness.yaml` or the collector default `/tmp/index_freshness.yaml`).

## Smallest change to scope freshness checks
- Add explicit `scope` metadata to freshness feed config entries and quality gate freshness/reconciliation entries.
- Filter feeds/thresholds by explicit scope first, falling back to existing prefix heuristics when `scope` is not set.
- Ensure index-only entrypoints default to `configs/freshness.index.yaml` and `configs/quality_gates.index.yaml` so index artifacts are isolated from macro freshness.
