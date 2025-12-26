# Run

Goal: Decouple index-only GO/NO-GO from macro freshness and add explicit scope in artifacts for index ladders.

Initial repo state: 3a95e827f5b3a1209e36e78bd1f3fbe1f34fefa4

Approach:
- Add explicit `scope` metadata to freshness feeds and quality gate thresholds; filter by scope first.
- Default index collectors/replay to `configs/freshness.index.yaml` and index gates to `configs/quality_gates.index.yaml`.
- Emit scoped GO/NO-GO artifacts for index runs (preflight + scan) and add a regression test.

Key decisions:
- Preserve existing prefix-based scoping as a fallback when `scope` is unset to avoid breaking macro paths.
- Keep index scope artifacts additive (new fields) so existing consumers still read `go`/`reasons`.

Known risks:
- Index preflight now depends on freshness artifact when not offline; missing artifacts will surface as `data_freshness_missing`.

Acceptance checklist:
- [x] Index-only entrypoints load index-specific freshness + quality gates configs
- [x] Freshness/quality gates support explicit scope and index runs only consider index-scoped feeds
- [x] Go/no-go artifact includes scope + scoped_blockers; unscoped_blockers empty/ignored for index runs
- [x] Preflight + supervisor index flows emit scoped artifact
- [x] Test covers macro stale while index fresh => GO
- [x] Tests + required commands executed and logged
- [x] Docs updated (PLAN_OF_RECORD, PROGRESS, CHANGELOG)
- [x] Run logs complete (NOTES, TESTS, COMMANDS, FILES_TOUCHED, ARTIFACTS, DIFF)
