# TICKET-121

## Goal
Harden index-only scope so index preflight/supervisor cannot be blocked by macro feeds and cannot accidentally run outside index-only trading scope.

## Scope
- Update `src/kalshi_alpha/exec/preflight_index.py` and `src/kalshi_alpha/exec/supervisor_index.py` (and any scope selectors) so index runs never require macro artifacts/feeds.
- Add tests simulating missing/stale macro artifacts to prove index preflight/supervisor remain index-only.
- Update one doc paragraph to state the invariant.
- Do NOT change gate thresholds, pricing math, or broker behavior.

## Acceptance Criteria
- `preflight_index` evaluates only index freshness/gates and does not reference macro freshness artifacts.
- `supervisor_index --series INXU --dry-run` remains index-only even if macro artifacts are missing/stale.
- Add a pytest that forces macro artifacts missing/stale and confirms index preflight behavior is unchanged.
- Docs mention the index-only invariant.

## Plan
1. Inspect preflight/supervisor freshness scope usage and current tests.
2. Enforce index-only freshness scope in `preflight_index` and wire index scope explicitly in `supervisor_index`.
3. Add/adjust pytest coverage for stale/missing macro artifacts in `tests/exec/test_preflight_index.py`.
4. Update docs and changelog/progress entries.

## Notes
- Keep diffs minimal; no changes to gate thresholds, pricing math, or broker behavior.
