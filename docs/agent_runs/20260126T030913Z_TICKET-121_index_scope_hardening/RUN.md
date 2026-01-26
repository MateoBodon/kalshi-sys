# RUN

## Summary
- Enforced index-only freshness scope in `preflight_index` and wired scope explicitly in `supervisor_index` so macro feeds cannot block index runs.
- Added/adjusted preflight tests for stale/missing macro freshness entries and fixed missing assertions.
- Updated index supervisor runbook, plan-of-record, logging doc, progress/changelog, and generated project_state + GPT bundles.

## Decisions
- Enforced `scope=index` in index preflight even if a caller supplies a different scope (recorded in `docs/DECISIONS.md`).

## Risks / Follow-ups
- Created `.venv` due to externally managed system Python; tests now run with `.venv/bin/python -m pytest -q` (pass).
- Repo had pre-existing modifications; left unchanged aside from files touched in this ticket.
