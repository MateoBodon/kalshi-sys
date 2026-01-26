# TICKET-123

## Goal
Make agentic bundle outputs comply with TRACKING_POLICY by moving zips to scratch zones, removing docs/_generated usage, and ensuring run logs carry a manifest.

## Scope
- Update tools/agentic outputs to write bundles under artifacts/_local or reports/_runs (no docs/_bundles).
- Stop repo_snapshot and bundle flows from writing/using docs/_generated.
- Align docs references (PROGRESS + logging/plan docs) with canonical run logs/artifacts only.
- Extend .gitignore to prevent non-canonical generated output churn.
- Add a small regression test asserting default bundle paths are in canonical scratch zones.

## Acceptance Criteria
- project_state_refresh --zip writes under artifacts/_local or reports/_runs by default.
- gpt_bundle --zip writes under artifacts/_local or reports/_runs by default.
- No code path writes bundles under docs/_bundles or docs/_generated.
- docs/PROGRESS.md references only docs/agent_runs and/or docs/artifacts.
- Regression test added for canonical bundle output paths.
- Bundle commands leave git status clean (outputs land in ignored scratch).

## Plan
1. Inspect tools/agentic bundle scripts and repo snapshot generation.
2. Update bundle defaults + docs references to scratch zones and add run log manifest guidance.
3. Add a targeted test for default bundle output paths.
4. Update changelog/progress and capture run log artifacts.

## Notes
- Keep changes minimal and fail-closed; do not touch non-agentic bundling flows unless required.
