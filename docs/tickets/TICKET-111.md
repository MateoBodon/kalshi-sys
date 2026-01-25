# TICKET-111

## Goal
- Regenerate and commit an up-to-date project_state snapshot aligned to current main HEAD.

## Scope
- Run the project_state generator on current main.
- Refresh project_state markdown and project_state/_generated indices.
- Update KNOWN_ISSUES.md and OPEN_QUESTIONS.md only if stale.
- Do not change trading/risk/broker/strategy logic.
- Keep diffs limited to project_state outputs + required run logs + docs/PROGRESS.md + CHANGELOG.md.

## Acceptance Criteria
- project_state/INDEX.md Git SHA equals `git rev-parse HEAD` and Branch matches current branch.
- project_state/_generated/git_head.txt matches HEAD and git_status/git_ls_files are refreshed.
- MODULE_SUMMARIES.md, FUNCTION_INDEX.md, and DEPENDENCY_GRAPH.md are regenerated with no old branch/SHA references.
- KNOWN_ISSUES.md and OPEN_QUESTIONS.md are consistent with current repo state (minimal factual edits only).
- `pytest -q` passes.

## Plan
1. Regenerate project_state/_generated artifacts and git metadata via tools scripts.
2. Refresh project_state markdown metadata and regenerate MODULE_SUMMARIES/FUNCTION_INDEX/DEPENDENCY_GRAPH.
3. Update KNOWN_ISSUES/OPEN_QUESTIONS only if stale.
4. Update docs/PROGRESS.md + CHANGELOG.md, create run logs, run tests, and bundle.

## Notes
- Required commands: `python3 tools/agentic/project_state_refresh.py --zip` and `pytest -q`.
- Emit GPT bundle via `python3 tools/agentic/gpt_bundle.py --zip --ticket TICKET-111`.
