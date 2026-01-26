# RUN

## Summary
- Updated the agentic GPT bundler to enforce scratch-only output under `artifacts/_local/gpt_bundles/` and added dirty-tree stash/restore handling with a `--no-stash` escape hatch.
- Aligned bundle docs/ignores with the scratch path and added regression coverage for output paths and stash behavior.
- Manually validated bundling succeeds on a dirty tree and preserves `git status --porcelain` exactly.

## Decisions
- Use a temporary stash (default) to keep bundling reproducible on dirty trees; allow opting out via `--no-stash`.

## Risks
- Pytest failures in existing index scanner fixture tests block a full green run; see TESTS.md.
- Stash wrapper reports mismatch errors but does not attempt automated resolution.

## Manifest
- Timestamp (UTC): 2026-01-26T20:57:22Z
- Ticket: TICKET-124
- Branch: codex/TICKET-124_gpt_bundle_dirty_safe
- Commit: 6941c3a1aaa6ca22e4f1a5961dc0be3d560bd3d9
- Commands: `python3 tools/agentic/gpt_bundle.py --zip --ticket TICKET-DIRTY-TEST`; `pytest -q`
- Inputs: `TRACKING_POLICY.md`, `tools/agentic/gpt_bundle.py`, `tools/agentic/project_state_refresh.py`, `docs/agent_runs/README.md`, `.gitignore`
