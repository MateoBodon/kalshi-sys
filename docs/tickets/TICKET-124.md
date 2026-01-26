# TICKET-124

## Goal
Ensure GPT bundle generation is never blocked by a dirty working tree and always writes to scratch outputs under `artifacts/_local/gpt_bundles/`.

## Scope
- Update the agentic GPT bundler to handle dirty trees safely via temporary stashing (with a disable flag).
- Enforce bundle output paths under `artifacts/_local/gpt_bundles/` and update ignores as needed.
- Update run-log guidance so future agents follow the scratch-only bundle policy.
- Add a lightweight test that asserts stash wrapping and output paths.

## Acceptance Criteria
- Bundler detects dirty status, stashes temporarily by default, and restores exactly.
- Bundles always land in `artifacts/_local/gpt_bundles/`.
- Docs note that bundling is allowed on dirty trees and outputs are scratch-only.
- Regression test covers dirty-tree handling and output path policy.

## Plan
1. Update `tools/agentic/gpt_bundle.py` for dirty-tree stashing and scratch-only output enforcement.
2. Align docs and ignores with the scratch bundle location.
3. Add/adjust tests for bundle output paths and stash wrapper.
4. Validate manually with a dirty tree and record run logs.

## Notes
- Keep diffs minimal and do not touch unrelated trading logic.
