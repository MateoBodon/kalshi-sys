# Run README

## Goal
- Ensure `reports/` is gitignored and no report artifacts are tracked.

## Summary
- Added `reports/` to `.gitignore` and removed tracked report outputs from the index.
- Updated `docs/PROGRESS.md` and `CHANGELOG.md` with the change record.

## Commands
- ls
- git status --short
- rg --files -g '.gitignore'
- cat .gitignore
- git checkout -b codex/TICKET-000_ignore-reports
- git rm -r --cached reports
- pytest -q

## Tests
- pytest -q

## Artifacts
- None
