# Run README

## Goal
- Move current progress to `main` and commit local policy doc updates.

## Summary
- Cherry-picked the ignore-reports commit onto `main`.
- Committed updated policy docs (`AGENTS.md`, `docs/DOCS_AND_LOGGING_SYSTEM.md`) and refreshed `docs/PROGRESS.md`/`CHANGELOG.md`.
- Ran the test suite and pushed `main` to origin.

## Commands
- git status --short
- git diff --stat
- git diff AGENTS.md
- git diff docs/DOCS_AND_LOGGING_SYSTEM.md
- date -u +%Y%m%d_%H%M%SZ
- git stash push -m "temp: local docs changes"
- git status --short
- git cherry-pick codex/TICKET-000_ignore-reports
- git stash pop
- git status --short
- rg -n "2025-12-23" docs/PROGRESS.md
- sed -n '25,40p' docs/PROGRESS.md
- sed -n '1,20p' CHANGELOG.md
- pytest -q
- git status --short
- git add -u docs/PROGRESS.md docs/DOCS_AND_LOGGING_SYSTEM.md
- git add AGENTS.md CHANGELOG.md
- git status --short
- git commit -m "Update policy docs" -m "Tests: pytest -q"
- git push origin main

## Tests
- pytest -q

## Artifacts
- None
