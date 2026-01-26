# RUN

## Summary
- Added an all-series settlement basis audit runner with `--all-series` and runlog/archive support in `tools/settlement_basis_audit.py`.
- Expanded basis audit gating coverage with stale/flip-risk tests and added an archive-path test for basis artifacts.
- Updated plan-of-record + logging docs plus changelog/progress entries for the new runner and archive behavior.
- Documented the new basis audit workflow in `docs/RUNBOOK.md`.

## Decisions
- Archived basis audit outputs by preserving project-relative paths under `--runlog/--archive-dir` to avoid filename collisions.

## Risks
- Tests pass, but many existing tests are skipped in this environment (expected).
