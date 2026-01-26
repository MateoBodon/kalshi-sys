# TICKET-120

## Goal
Make pilot-readiness + scoreboard outputs portable and archived so evidence survives machine moves and NO-GO reports are actionable.

## Scope
- Update Makefile + report writers (scoreboard + pilot readiness/ramp) to support optional `--archive-dir`/`--runlog` output.
- Sanitize markdown to avoid absolute paths.
- Add a short "How to fix" block on NO-GO reports.
- Do not change pricing/strategy math, broker behavior, or gate thresholds.

## Acceptance Criteria
- `make pilot-readiness` writes/copies `pilot_readiness.md` + `pilot_ready.json` into a stable archive/runlog folder when configured.
- `python -m kalshi_alpha.exec.scoreboard` writes/copies `scoreboard_7d.md` + `scoreboard_30d.md` into the same archive/runlog folder when configured.
- Generated markdown contains no absolute paths (no `/home/`, `/Users/`, or drive-letter paths).
- NO-GO markdown includes a brief "How to fix" section with canonical commands.
- Pytest fails if absolute paths appear in readiness/scoreboard markdown.

## Plan
1) Add archive/runlog options to scoreboard + ramp readiness writers and copy outputs.
2) Sanitize markdown path fields in readiness outputs and add NO-GO "How to fix" section.
3) Update Makefile targets to pass optional archive/runlog args.
4) Add/adjust tests for archive copies and absolute-path checks.
5) Update docs and run logs.

## Notes
- Keep diffs minimal and fail-closed if archive outputs are missing when requested.
