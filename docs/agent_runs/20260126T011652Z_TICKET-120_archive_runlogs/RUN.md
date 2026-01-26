# RUN — TICKET-120

Goal:
- Make pilot-readiness + scoreboard outputs portable and archived so evidence survives machine moves and NO-GO reports are actionable.

Summary:
- Added optional `--archive-dir`/`--runlog` handling to scoreboard + ramp readiness and copy outputs into the archive folder.
- Sanitized readiness markdown paths to avoid absolute locations and added a NO-GO "How to fix" block with canonical commands.
- Updated Makefile + docs, added tests for archive outputs and absolute-path rejection.

Decisions:
- Strip root/drive prefixes from absolute paths when rendering readiness markdown to preserve portability.

Risks / Follow-ups:
- None; deps installed and tests/commands executed successfully.
