# Run Summary

Goal: Surface calibration ages in a single artifact and in readiness/scoreboard outputs; make preflight calibration NO-GO reasons include explicit file paths.

Approach:
- Added a calibration-age inspector + CLI report writer (mtime-based) for index ladder calibrations.
- Wired calibration summaries into ramp readiness + scoreboard markdown and JSON outputs.
- Expanded preflight calibration reasons to include the specific params.json file path.
- Added a unit test asserting stale calibration reasons include the filename/path.

Key Decisions:
- Use file mtime in UTC for calibration age status (per ticket), with hourly horizons enumerated per target hour.
- Use series-level worst-case age to drive readiness/scoreboard status and reasons.
- Write calibration ages report during pilot-readiness generation and via a standalone CLI.

Risks / Notes:
- Calibration age now reflects file mtime (not generated_at), which could differ if files are copied.
- Scoreboard/readiness statuses are based on worst-case hourly horizon; this is stricter than a min-age approach.
- Credential check: `.env.local` contains required env var names (`KALSHI_API_KEY_ID`, `KALSHI_PRIVATE_KEY_PEM_PATH`, `POLYGON_API_KEY`). Values not logged.

Additional actions:
- Confirmed credentials in `.env.local` (names only).
- `make calibrate-index` failed due to missing `kalshi_alpha.jobs` module; ran equivalent `python -m jobs.calibrate_hourly/close` with `PYTHONPATH=src` instead.
- Re-ran `make pilot-readiness` to refresh calibration age report.
- Calibration params refreshed under `data/proc/calib/index/{spx,ndx}/...` (tracked in git).
