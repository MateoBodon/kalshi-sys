# Run Summary

Goal: Fix `make calibrate-index` to invoke the correct jobs modules and verify it works with real data/credentials.

Approach:
- Updated the Makefile target to call `jobs.calibrate_hourly` and `jobs.calibrate_close` with `PYTHONPATH=src`.
- Ran `make calibrate-index` using `.env.local` (names only; values not logged).
- Re-ran `make pilot-readiness` to refresh calibration age/readiness artifacts.

Key Decisions:
- Use the top-level `jobs` package instead of the non-existent `kalshi_alpha.jobs` namespace.
- Set `PYTHONPATH=src` to ensure `kalshi_alpha` imports resolve in the jobs modules.

Risks / Notes:
- Calibration outputs depend on live Polygon availability and may vary by run window.
- Calibration params under `data/proc/calib/index` were updated (tracked in git) as a result of the live run.
- Credential check: `.env.local` exists and contains required env var names (`KALSHI_API_KEY_ID`, `KALSHI_PRIVATE_KEY_PEM_PATH`, `POLYGON_API_KEY`). Values were not logged.

Additional actions:
- Executed `pytest -q`.
