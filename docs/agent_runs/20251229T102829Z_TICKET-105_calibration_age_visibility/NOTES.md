# Exploration Notes

- Preflight calibration gate lives in `src/kalshi_alpha/exec/preflight_index.py` with a 14-day max age threshold and uses `data/proc/calib/index_polygon/<SERIES>/<HORIZON>/params.json`.
- Strategy calibration files used by index ladders live under `data/proc/calib/index/{spx,ndx}/` with hourly variants (`hourly/<HH00>/params.json`) and close params (`close/params.json`).
- Scoreboard and pilot readiness previously only showed a single calibration age value; no consolidated report existed.
- `make calibrate-index` runs `jobs.calibrate_hourly` and `jobs.calibrate_close`, which use Polygon data (API key needed).
- `.env.local` contains the required credential variable names (values not logged).
