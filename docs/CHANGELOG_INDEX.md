## 2025-11-04

- fit paper-ledger execution curves for INX/INXU/NASDAQ100/NASDAQ100U via `jobs.fit_index_execution`, persisted per-series `alpha.json`/`slippage.json`, and wired scanners/reports to consume the curves (fill% vs α, realized slippage deltas, scoreboard readiness updates).
- reran INX/NDX hourly and close calibrations on Polygon minutes through 2025-11-03 (covers DST week plus latest CPI/FOMC), persisting refreshed `κ_event`/`λ_close` payloads in `data/proc/calib/index/{spx,ndx}`.
- gated the close variance bump `λ_close` behind CPI/FOMC event tags so calm days stay unchanged, and tightened regression to assert CRPS/Brier equality on non-event fixtures.
- ran offline index hourly/close DRY scanners to hydrate reports with the new calibration set.
- shipped `python -m kalshi_alpha.exec.live_smoke` CLI to validate signed auth, next-hour U-series availability, and outstanding-order state before GO/NO-GO; documented the workflow in the Runbook and linked it from the operations vision.
- expanded Polygon fixtures to include DST (2024-11-04), gap (2024-04-19), and quiet (2024-06-03) windows for SPX/NDX noon+close tests with refreshed metadata checksums.

## 2025-11-03

- close-range calibration now loads late-day variance bump `λ_close` and event-tail multiplier `κ_event`; tests cover mass and CRPS/Brier guard rails.
- restricted `λ_close` variance bump to the 15:50–16:00 ET window and added regression covering pre-window behavior.
- added drivers/calendar events feed with DST-safe loader; index scanners now emit event tags for noon/close inputs.
- expanded Polygon index fixtures (gap/quiet/CPI/FOMC coverage for Aug–Sep 2024) and updated math regression to iterate every file with dynamic strike grids.
- re-ran SPX/NDX noon+close calibrations with calendar-tagged fixtures, capped noon event tails, persisted new params, refreshed tests, and generated DRY scans for INXU/NASDAQ100U/INX/NASDAQ100.
- switched CLI to `jobs.calibrate_hourly`, auto-extended calibration window around DST/CPI/FOMC weeks, and persisted explicit `κ_event`/`λ_close` fields for SPX/NDX hourly+close params with regression coverage.
