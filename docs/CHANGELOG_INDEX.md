## 2025-11-03

- close-range calibration now loads late-day variance bump `λ_close` and event-tail multiplier `κ_event`; tests cover mass and CRPS/Brier guard rails.
- restricted `λ_close` variance bump to the 15:50–16:00 ET window and added regression covering pre-window behavior.
- added drivers/calendar events feed with DST-safe loader; index scanners now emit event tags for noon/close inputs.
- expanded Polygon index fixtures (gap/quiet/CPI/FOMC coverage for Aug–Sep 2024) and updated math regression to iterate every file with dynamic strike grids.
