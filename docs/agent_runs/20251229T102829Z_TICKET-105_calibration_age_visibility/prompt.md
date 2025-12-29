# Prompt

Ticket: TICKET-105 — Calibration age visibility (single summary artifact + scoreboard + explicit NO-GO reasons)

Primary requirements:
- Single calibration age artifact: reports/calibration/calibration_ages_<ASOF_DATE>.md
- Scoreboard shows calibration age status per series (OK/STALE/MISSING + age)
- Preflight NO-GO reasons include explicit calibration file paths
- Add test: stale calibration -> NO-GO reason includes filename
