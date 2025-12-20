# Current Results Snapshot

This snapshot is based on committed artifacts in `reports/` and `reports/_artifacts/`. Dates below are derived from filenames or file contents; the latest visible artifacts are from November 2025.

## Scoreboards
- `reports/scoreboard_7d.md` (7-day): Data Freshness OK; Polygon WS age reported as 898 ms; **no ledger data available**.
- `reports/scoreboard_30d.md` (30-day): Same status (freshness OK, no ledger data available).

## Pilot readiness
- `reports/pilot_readiness.md` (14-day): all four index series (`INXU`, `NASDAQ100U`, `INX`, `NASDAQ100`) are **NO-GO** due to `insufficient_data` (fills = 0, sample size = 0).

## GO/NO-GO gate
- `reports/_artifacts/go_no_go.json`: `go=false` with reasons including stale feeds and heartbeat/monitor staleness. Specific reasons list:
  - stale Cleveland nowcast and Treasury yields (hundreds of hours)
  - missing/low macro metrics (CRPS/Brier) for CPI/claims/teny/weather
  - `STALE_FEEDS`, `monitors_stale`, `heartbeat_stale`

## Replay / parity artifacts
- `reports/_artifacts/replay_ev.parquet` and `reports/_artifacts/monitors/ev_gap.json` exist, but no summary metrics are surfaced in scoreboards due to missing ledger fills.

## Digest artifacts
- `reports/digests/digest_2025-11-10.md` and `reports/_artifacts/digests/digest_2025-11-10.json` indicate a most recent digest around November 10, 2025.

## Interpretation
- The system is operationally instrumented, but current committed artifacts show **no recent fills** and **NO-GO** gating due to stale data and missing macro metrics. Any current evaluation should refresh data, rerun scans, and regenerate scoreboards.
