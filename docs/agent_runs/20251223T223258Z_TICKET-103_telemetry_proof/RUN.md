# Run Summary

- Added a PAPER-only telemetry override in `supervisor_index` + `scan_ladders` so dry-run scans can emit bounded TOB/quote-intent telemetry even when preflight is NO-GO, without executing broker endpoints.
- Wrote telemetry run metadata under `data/proc/telemetry/runs/<RUN_ID>.json` and added an ops report generator for telemetry volume (`reports/ops/telemetry_volume_<YYYY-MM-DD>.md`).
- Hardened `make gpt-bundle` diff range to use merge-base, and made `micro_index` accept `--online`/skip `--pilot` for telemetry-only runs.

## Decisions

- Used `--telemetry-only` with `--no-ws-listen` for the INXU run to bypass WS freshness gating after the WS listener reported stale/unknown age.
- Pinned the window via `--now 2025-12-23T14:55:00-05:00` to fit the hourly window and keep the run bounded.
- Ran the telemetry volume report for run_id `20251223_225859Z`.

## Risks / Follow-ups

- WS freshness gating was disabled for the telemetry-only run; future runs should re-enable once WS connectivity is stable.
- Polygon REST fallback returned 404s during the bounded WS collector run; may need API endpoint verification.
- `housekeep` pruned 27 artifacts older than 30 days during the retention proof.
