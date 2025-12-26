# RUN — TICKET-104

- Goal: Build fill calibration dataset + conservative maker fill curves + wire into defaults.
- Approach: Implement fillcalib dataset builder + report, add fill curve loader/status in execution, add fixture-based tests, update docs and run logs.

## Decisions
- Proxy fills are computed from TOB crossings within a 30s horizon; p_fill = clamp(proxy_rate * 0.25, 0, 0.25) with min_samples=200.
- Missing/invalid curves fail closed: fill alpha clamps to 0 and scanner reports `fill_curve.uncalibrated=true` with a single reason code.
- Series default probability is computed as a sample-weighted average of bucket p_fill values for use when bucket metadata is unavailable.

## Risks
- Proxy fills are an upper bound (no queue position/cancel latency), so curves may still be optimistic even after scaling.
- If curves are missing or stale, fill alpha collapses to zero; paper fill metrics will be conservative and may suppress diagnostics.

## Follow-up
- Generated real curves from telemetry for INXU (2025-12-23 → 2025-12-26) and ran `make pilot-readiness` to refresh readiness artifacts.
