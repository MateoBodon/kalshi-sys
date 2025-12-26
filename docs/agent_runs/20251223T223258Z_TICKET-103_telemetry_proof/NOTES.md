# Notes

- `supervisor_index` exits early on preflight NO-GO before invoking `micro_index`, so TOB/quote-intent telemetry only emits after a GO unless explicitly overridden.
- Preflight NO-GO reasons observed during this run: `basis_audit_missing:NASDAQ100`, `basis_audit_stale:INXU:generated_in_future`, and `basis_flip_risk:INXU`.
- WS freshness gating can skip a window with `polygon WS stale (age=unknown)` before running the scan.
- `micro_index` rejected `--online` from supervisor (flag missing) and always appended `--pilot`; both were fixed for telemetry-only runs.
- `collectors.polygon_ws` emitted REST fallback warnings (`404 page not found`) during the bounded run; WS listen was disabled for the telemetry-only supervisor run to proceed.
