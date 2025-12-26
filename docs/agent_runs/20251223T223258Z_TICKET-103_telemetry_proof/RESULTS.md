# Results

## Telemetry runs
- NASDAQ100 (run_id `20251223_225626Z`): preflight NO-GO `basis_audit_missing:NASDAQ100`; WS stale (`age=unknown`); no telemetry files emitted.
- INXU (run_id `20251223_225724Z`): preflight NO-GO `basis_audit_stale:INXU:generated_in_future`, `basis_flip_risk:INXU`; supervisor failed before scan due to missing `--online` flag in `micro_index` (fixed in this ticket).
- INXU (run_id `20251223_225825Z`): preflight NO-GO `basis_audit_stale:INXU:generated_in_future`, `basis_flip_risk:INXU`; scan aborted due to `--telemetry-only` + `--pilot` conflict (fixed in this ticket).
- INXU (run_id `20251223_225859Z`): telemetry-only dry-run succeeded with `--no-ws-listen`.
  - `data/proc/telemetry/tob/20251223_225859Z.jsonl.gz` (1 line, 359 bytes)
  - `data/proc/telemetry/quote_intents/20251223_225859Z.jsonl.gz` (2 lines, 629 bytes)
  - `data/proc/telemetry/runs/20251223_225859Z.json` (status NO-GO, telemetry-only override)
  - `reports/ops/telemetry_volume_2025-12-23.md`
  - `reports/_artifacts/go_no_go.json` (latest preflight NO-GO details)

## Collectors
- `collectors.polygon_ws` bounded run emitted REST fallback warnings (`404 page not found`); run was interrupted after repeated warnings.

## Retention proof
- `python -m kalshi_alpha.exec.housekeep --keep-days 30` removed 27 artifacts older than 30 days (including the synthetic `fixture_old.jsonl.gz`).

## Commit list (origin/main..HEAD)
- 1b08527 Update telemetry docs and progress
- 2685a52 Fix gpt-bundle diff range
- b3d80f0 Add telemetry volume ops report
- 0bd042b Add telemetry-only dry-run path for index telemetry

## Bundle
- GPT bundle path: `docs/gpt_bundles/gpt_bundle_TICKET-103_20251223T223258Z_TICKET-103_telemetry_proof.zip`
