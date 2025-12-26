# Exploration Notes — TICKET-102

## Current basis audit tool (tools/settlement_basis_audit.py)
- Writes dataset to `data/proc/settlement_basis/<YYYY-MM-DD>_<SERIES>.parquet` and report to `reports/settlement_basis/<YYYY-MM-DD>_<SERIES>.md`.
- Dataset row schema includes per-window fields: polygon/kalshi values, basis, nearest strike + margin, and a per-window `flip_risk` flag (`abs(basis) >= nearest_strike_margin`).
- Report summarizes mean/median/p95/p99 for basis and abs(basis), plus top |basis| windows and flip-risk window table.
- Supports offline fixtures via `tests/fixtures/settlement_basis` (Kalshi + Polygon JSON).

## Preflight (src/kalshi_alpha/exec/preflight_index.py)
- GO/NO-GO checks: env presence, kill switch, calibration freshness, data freshness, Polygon ping.
- No basis audit gating; no basis artifact awareness in details.

## Quality gates (src/kalshi_alpha/core/gates/quality_gates.py + configs/quality_gates.index.yaml)
- Index scope gates cover data freshness + monitor limits only; no basis audit gate.

## Existing artifacts on disk
- `data/proc/settlement_basis/` contains sample parquet files (e.g., `2025-11-10_INXU.parquet`, `2025-11-10_NASDAQ100U.parquet`).
- No `data/proc/basis/` directory and no `reports/basis/` directory at repo root.

## Gaps vs ticket
- Basis audit artifacts not at required `data/proc/basis/<SERIES>/<YYYY-MM-DD>.json` + `reports/basis/<SERIES>/<YYYY-MM-DD>.md`.
- No daily summary JSON schema, no flip-risk summary block, and no PASS/FAIL thresholds.
- Preflight lacks basis audit gate (missing/stale/risky) and no `details["basis_audit"]` enrichment.
