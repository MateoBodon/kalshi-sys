# Notes

- `tools/settlement_basis_audit.py` already emits per-series JSON/MD with `asof_date`, `generated_at`, and `flip_risk` summaries under `data/proc/basis/<SERIES>/<YYYY-MM-DD>.json` and `reports/basis/<SERIES>/<YYYY-MM-DD>.md`.
- `preflight_index._check_basis_audit` fail-closes on missing artifacts, `asof_date` mismatch, missing/invalid `generated_at`, and `flip_risk.flag` true/missing.
- Offline fixtures only exist for INXU/NASDAQ100U; all-series offline runs would need additional fixtures for INX/NASDAQ100.
