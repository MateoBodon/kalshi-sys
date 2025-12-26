# Tests

- `pytest -q` (exit 0)
  - Results: 117 passed, 743 skipped
- `pytest -q` (exit 0)
  - Results: 124 passed, 736 skipped
- `pytest -q` (exit 0)
  - Results: 124 passed, 736 skipped

Operational commands
- `python3 tools/settlement_basis_audit.py --series INXU --day 2025-12-23` (exit 1)
  - Result: 401 Unauthorized (Kalshi public markets endpoint), no offline fallback.
- `PYTHON=python3 make calibrate-index` (exit 2)
  - Result: Module `kalshi_alpha.jobs` not found.
- `PYTHONPATH=src PYTHON=python3 make calibrate-index` (exit 2)
  - Result: same module path error.
- `PYTHONPATH=src python3 -m jobs.calibrate_hourly --series INXU NASDAQ100U` (exit 0)
- `PYTHONPATH=src python3 -m jobs.calibrate_close --series INX NASDAQ100` (exit 0)
- `PYTHONPATH=src python3 -m kalshi_alpha.exec.collectors.polygon_ws --max-runtime 10` (exit 0)
  - Result: snapshots written; REST fallback returned 404s.
- `python3 -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run --record-tob` (exit 0)
  - Result: NO-GO (missing index_polygon calibration params + INXU basis audit).
- `python3 tools/settlement_basis_audit.py --series INXU --day 2025-12-23` (exit 0)
  - Result: basis parquet/json/md written.
- `PYTHONPATH=src python3 -m kalshi_alpha.exec.ingest.polygon_index --start 2025-12-23 --end 2025-12-24 --symbols I:SPX I:NDX --output-root data/raw/polygon` (exit 0)
- `python3 scripts/build_index_panel_polygon.py` (exit 0)
- `PYTHONPATH=src python3 -m jobs.calibrate_index_polygon_model` (exit 0)
- `python -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run --record-tob` (exit 127)
  - Result: `python` not found on PATH (reran with python3).
- `python3 -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run --record-tob` (exit 0)
  - Result: NO-GO (`basis_flip_risk:INXU`).
