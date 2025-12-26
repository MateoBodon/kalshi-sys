# TESTS

- python3 tools/build_fillcalib_dataset.py --help (exit 0)
- python3 tools/build_fillcalib_dataset.py --series INXU --from 2025-01-01 --to 2025-01-01 --telemetry-root tests/fixtures/telemetry --output-curves <tmp>/curves.json --output-report <tmp>/report.md --output-parquet <tmp>/dataset.parquet --write-parquet (exit 0)
- pytest -q (exit 0; 124 passed, 742 skipped)
- PYTHON=python3 make pilot-readiness (exit 0; NO_GO due to ledger_stale + monitors_stale)

Not run:
- make pilot-readiness (initial ticket run; executed later as requested)
