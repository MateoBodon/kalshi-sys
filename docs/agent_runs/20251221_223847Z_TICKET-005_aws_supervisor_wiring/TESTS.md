# Tests

## pytest -q
Result: PASS
Summary: 117 passed, 741 skipped
Notes: initial run failed in tests/test_index_scanner_fixtures.py; stabilized macro-stale fixture test and re-ran.

## pytest tests/test_index_scanner_fixtures.py::test_macro_stale_allows_execution_with_index_gates -q
Result: PASS

## python -m kalshi_alpha.exec.supervisor_index --help
Result: FAIL (ModuleNotFoundError: No module named 'kalshi_alpha')
Note: repo uses src layout; reran with PYTHONPATH=src.

## PYTHONPATH=src python -m kalshi_alpha.exec.supervisor_index --help
Result: PASS
Summary: CLI help rendered with new --series and --dry-run flags.
