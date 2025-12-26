# Tests

## python tools/settlement_basis_audit.py --help
- Command: `python tools/settlement_basis_audit.py --help`
- Exit: 127
- Note: `python` not found in PATH; reran with `python3`.

## python3 tools/settlement_basis_audit.py --help
- Command: `python3 tools/settlement_basis_audit.py --help`
- Exit: 0
- Output (excerpt):
  - `usage: settlement_basis_audit.py [-h] --day DAY --series {INX,INXU,NASDAQ100,NASDAQ100U} ...`

## python -m kalshi_alpha.exec.preflight_index --offline
- Command: `python -m kalshi_alpha.exec.preflight_index --offline`
- Exit: 127
- Note: `python` not found in PATH; reran with `python3`.

## python3 -m kalshi_alpha.exec.preflight_index --offline
- Command: `python3 -m kalshi_alpha.exec.preflight_index --offline`
- Exit: 1
- Output (excerpt):
  - `PRECHECK index: NO-GO reasons=4 series=INX,NASDAQ100,INXU,NASDAQ100U`

## pytest -q
- Command: `pytest -q`
- Exit: 0
- Summary: `117 passed, 742 skipped`
