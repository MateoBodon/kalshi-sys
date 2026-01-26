# Tests

Date (UTC): 2026-01-26

Commands:
- `python -m pip install pytest` (failed: externally-managed environment)
- `python -m venv .venv` (ok)
- `.venv/bin/python -m pip install pytest` (ok; already satisfied)
- `PATH=.venv/bin:$PATH pytest -q` (pass)

Outcome: PASS
