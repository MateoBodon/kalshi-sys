# TESTS

- [fail: exit 127] pytest -q (pytest not found)
- [fail: exit 1] python -m pytest -q (No module named pytest)
- [pass] .venv/bin/python -m pytest -q
