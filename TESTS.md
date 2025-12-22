# Tests

Date: 2025-12-22

- pytest -q
- /tmp/kalshi_pkg_smoke_venv: python -m kalshi_alpha.exec.preflight_index --offline --now "2025-12-22T10:50:00-05:00"
- /tmp/kalshi_pkg_smoke_venv: python -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run --offline --now "2025-12-22T10:50:00-05:00" --no-ws-listen
