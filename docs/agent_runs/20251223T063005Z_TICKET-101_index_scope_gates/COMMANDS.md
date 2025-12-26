# Commands Executed

- date -u +%Y%m%dT%H%M%SZ
- git rev-parse HEAD
- git checkout -b codex/TICKET-101_index_scope_gates
- pytest -q
- make monitors
- PYTHON=python3 make monitors
- python3 -m kalshi_alpha.exec.preflight_index --offline
- python3 -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run --offline --max-runtime-seconds 5
