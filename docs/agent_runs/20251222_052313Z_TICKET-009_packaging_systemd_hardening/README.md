# Run Log

run_name: 20251222_052313Z_TICKET-009_packaging_systemd_hardening
ticket_id: TICKET-009_packaging_systemd_hardening
agent: Codex CLI (gpt-5-codex)
branch: codex/TICKET-009_packaging_systemd_hardening
start_utc: 2025-12-22T05:23:13Z
end_utc: 2025-12-22T05:40:21Z
environment: local
network_access: yes
web_search_used: no

## Goal
- Harden packaging/systemd to avoid PYTHONPATH=src reliance and keep paper-only defaults.

## Summary of changes
- Updated systemd unit to use venv python and correct StartLimit placement (paper-only defaults preserved).
- Updated AWS/index ladder runbooks to use venv-based execution and added EC2 bootstrap script.
- Added scipy/pandas runtime deps to pyproject for index model imports.

## Commands run (high level)
- rg/sed to locate packaging + systemd/runbook references.
- pytest -q
- Fresh venv smoke: preflight_index + supervisor_index (offline, dry-run).
- make gpt-bundle (at end).

## Tests run
- pytest -q
- /tmp/kalshi_pkg_smoke_venv: python -m kalshi_alpha.exec.preflight_index --offline --now "2025-12-22T10:50:00-05:00"
- /tmp/kalshi_pkg_smoke_venv: python -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run --offline --now "2025-12-22T10:50:00-05:00" --no-ws-listen

## Artifacts produced
- reports/_artifacts/go_no_go.json (offline smoke output)

## Known risks / TODOs
- Offline smoke shows NO-GO due to missing calibration (expected without fixtures).
- project_state/KNOWN_ISSUES.md updated locally but directory is gitignored.
