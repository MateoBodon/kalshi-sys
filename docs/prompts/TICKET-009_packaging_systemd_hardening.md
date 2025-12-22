# TICKET-009_packaging_systemd_hardening

Date: 2025-12-22
Ticket: #9 — Packaging/systemd import-path hardening

## Goal
- Remove reliance on PYTHONPATH=src for systemd and runbooks.
- Ensure editable install + venv python works for supervisor_index.
- Keep paper/dry defaults.

## Constraints
- Follow AGENTS.md (fail-closed, no live trading changes).
- Do not relax configs/pilot.yaml.
- Do not touch secrets or log credentials.
- Add minimal, safe changes only.

## Intended commands
- rg/sed to locate packaging + systemd/runbook references.
- pytest -q
- python -m kalshi_alpha.exec.preflight_index --offline --now "2025-12-22T10:50:00-05:00"
- python -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run --offline --now "2025-12-22T10:50:00-05:00" --no-ws-listen
- make gpt-bundle TICKET=TICKET-009_packaging_systemd_hardening RUN_NAME=20251222_052313Z_TICKET-009_packaging_systemd_hardening
