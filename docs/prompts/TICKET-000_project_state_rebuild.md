# TICKET-000 — Project State Rebuild

Date: 2025-12-22

Goal:
- Rebuild `project_state/` with generated indices and updated docs.

Prompt source:
- See `docs/prompts/PROJECT_STATE_REBUILD` (full prompt text from the requester).

Intended commands:
- `rg --files`
- `python tools/project_state_build.py`
- `pytest -q`
- `zip -r docs/gpt_bundles/project_state_<timestamp>_<shortsha>.zip ...`

Safety constraints:
- Documentation only; preserve runtime behavior.
- Do not enable or modify live trading behavior.
- Do not touch secrets.
