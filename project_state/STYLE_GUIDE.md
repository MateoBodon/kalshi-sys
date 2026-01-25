# Style Guide

## Metadata
- Generated: 2026-01-10T11:43:04Z
- Git SHA: 31316e59451269689f2da173d8a9c6d9049d3d5e
- Branch: codex/TICKET-111_project_state_refresh
- Commands: `python tools/project_state_build.py`, `python3 tools/agentic/project_state_refresh.py --zip`, `rg --files`, `sed -n '1,200p' README.md`, `sed -n '1,200p' docs/PROGRESS.md`, `sed -n '1,200p' CHANGELOG.md`, `sed -n '1,200p' pyproject.toml`, `sed -n '1,200p' Makefile`

## Python conventions
- Target Python: 3.11 (`pyproject.toml`).
- Formatting: Ruff format with `line-length = 120` and double quotes.
- Linting: Ruff rules E/F/I/B/UP/S/DTZ/TID/ERA/PL/ANN with per-file ignores (see `pyproject.toml`).
- Typing: mypy strictness on key execution modules (see `[tool.mypy]` in `pyproject.toml`).

## Repo-specific safety conventions
- Fail-closed defaults for live trading; use explicit acknowledgements.
- No secrets in repo; `.env.local` is gitignored.
- Kill-switch file gates execution (`data/proc/state/kill_switch`).

## Documentation conventions
- Run logs are local-only under `docs/agent_runs/`.
- Update `docs/PROGRESS.md` + root `CHANGELOG.md` for every ticket.

## Commands
- `make fmt` (ruff format + import sorting).
- `make lint` (ruff check).
- `make typecheck` (mypy).
- `pytest -q` (tests).
