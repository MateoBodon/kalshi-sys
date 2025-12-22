# Style Guide

## Metadata
- Generated: 2025-12-22T19:42:20Z
- Git SHA: a907a2eed87531d8178c3dc183d6f070182f9ebe
- Branch: codex/TICKET-000_project_state_rebuild
- Commands: `python tools/project_state_build.py`, `rg --files`, `sed -n '1,200p' README.md`, `sed -n '1,200p' docs/PROGRESS.md`, `sed -n '1,200p' CHANGELOG.md`, `sed -n '1,200p' pyproject.toml`, `sed -n '1,200p' Makefile`

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
