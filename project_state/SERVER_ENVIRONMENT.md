# Server Environment

## Metadata
- Generated: 2026-01-10T11:43:04Z
- Git SHA: 31316e59451269689f2da173d8a9c6d9049d3d5e
- Branch: codex/TICKET-111_project_state_refresh
- Commands: `python tools/project_state_build.py`, `python3 tools/agentic/project_state_refresh.py --zip`, `rg --files`, `sed -n '1,200p' README.md`, `sed -n '1,200p' docs/PROGRESS.md`, `sed -n '1,200p' CHANGELOG.md`, `sed -n '1,200p' pyproject.toml`, `sed -n '1,200p' Makefile`

## Local environment (this snapshot)
- OS: Darwin (macOS) `Darwin Kernel Version 25.0.0` on arm64.
- Python: 3.11.14 (`python --version`); pytest reported 3.12.2 during `pytest -q` (likely a different interpreter/venv).

## Expected runtime environment
- Python 3.11+ with dependencies from `pyproject.toml`.
- `.env.local` for secrets (gitignored), loaded by `kalshi_alpha.utils.env.load_env()`.
- Systemd templates for supervisors and monitors under `configs/systemd/`.
- CloudWatch agent configuration template under `configs/cloudwatch/kalshi-supervisor-index.json`.

## Key environment variables (non-exhaustive)
- `KALSHI_API_KEY_ID`, `KALSHI_PRIVATE_KEY_PEM_PATH` (live broker auth).
- `POLYGON_API_KEY` (Polygon REST + WS health checks).
- `FAMILY` (optional family scoping; default behavior should remain index-only in production).

## Runtime safety defaults
- Kill switch: `data/proc/state/kill_switch` (presence forces NO-GO + cancel intent).
- Live broker requires explicit acknowledgements and pilot caps (`configs/pilot.yaml`).

## Deployment docs
- `docs/runbooks/` for AWS/systemd guidance.
- `docs/PROGRESS.md` for the most recent verification logs.
