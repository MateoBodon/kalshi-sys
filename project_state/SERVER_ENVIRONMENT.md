# Server Environment

## Metadata
- Updated: 2026-01-26T00:04:48Z
- Git SHA: c78b933ec78e5a01a1b9e943de3dfd17ec5cd260
- Branch: codex/TICKET-000_project_state_refresh
- Sources: `uname -a`, `python3 --version`, `README.md`

## Local environment (this snapshot)
- OS: Linux `6.8.0-90-generic` on x86_64 (Codex sandbox).
- Python: 3.12.3 (`python3 --version`).

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
