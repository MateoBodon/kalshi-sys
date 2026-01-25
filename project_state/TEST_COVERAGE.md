# Test Coverage

## Metadata
- Generated: 2026-01-10T11:43:04Z
- Git SHA: 31316e59451269689f2da173d8a9c6d9049d3d5e
- Branch: codex/TICKET-111_project_state_refresh
- Commands: `python tools/project_state_build.py`, `python3 tools/agentic/project_state_refresh.py --zip`, `rg --files`, `sed -n '1,200p' README.md`, `sed -n '1,200p' docs/PROGRESS.md`, `sed -n '1,200p' CHANGELOG.md`, `sed -n '1,200p' pyproject.toml`, `sed -n '1,200p' Makefile`

## Suite overview
- Test runner: pytest (`pytest -q`), configured in `pyproject.toml`.
- Approximate test files: 167 files matching `tests/test_*.py`.
- Coverage config: `[tool.coverage.run]` in `pyproject.toml` (branch coverage, omit tests + notebooks).

## Major test areas
- Brokers + live safety: `tests/brokers/`, `tests/test_broker_live_safety.py`, `tests/test_broker_dry.py`.
- Index scanners + runners: `tests/exec/*`, `tests/test_index_scanners.py`, `tests/test_micro_runner.py`.
- Risk, limits, sizing: `tests/test_var_limiter.py`, `tests/test_limits.py`, `tests/test_sizing*.py`.
- Fees + slippage: `tests/test_exec_fees.py`, `tests/test_fees*.py`.
- Monitors + freshness: `tests/test_sigma_drift_monitor.py`, `tests/test_freshness_gate*.py`.
- Replay/backtest: `tests/backtest/*`, `tests/test_replay_scorecards.py`.
- Data drivers: `tests/drivers/*` and fixtures under `tests/fixtures/`.

## Default commands
- `pytest -q` (default).
- `make test` (pytest + sanity_check).

## Notes
- Many tests rely on fixtures under `tests/fixtures/` and `tests/data_fixtures/` for offline-safe execution.
- Live tests are gated and should remain dry/offline in CI unless explicitly instructed.
