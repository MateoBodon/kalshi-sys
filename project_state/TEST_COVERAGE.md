# Test Coverage

## Metadata
- Updated: 2026-01-26T00:04:48Z
- Git SHA: c78b933ec78e5a01a1b9e943de3dfd17ec5cd260
- Branch: codex/TICKET-000_project_state_refresh
- Sources: `pyproject.toml`, `rg --files -g 'tests/test_*.py'`

## Suite overview
- Test runner: pytest (`pytest -q`), configured in `pyproject.toml`.
- Approximate test files: 141 files matching `tests/test_*.py`.
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
