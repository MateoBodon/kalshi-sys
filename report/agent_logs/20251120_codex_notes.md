## Session notes (2025-11-20)

- Repo focus: SPX/NDX ladders only (`INX`, `INXU`, `NASDAQ100`, `NASDAQ100U`). Other families (CPI/claims/weather/gas) secondary and should stay untouched unless marked paused.
- Objectives today:
  1) Add an index-only family switch so pipelines/scoreboards honor `FAMILY=index` defaults and ignore other series unless explicitly requested.
  2) Centralize ET trading windows for noon (U) and close, making hourly/close runners refuse to arm outside configured windows, and ensure websocket freshness guard applies in the final minute for all four series.
  3) Wire new behaviours into tests/CI; prefer real fixtures.
- Key files to inspect/edit: `src/kalshi_alpha/sched/windows.py`, `src/kalshi_alpha/exec/pipelines/*`, `src/kalshi_alpha/exec/scoreboard.py`, `src/kalshi_alpha/exec/runners/pilot_hourly.py`, `pilot_close.py`, `scan_ladders.py`, relevant tests/fixtures, and configs (`configs/index_ops.yaml`, quality gate configs).
- Constraints/reminders: keep kill-switch/GO–NO-GO guards intact, maintain maker-only defaults, reuse existing patterns; update docs (AGENTS/README) and add report summary on completion.

### Progress log
- Added FAMILY switch (default `index`) across pipelines (`daily`, `today`, `week`) and scoreboard; skip macros unless `--family macro`/`FAMILY=macro`.
- Introduced shared window guard + `next_window_for_series`; gated pilot/micro runners and supervisor to configured ET windows with final-minute rerun, plus `--now` overrides for tests/ops.
- Updated CI workflow to pass `--family macro` for legacy macro pipelines; docs (README, AGENTS) note new default; pilot_teny forwards macro family.
- New tests: family gating/scoreboard skip, window guard/final-minute detection, pilot/micro runner window enforcement; conftest treats them as active index tests.

### Test snapshot
- `PYTHONPATH=src pytest -q tests/test_family_switch.py tests/test_index_windows_guard.py tests/test_micro_runner.py tests/test_pilot_runners.py`
- Ruff full repo fails (many pre-existing issues); targeted ruff still flags legacy long functions (today.py, scoreboard.py, conftest) unchanged in this session.
- ` PYTHONPATH=src python -m mypy ...` timed out on this machine (mypy available but large config); offline index scan CLI attempts timed out (>60s). Need follow-up run in better-resourced env.
