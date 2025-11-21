## Session notes (Polygon index panel + model/backtest)

- Added offline Polygon index history loader and panel builder with ET-normalized timestamps and simple time-to-target/volatility features (`scripts/build_index_panel_polygon.py`, `drivers/index_polygon.py`).
- Implemented lightweight Student-t modelling + PMF projection for noon/close horizons and params serialization (`strategies/index/model_polygon.py` + `jobs/calibrate_index_polygon_model.py`).
- Built a minimal EV-aware backtest harness + CLI that simulates maker-only ladder trades over historical Polygon minutes (`strategies/index/backtest_polygon.py`, `exec/backtest_index_polygon.py`).
- New real-data fixtures under `tests/data_fixtures/index_panel_fast` and `tests/data_fixtures/index_panel_backtest` plus fast tests that exercise panel build, model fit/PMF, and CLI backtest.

Tests ran:
- `PYTHONPATH=src pytest tests/strategies/test_index_panel_polygon.py tests/strategies/test_model_polygon.py tests/exec/test_backtest_index_polygon.py`

Limitations / TODOs:
- Strike grid + fill/price model is approximate; consider wiring to historical ladder quotes when available.
- Model currently uses simple Student-t scaling by sqrt(time); could condition on realized_vol_30m or day-of-week with regression later.
- Backtest assumes maker=0 fee for indices and a fixed maker edge; refine with empirical alpha/slippage curves when ready.
