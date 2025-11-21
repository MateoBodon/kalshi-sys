Session: index paper ledger + scoreboard.

What changed:
- Added `kalshi_alpha.exec.index_paper_ledger.log_index_paper_trade` (JSONL at `data/proc/ledger/index_paper.jsonl`, env override `KALSHI_INDEX_PAPER_LEDGER_PATH`) and wired it into the dry broker for INX/INXU/NASDAQ100/NASDAQ100U proposals.
- Added `kalshi_alpha.exec.scoreboard_index_paper` CLI to aggregate ledger trades by trading day/series/window and emit Markdown reports under `reports/index_paper/`.
- Added fixtures and tests for the new ledger + scoreboard; enabled the tests in `tests/conftest.py`.
- Relaxed freshness gating for index scans to ignore stale non-index feeds; only Polygon index websocket staleness now blocks index scans.
- Smoke: executed `scan_series` (offline fixtures, INXU, now_override 2025-11-03 16:50Z) + `execute_broker` (dry) to populate `data/proc/ledger/index_paper.jsonl` and generated `reports/index_paper/20251121_scoreboard.md`.

Tests run:
- `PYTHONPATH=src pytest -q tests/exec/test_index_paper_ledger.py tests/exec/test_scoreboard_index_paper.py`
- `PYTHONPATH=src pytest -q tests/drivers/test_kalshi_index_history.py tests/strategies/test_index_panel_polygon.py tests/strategies/test_model_polygon.py tests/strategies/test_fill_model.py tests/exec/test_backtest_index_polygon.py tests/exec/test_fast_index_scans.py tests/exec/test_time_awareness.py`

Notes / limits:
- Ledger logging is dry-only; live broker untouched.
- `ev_after_fees_cents` is per-contract EV at decision time; realized P&L will get added once settlement history is connected.
- DRY runs must still generate proposals; otherwise no ledger rows are written.
