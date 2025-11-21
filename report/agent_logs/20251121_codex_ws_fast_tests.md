Kalshi Alpha – session summary (2025-11-21)

Changes:
- Added `kalshi_alpha.drivers.polygon_index_ws` singleton/context manager, refactored supervisor websocket loop to use it, and exposed connection + last-message metrics.
- Introduced fast offline index scan mode (`--fast-fixtures`) with trimmed Polygon fixtures and lightweight runners/tests.
- Hardened time-awareness coverage for window gating and final-minute rerun logic plus websocket lifecycle tests.

Tests:
- PYTHONPATH=src pytest tests/drivers/test_polygon_index_ws.py tests/exec/test_ws_lifecycle.py tests/exec/test_fast_index_scans.py tests/exec/test_time_awareness.py
