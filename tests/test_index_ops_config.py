from __future__ import annotations

from datetime import time

from kalshi_alpha.config import load_index_ops_config
from kalshi_alpha.exec.runners import micro_index, scan_ladders


def test_index_ops_config_shared_defaults() -> None:
    config = load_index_ops_config()

    assert scan_ladders.INDEX_OPS_CONFIG == config
    assert micro_index.INDEX_OPS_CONFIG == config

    hourly = config.window_hourly
    assert hourly.start_offset_minutes == 15
    assert hourly.end_at_target is True
    assert hourly.cancel_buffer_seconds == 2.0

    close = config.window_close
    assert close.start == time(15, 50)
    assert close.end == time(16, 0)
    assert close.cancel_buffer_seconds == 2.0

    assert config.min_ev_usd == 0.05
    assert config.max_bins_per_series == 2
