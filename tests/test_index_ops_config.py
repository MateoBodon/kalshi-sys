from __future__ import annotations

import pytest

from kalshi_alpha.config import load_index_ops_config
from kalshi_alpha.exec.runners import micro_index, scan_ladders
from kalshi_alpha.exec.scanners import index_scan_common


@pytest.mark.parametrize("series", ["INX", "INXU", "NASDAQ100", "NASDAQ100U"])
def test_scan_and_microlive_share_ops_windows(series: str) -> None:
    config = load_index_ops_config()
    window_expected = config.window_for_series(series)
    window_scan = scan_ladders._ops_window_for_series(series)  # noqa: SLF001
    assert window_scan is not None
    assert window_scan.name == window_expected.name
    assert window_scan.start == window_expected.start
    assert window_scan.end == window_expected.end
    window_micro = micro_index.INDEX_OPS_CONFIG.window_for_series(series)
    assert window_micro.name == window_expected.name
    assert window_micro.start == window_expected.start
    assert window_micro.end == window_expected.end
    assert scan_ladders.CANCEL_BUFFER_SECONDS == pytest.approx(config.cancel_buffer_seconds)


@pytest.mark.parametrize("series, cap", [
    ("INXU", 1000),
    ("NASDAQ100U", 1000),
    ("INX", 1200),
    ("NASDAQ100", 1200),
])
def test_scanner_pal_guard_respects_policy(series: str, cap: int) -> None:
    guard = index_scan_common._pal_guard(series)  # noqa: SLF001
    assert guard.policy.default_max_loss <= cap
