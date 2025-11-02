from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from kalshi_alpha.exec.monitors.sequential import (
    SequentialGuardParams,
    evaluate_sequential_guard,
)


def test_evaluate_sequential_guard_triggers_alert() -> None:
    now = datetime(2025, 11, 2, 12, 0, tzinfo=UTC)
    deltas = [0.5, 1.2, 2.4, 3.6, 6.0]
    frame = pl.DataFrame(
        {
            "series": ["CPI"] * len(deltas),
            "ev_expected_bps": [10.0] * len(deltas),
            "ev_realized_bps": [10.0 + delta for delta in deltas],
            "timestamp_et": [now - timedelta(minutes=5 * idx) for idx in range(len(deltas))],
        }
    )

    params = SequentialGuardParams(threshold=4.0, drift=0.2, min_sample=3)
    result = evaluate_sequential_guard(frame, params=params, window_start=now - timedelta(hours=2))

    assert result.alert is True
    assert result.triggers, "Expected at least one sequential trigger"
    trigger = result.triggers[0]
    assert trigger["series"] == "CPI"
    assert trigger["direction"] == "positive"
    assert trigger["stat"] > params.threshold
    assert result.series_stats["CPI"]["sample_size"] == len(deltas)
