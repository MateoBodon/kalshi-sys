from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from kalshi_alpha.drivers.polygon_index.client import MinuteBar
from kalshi_alpha.drivers.polygon_index import snapshots


def _make_bars(values: list[float]) -> list[MinuteBar]:
    start = datetime(2023, 11, 1, 15, 0, tzinfo=UTC)
    bars: list[MinuteBar] = []
    for index, value in enumerate(values):
        ts = start + timedelta(minutes=index)
        bars.append(
            MinuteBar(
                timestamp=ts,
                open=value,
                high=value + 0.5,
                low=value - 0.5,
                close=value,
                volume=1000 + index * 10,
                vwap=value,
                trades=50 + index,
            )
        )
    return bars


def test_distance_to_strikes_handles_none_price() -> None:
    distances = snapshots.distance_to_strikes(None, [5000.0, 5010.0])
    assert 5000.0 in distances and math.isnan(distances[5000.0])
    distances_real = snapshots.distance_to_strikes(5005.0, [5000.0, 5010.0])
    assert distances_real[5000.0] == pytest.approx(5.0)
    assert distances_real[5010.0] == pytest.approx(-5.0)


def test_ewma_sigma_now_positive() -> None:
    bars = _make_bars([5000.0, 5002.0, 5004.0, 5003.0, 5005.0, 5006.0])
    sigma = snapshots.ewma_sigma_now(bars, span=10, min_samples=3)
    assert sigma > 0.0


def test_micro_drift_averages_recent_changes() -> None:
    bars = _make_bars([5000.0, 5001.0, 5002.0, 5004.0, 5003.0])
    drift = snapshots.micro_drift(bars, window=3)
    expected = ((5002.0 - 5001.0) + (5004.0 - 5002.0) + (5003.0 - 5004.0)) / 3
    assert drift == pytest.approx(expected)


def test_build_snapshot_metrics_bundle() -> None:
    bars = _make_bars([5000.0, 5001.0, 5002.5, 5003.0])
    metrics = snapshots.build_snapshot_metrics(
        price=5003.0,
        strikes=[4995.0, 5005.0],
        bars=bars,
        ewma_span=8,
        drift_window=2,
    )
    assert "sigma_now" in metrics and metrics["sigma_now"] >= 0.0
    assert "micro_drift" in metrics
    assert "distance_to_strike" in metrics
    assert metrics["distance_to_strike"][4995.0] == pytest.approx(8.0)
