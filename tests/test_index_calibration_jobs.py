from __future__ import annotations

import json
from datetime import UTC, date, datetime, time
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from jobs._index_calibration import build_sigma_curve, extend_calibration_window
from jobs.calibrate_close import _write_params as write_close_params
from jobs.calibrate_hourly import _write_params as write_hourly_params
from kalshi_alpha.drivers.polygon_index.client import MinuteBar


def _minute_bar(et_timestamp: datetime, close: float) -> MinuteBar:
    ts = et_timestamp.astimezone(UTC)
    return MinuteBar(
        timestamp=ts,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1000.0,
        vwap=close,
        trades=10,
    )


def test_build_sigma_curve_computes_minutes() -> None:
    et = ZoneInfo("America/New_York")
    day = datetime(2024, 1, 2, tzinfo=et)
    bars = [
        _minute_bar(day.replace(hour=11, minute=58), 5000.0),
        _minute_bar(day.replace(hour=11, minute=59), 5001.0),
        _minute_bar(day.replace(hour=12, minute=0), 5002.0),
    ]
    frame = build_sigma_curve({"I:SPX": bars}, target_time=time(12, 0), residual_window=2)
    assert isinstance(frame, pl.DataFrame)
    assert not frame.is_empty()
    minutes = frame.get_column("minutes_to_target").to_list()
    assert 0 in minutes
    assert max(minutes) == 2


def test_write_params_emits_json(tmp_path) -> None:
    et = ZoneInfo("America/New_York")
    day = datetime(2024, 1, 2, tzinfo=et)
    bars = {
        "I:SPX": [
            _minute_bar(day.replace(hour=11, minute=58), 5000.0),
            _minute_bar(day.replace(hour=11, minute=59), 5001.0),
            _minute_bar(day.replace(hour=12, minute=0), 5002.0),
        ]
    }
    frame = build_sigma_curve(bars, target_time=time(12, 0), residual_window=2)
    write_hourly_params(frame, tmp_path)
    params_path = tmp_path / "spx" / "hourly" / "params.json"
    assert params_path.exists()
    payload = json.loads(params_path.read_text(encoding="utf-8"))
    assert payload["symbol"] == "I:SPX"
    assert payload["horizon"] == "hourly"
    minutes = payload["minutes_to_target"]
    assert "0" in minutes and "sigma" in minutes["0"]
    assert payload["kappa_event"] == pytest.approx(1.0, rel=1e-6)


def test_close_write_params_includes_extras(tmp_path) -> None:
    et = ZoneInfo("America/New_York")
    day = datetime(2024, 1, 2, tzinfo=et)
    bars = {
        "I:SPX": [
            _minute_bar(day.replace(hour=15, minute=58), 5000.0),
            _minute_bar(day.replace(hour=15, minute=59), 5001.0),
            _minute_bar(day.replace(hour=16, minute=0), 5002.0),
        ]
    }
    frame = build_sigma_curve(bars, target_time=time(16, 0), residual_window=2)
    extras = {
        "I:SPX": {
            "event_tail": {"tags": ("CPI", "FOMC"), "kappa": 1.55},
            "late_day_variance": {"minutes_threshold": 10, "lambda": 4.2},
        }
    }
    write_close_params(frame, tmp_path, horizon="close", extras=extras)
    params_path = tmp_path / "spx" / "close" / "params.json"
    payload = json.loads(params_path.read_text(encoding="utf-8"))
    assert payload["event_tail"]["tags"] == ["CPI", "FOMC"]
    assert payload["event_tail"]["kappa"] == pytest.approx(1.55, rel=1e-6)
    assert payload["kappa_event"] == pytest.approx(1.55, rel=1e-6)
    assert payload["late_day_variance"]["minutes_threshold"] == 10
    assert payload["late_day_variance"]["lambda"] == pytest.approx(4.2, rel=1e-6)
    assert payload["lambda_close"] == pytest.approx(4.2, rel=1e-6)


def test_extend_window_covers_recent_events() -> None:
    start, end = date(2025, 10, 20), date(2025, 11, 3)
    extended_start, extended_end = extend_calibration_window(start, end, tags=("CPI", "FOMC"))
    assert extended_end == end
    # 2025-10-15 CPI should force the window to extend back at least two days
    assert extended_start <= date(2025, 10, 13)


def test_extend_window_covers_dst_week() -> None:
    start, end = date(2025, 11, 1), date(2025, 11, 3)
    extended_start, _ = extend_calibration_window(start, end, tags=())
    # DST transition occurs 2025-11-02; with pad of 3 days expect <= 2025-10-30
    assert extended_start <= date(2025, 10, 30)
