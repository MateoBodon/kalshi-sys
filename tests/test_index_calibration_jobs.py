from __future__ import annotations

import json
from datetime import date, time
from pathlib import Path

import polars as pl
import pytest

from jobs._index_calibration import (
    build_sigma_curve,
    event_tail_multiplier,
    extend_calibration_window,
    late_day_lambda,
)
from jobs.calibrate_close import _write_params as write_close_params
from jobs.calibrate_hourly import _write_params as write_hourly_params
from kalshi_alpha.drivers.polygon_index.client import MinuteBar

_FIXTURE_ROOT = Path("tests/data_fixtures/index")
_SPX_NOON_FIXTURE = _FIXTURE_ROOT / "I_SPX_2024-10-21_noon.parquet"
_SPX_CLOSE_EVENT_FIXTURE = _FIXTURE_ROOT / "I_SPX_2024-09-11_close.parquet"
_SPX_CLOSE_BASE_FIXTURE = _FIXTURE_ROOT / "I_SPX_2024-10-21_close.parquet"


def _load_minute_bars(path: Path) -> list[MinuteBar]:
    frame = pl.read_parquet(path)
    bars: list[MinuteBar] = []
    for row in frame.iter_rows(named=True):
        vwap_value = row.get("vwap")
        trades_value = row.get("trades")
        trades_converted = None
        if trades_value is not None:
            try:
                trades_converted = int(trades_value)
            except (TypeError, ValueError):
                trades_converted = None
        bars.append(
            MinuteBar(
                timestamp=row["timestamp"],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                vwap=float(vwap_value) if vwap_value is not None else None,
                trades=trades_converted,
            )
        )
    return bars


def test_build_sigma_curve_computes_minutes() -> None:
    bars = {"I:SPX": _load_minute_bars(_SPX_NOON_FIXTURE)}
    frame = build_sigma_curve(bars, target_time=time(12, 0), residual_window=5)
    assert isinstance(frame, pl.DataFrame)
    assert not frame.is_empty()
    minutes = frame.get_column("minutes_to_target").to_list()
    assert 0 in minutes
    assert max(minutes) == 15


def test_write_params_emits_json(tmp_path) -> None:
    bars = {"I:SPX": _load_minute_bars(_SPX_NOON_FIXTURE)}
    frame = build_sigma_curve(bars, target_time=time(12, 0), residual_window=5)
    write_hourly_params(frame, tmp_path)
    params_path = tmp_path / "spx" / "hourly" / "params.json"
    assert params_path.exists()
    payload = json.loads(params_path.read_text(encoding="utf-8"))
    assert payload["symbol"] == "I:SPX"
    assert payload["horizon"] == "hourly"
    minutes = payload["minutes_to_target"]
    assert "0" in minutes and "sigma" in minutes["0"]
    assert payload["kappa_event"] == pytest.approx(1.0, rel=1e-6)


def test_write_params_hourly_alias(tmp_path) -> None:
    bars = {"I:SPX": _load_minute_bars(_SPX_NOON_FIXTURE)}
    frame = build_sigma_curve(bars, target_time=time(12, 0), residual_window=5)
    write_hourly_params(frame, tmp_path, horizon="noon")
    params_path = tmp_path / "spx" / "noon" / "params.json"
    assert params_path.exists()
    payload = json.loads(params_path.read_text(encoding="utf-8"))
    assert payload["horizon"] == "noon"
    assert payload["kappa_event"] == pytest.approx(1.0, rel=1e-6)


def test_close_write_params_includes_extras(tmp_path) -> None:
    combined_bars = _load_minute_bars(_SPX_CLOSE_EVENT_FIXTURE) + _load_minute_bars(_SPX_CLOSE_BASE_FIXTURE)
    frame, records = build_sigma_curve(
        {"I:SPX": combined_bars},
        target_time=time(16, 0),
        residual_window=15,
        return_records=True,
    )
    kappa = event_tail_multiplier(
        records,
        "I:SPX",
        window=60,
        tags=("CPI", "FOMC"),
        clamp=(1.0, 1.75),
    )
    late_day = late_day_lambda(
        records,
        frame,
        "I:SPX",
        window=10,
        tags=("CPI", "FOMC"),
    )
    extras = {"I:SPX": {"event_tail": {"tags": ("CPI", "FOMC"), "kappa": kappa}}}
    if late_day is not None:
        extras["I:SPX"]["late_day_variance"] = late_day
    write_close_params(frame, tmp_path, horizon="close", extras=extras)
    params_path = tmp_path / "spx" / "close" / "params.json"
    payload = json.loads(params_path.read_text(encoding="utf-8"))
    assert payload["event_tail"]["tags"] == ["CPI", "FOMC"]
    assert 1.0 <= payload["event_tail"]["kappa"] <= 1.8
    assert payload["event_tail"]["kappa"] == pytest.approx(kappa, rel=1e-6)
    assert payload["kappa_event"] == pytest.approx(kappa, rel=1e-6)
    if late_day is not None:
        assert payload["late_day_variance"]["minutes_threshold"] == late_day["minutes_threshold"]
        assert payload["late_day_variance"]["lambda"] == pytest.approx(late_day["lambda"], rel=1e-6)
        assert payload["lambda_close"] == pytest.approx(late_day["lambda"], rel=1e-6)
    else:
        assert payload["lambda_close"] == pytest.approx(0.0, abs=1e-9)


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
