from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from kalshi_alpha.drivers.nws_cli import parse_multi_station_report
from kalshi_alpha.strategies.weather import WeatherInputs, calibrate, pmf

FIXTURE_ROOT = Path(__file__).parent / "fixtures"


def test_weather_calibration_per_station(isolated_data_roots: tuple[Path, Path]) -> None:
    _, proc_root = isolated_data_roots
    history = json.loads((FIXTURE_ROOT / "weather" / "history.json").read_text(encoding="utf-8"))["history"]

    frame = calibrate(history)

    path = proc_root / "weather_calib.parquet"
    assert path.exists()

    summary = frame.filter(pl.col("record_type") == "params")
    row = summary.row(0, named=True)
    assert row["crps"] < row["baseline_crps"]

    stations = frame.filter(pl.col("record_type") == "station")
    assert stations.height == len({entry["station"] for entry in history})

    sample = history[0]
    inputs = WeatherInputs(
        forecast_high=float(sample["forecast_high"]),
        bias=float(sample.get("bias", 0.0)),
        spread=float(sample.get("spread", 3.0)),
        station=sample["station"],
    )
    strikes = [inputs.forecast_high - 6, inputs.forecast_high, inputs.forecast_high + 6]
    ladder = pmf(strikes, inputs=inputs)
    assert abs(sum(seg.probability for seg in ladder) - 1.0) < 1e-6


def test_parse_multi_station_report() -> None:
    records = parse_multi_station_report(FIXTURE_ROOT / "weather" / "nws_dcr_multi.txt")
    assert len(records) == 3
    assert {record.station_id for record in records} == {"KBOS", "KATL", "KSEA"}
