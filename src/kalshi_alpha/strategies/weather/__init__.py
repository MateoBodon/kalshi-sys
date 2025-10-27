"""Weather strategy stubs enforcing NOAA/NWS DCR settlement requirements."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import mean as stat_mean
from statistics import pstdev
from typing import Any

import polars as pl

from kalshi_alpha.core.backtest import crps_from_pmf
from kalshi_alpha.core.pricing import LadderBinProbability
from kalshi_alpha.datastore.paths import PROC_ROOT
from kalshi_alpha.strategies import base

SETTLEMENT_SOURCE = "NWS Daily Climate Report"
CALIBRATION_PATH = PROC_ROOT / "weather_calib.parquet"

WeatherCalibration = dict[str, dict[str, dict[str, float]]]


def _to_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Unable to coerce value {value!r} to float")


@dataclass(frozen=True)
class EnsembleMember:
    value: float
    weight: float = 1.0


@dataclass(frozen=True)
class WeatherInputs:
    forecast_high: float
    bias: float = 0.0
    spread: float = 3.0
    station: str | None = None


def pmf(
    strikes: Sequence[float],
    ensemble: Iterable[EnsembleMember] | None = None,
    inputs: WeatherInputs | None = None,
    *,
    calibration: WeatherCalibration | None = None,
) -> list[LadderBinProbability]:
    members: list[EnsembleMember]
    if inputs is not None:
        mean = inputs.forecast_high + inputs.bias
        spread = max(min(inputs.spread, 3.0) * 0.5, 0.5)
        applied_cal = calibration or _load_calibration()
        if applied_cal and inputs.station:
            stations = applied_cal.get("stations", {})
            station_cal = stations.get(inputs.station)
            if station_cal is not None:
                mean += station_cal.get("bias", 0.0)
                spread = max(spread, station_cal.get("spread", spread))
        members = [
            EnsembleMember(value=mean, weight=0.7),
            EnsembleMember(value=mean + spread, weight=0.15),
            EnsembleMember(value=mean - spread, weight=0.15),
        ]
    else:
        members = list(ensemble or [])
        if not members:
            members = [EnsembleMember(value=strike, weight=1.0) for strike in strikes]

    total_weight = sum(member.weight for member in members)
    if total_weight <= 0:
        raise ValueError("ensemble weights must sum to a positive value")

    distribution = {round(member.value, 2): member.weight / total_weight for member in members}
    return base.grid_distribution_to_pmf(distribution)


def settlement_reminder(station_id: str) -> str:
    return (
        f"Settlement for station {station_id} must reference the {SETTLEMENT_SOURCE}. "
        "Ensure Daily Climate Report access before trade execution."
    )


def calibrate(history: Sequence[Mapping[str, object]]) -> pl.DataFrame:
    if not history:
        raise ValueError("weather history required for calibration")

    residuals: defaultdict[str, list[float]] = defaultdict(list)
    spreads: defaultdict[str, list[float]] = defaultdict(list)
    for entry in history:
        station = str(entry["station"])
        forecast = _to_float(entry["forecast_high"]) + _to_float(entry.get("bias", 0.0))
        actual = _to_float(entry["actual_high"])
        residuals[station].append(actual - forecast)
        spreads[station].append(_to_float(entry.get("spread", 3.0)))

    station_params: dict[str, dict[str, float]] = {}
    for station, residuals_list in residuals.items():
        bias = stat_mean(residuals_list)
        base_spread = stat_mean(spreads[station])
        if len(residuals_list) > 1:
            spread_estimate = pstdev(residuals_list)
        else:
            spread_estimate = base_spread
        station_params[station] = {
            "bias": bias,
            "spread": max(spread_estimate, 0.5),
            "observations": float(len(residuals_list)),
        }

    calibration_payload: WeatherCalibration = {"stations": station_params}

    records: list[dict[str, Any]] = [
        {
            "record_type": "station",
            "station": station,
            "bias": params["bias"],
            "spread": params["spread"],
            "observations": params["observations"],
            "forecast_high": None,
            "actual_high": None,
            "crps": None,
            "baseline_crps": None,
        }
        for station, params in station_params.items()
    ]

    model_crps: list[float] = []
    baseline_crps: list[float] = []
    for entry in history:
        station = str(entry["station"])
        inputs = WeatherInputs(
            forecast_high=_to_float(entry["forecast_high"]),
            bias=_to_float(entry.get("bias", 0.0)),
            spread=_to_float(entry.get("spread", 3.0)),
            station=station,
        )
        strike_start = int(inputs.forecast_high) - 10
        strikes = [float(value) for value in range(strike_start, strike_start + 21, 2)]
        pmf_values = pmf(strikes, inputs=inputs, calibration=calibration_payload)
        crps_value = crps_from_pmf(pmf_values, _to_float(entry["actual_high"]))
        model_crps.append(crps_value)

        baseline_inputs = WeatherInputs(
            forecast_high=_to_float(entry["forecast_high"]),
            bias=0.0,
            spread=_to_float(entry.get("spread", 3.0)),
            station=None,
        )
        baseline_pmf = pmf(strikes, inputs=baseline_inputs, calibration={"stations": {}})
        baseline_value = crps_from_pmf(baseline_pmf, _to_float(entry["actual_high"]))
        baseline_crps.append(baseline_value)

        records.append(
            {
                "record_type": "evaluation",
                "station": station,
                "bias": station_params[station]["bias"],
                "spread": station_params[station]["spread"],
                "observations": None,
                "forecast_high": _to_float(entry["forecast_high"]),
                "actual_high": _to_float(entry["actual_high"]),
                "crps": crps_value,
                "baseline_crps": baseline_value,
            }
        )

    summary_row = {
        "record_type": "params",
        "station": None,
        "bias": None,
        "spread": None,
        "observations": None,
        "forecast_high": None,
        "actual_high": None,
        "crps": sum(model_crps) / len(model_crps),
        "baseline_crps": sum(baseline_crps) / len(baseline_crps),
    }
    records.insert(0, summary_row)

    frame = pl.DataFrame(records)
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(CALIBRATION_PATH)
    return frame


def _load_calibration() -> WeatherCalibration | None:
    if not CALIBRATION_PATH.exists():
        return None
    frame = pl.read_parquet(CALIBRATION_PATH)
    stations = {}
    for row in frame.iter_rows(named=True):
        if row.get("record_type") != "station" or row.get("station") is None:
            continue
        stations[str(row["station"])] = {
            "bias": float(row.get("bias", 0.0)),
            "spread": float(row.get("spread", 0.5)),
        }
    if not stations:
        return None
    return {"stations": stations}
