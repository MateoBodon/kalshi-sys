"""Weather strategy stubs enforcing NOAA/NWS DCR settlement requirements."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import mean as stat_mean
from statistics import pstdev

from kalshi_alpha.core.pricing import LadderBinProbability
from kalshi_alpha.datastore.paths import PROC_ROOT
from kalshi_alpha.strategies import base

WeatherCalibration = dict[str, dict[str, dict[str, float]]]


def _to_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Unable to coerce value {value!r} to float")

SETTLEMENT_SOURCE = "NWS Daily Climate Report"


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
) -> list[LadderBinProbability]:
    members: list[EnsembleMember]
    if inputs is not None:
        mean = inputs.forecast_high + inputs.bias
        spread = max(min(inputs.spread, 3.0) * 0.5, 0.5)
        calibration = _load_calibration()
        if calibration and inputs.station:
            stations = calibration.get("stations", {})
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


CALIBRATION_DIR = PROC_ROOT / "calibration"
CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
CALIBRATION_PATH = CALIBRATION_DIR / "weather.json"


def calibrate(history: Sequence[Mapping[str, object]]) -> WeatherCalibration:
    if not history:
        raise ValueError("weather history required for calibration")
    residuals: dict[str, list[float]] = defaultdict(list)
    spreads: dict[str, list[float]] = defaultdict(list)
    for entry in history:
        station = str(entry["station"])
        forecast = _to_float(entry["forecast_high"]) + _to_float(entry.get("bias", 0.0))
        actual = _to_float(entry["actual_high"])
        residuals[station].append(actual - forecast)
        spreads[station].append(_to_float(entry.get("spread", 3.0)))
    stations_payload: dict[str, dict[str, float]] = {}
    for station, residuals_list in residuals.items():
        bias = stat_mean(residuals_list)
        base_spread = stat_mean(spreads[station])
        if len(residuals_list) > 1:
            spread_estimate = pstdev(residuals_list)
        else:
            spread_estimate = base_spread
        stations_payload[station] = {
            "bias": bias,
            "spread": max(spread_estimate, 0.5),
        }
    payload: WeatherCalibration = {
        "stations": {
            station: station_cal
            for station, station_cal in stations_payload.items()
        }
    }
    CALIBRATION_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _load_calibration() -> WeatherCalibration | None:
    if not CALIBRATION_PATH.exists():
        return None
    try:
        data = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    stations_raw = data.get("stations")
    if not isinstance(stations_raw, dict):
        return None
    stations: dict[str, dict[str, float]] = {}
    for station, values in stations_raw.items():
        if not isinstance(values, dict):
            continue
        stations[station] = {
            "bias": float(values.get("bias", 0.0)),
            "spread": float(values.get("spread", 0.5)),
        }
    return {"stations": stations}
