"""Weather strategy stubs enforcing NOAA/NWS DCR settlement requirements."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from kalshi_alpha.core.pricing import LadderBinProbability
from kalshi_alpha.strategies import base

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
