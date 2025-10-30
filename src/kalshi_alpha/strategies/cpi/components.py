"""Component signals used in CPI v1.5 nowcasting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping

import polars as pl

from kalshi_alpha.datastore.paths import PROC_ROOT

_AAA_DAILY = "aaa_daily.parquet"
_SHELTER_PARQUET = "cpi_shelter_components.parquet"
_USED_CAR_PARQUET = "used_car_spread.parquet"


@dataclass(frozen=True)
class ComponentWeights:
    gas: float = 0.4
    shelter: float = 0.35
    autos: float = 0.25


def component_signals(
    *,
    fixtures_dir: Path | None = None,
    offline: bool = True,  # Reserved for future use
) -> dict[str, float]:
    """Return lightweight component signals used by CPI v1.5."""

    base_dir = fixtures_dir or PROC_ROOT
    signals: dict[str, float] = {}

    gas_signal = _gas_mtd_signal(base_dir)
    if gas_signal is not None:
        signals["gas"] = gas_signal

    shelter_signal = _shelter_lag_signal(base_dir)
    if shelter_signal is not None:
        signals["shelter"] = shelter_signal

    autos_signal = _used_car_signal(base_dir)
    if autos_signal is not None:
        signals["autos"] = autos_signal

    return signals


def _resolve_candidates(base_dir: Path, filename: str) -> list[Path]:
    candidates = [
        base_dir / filename,
        base_dir / "cpi" / filename,
        base_dir / "aaa" / filename,
        PROC_ROOT / filename,
    ]
    return [path for path in candidates if path.exists()]


def _gas_mtd_signal(base_dir: Path) -> float | None:
    for path in _resolve_candidates(base_dir, _AAA_DAILY):
        try:
            frame = pl.read_parquet(path)
        except Exception:  # pragma: no cover - parquet read fallback
            continue
        if {"date", "price"} - set(frame.columns):
            continue
        frame = frame.with_columns(pl.col("date").cast(pl.Date)).sort("date")
        if frame.is_empty():
            continue
        latest_row = frame.select(pl.all().last())
        latest_date = latest_row["date"][0]
        latest_price = float(latest_row["price"][0])
        month_mask = (pl.col("date").dt.month() == latest_date.month) & (
            pl.col("date").dt.year() == latest_date.year
        )
        month_slice = frame.filter(month_mask)
        if month_slice.is_empty():
            continue
        start_price = float(month_slice["price"][0])
        days = max(month_slice.height, 1)
        return (latest_price - start_price) / days
    return None


def _shelter_lag_signal(base_dir: Path) -> float | None:
    for path in _resolve_candidates(base_dir, _SHELTER_PARQUET):
        try:
            frame = pl.read_parquet(path)
        except Exception:  # pragma: no cover
            continue
        candidate_cols = {"period", "shelter_mom", "lag_proxy"}
        if candidate_cols - set(frame.columns):
            continue
        frame = frame.sort("period")
        if frame.height < 2:
            continue
        latest_proxy = float(frame[-1, "lag_proxy"])
        prior_proxy = float(frame[-2, "lag_proxy"])
        latest_shelter = float(frame[-1, "shelter_mom"])
        shelter_delta = latest_proxy - prior_proxy
        mom_gap = latest_proxy - latest_shelter
        return 0.5 * shelter_delta + 0.5 * mom_gap
    return None


def _used_car_signal(base_dir: Path) -> float | None:
    for path in _resolve_candidates(base_dir, _USED_CAR_PARQUET):
        try:
            frame = pl.read_parquet(path)
        except Exception:  # pragma: no cover
            continue
        if {"period", "spread"} - set(frame.columns):
            continue
        frame = frame.sort("period")
        if frame.is_empty():
            continue
        latest_spread = float(frame["spread"][-1])
        trailing = frame.tail(min(4, frame.height))["spread"].to_list()
        trailing_avg = sum(trailing) / len(trailing)
        return latest_spread - trailing_avg
    return None


def blend_component_shift(signals: Mapping[str, float], weights: ComponentWeights | Mapping[str, float]) -> float:
    if isinstance(weights, ComponentWeights):
        weight_map = {"gas": weights.gas, "shelter": weights.shelter, "autos": weights.autos}
    else:
        weight_map = dict(weights)
    total_weight = sum(weight_map.values())
    if total_weight <= 0:
        return 0.0
    normalized = {key: value / total_weight for key, value in weight_map.items()}
    return sum(normalized.get(key, 0.0) * signals.get(key, 0.0) for key in normalized)
