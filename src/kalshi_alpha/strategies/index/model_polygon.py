"""Simple Polygon-only distribution model for SPX/NDX ladders."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, time
from math import sqrt
from pathlib import Path
from typing import Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pandas.api import types as pd_types
from scipy import stats

from kalshi_alpha.datastore.paths import PROC_ROOT
from kalshi_alpha.drivers.polygon_index.symbols import resolve_series
from kalshi_alpha.strategies import base
from kalshi_alpha.core.pricing import LadderBinProbability

ET = ZoneInfo("America/New_York")
PARAM_ROOT = PROC_ROOT / "calib" / "index_polygon"


@dataclass(frozen=True)
class ModelParams:
    symbol: str
    horizon: str
    mean_per_sqrt_hour: float
    std_per_sqrt_hour: float
    df: float | None
    sample_count: int
    minutes_range: tuple[float, float]
    generated_at: str

    def as_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "horizon": self.horizon,
            "mean_per_sqrt_hour": self.mean_per_sqrt_hour,
            "std_per_sqrt_hour": self.std_per_sqrt_hour,
            "df": self.df,
            "sample_count": self.sample_count,
            "minutes_range": list(self.minutes_range),
            "generated_at": self.generated_at,
        }


def _ensure_timestamp(series: pd.Series) -> pd.Series:
    ts = series.copy()
    if not pd_types.is_datetime64_any_dtype(ts):
        ts = pd.to_datetime(ts)
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    return ts.dt.tz_convert(ET)


def _target_price_map(df: pd.DataFrame, target_clock: time) -> dict[pd.Timestamp, float]:
    """Return a per-day target price map (ET) using last price <= target."""

    targets: dict[pd.Timestamp, float] = {}
    for trading_day, group in df.groupby("trading_day"):
        target_dt = datetime.combine(trading_day, target_clock, tzinfo=ET)
        before = group[group["timestamp_et"] <= target_dt]
        if not before.empty:
            row = before.loc[before["timestamp_et"].idxmax()]
            targets[target_dt] = float(row["price"])
            continue
        after = group[group["timestamp_et"] >= target_dt]
        if after.empty:
            continue
        row = after.loc[after["timestamp_et"].idxmin()]
        targets[target_dt] = float(row["price"])
    return targets


def _normalized_returns(
    df: pd.DataFrame,
    *,
    target_clock: time,
) -> tuple[np.ndarray, list[float]]:
    targets = _target_price_map(df, target_clock)
    deltas: list[float] = []
    scaled: list[float] = []
    minutes_list: list[float] = []

    for _, row in df.iterrows():
        trading_day = row["trading_day"]
        target_dt = datetime.combine(trading_day, target_clock, tzinfo=ET)
        target_price = targets.get(target_dt)
        if target_price is None:
            continue
        now_ts = row["timestamp_et"]
        minutes_to_target = (target_dt - now_ts).total_seconds() / 60.0
        if minutes_to_target <= 0:
            continue
        delta = target_price - float(row["price"])
        norm = delta / sqrt(max(minutes_to_target / 60.0, 1e-6))
        deltas.append(delta)
        scaled.append(norm)
        minutes_list.append(minutes_to_target)
    return np.asarray(scaled), minutes_list


def fit_from_panel(
    panel: Mapping | "pd.DataFrame",
    horizon: str,
    *,
    symbols: Sequence[str] | None = None,
) -> dict[str, ModelParams]:
    """Fit per-symbol Student-t parameters from a Polygon index panel."""

    horizon_norm = horizon.lower()
    if horizon_norm not in {"noon", "close"}:
        raise ValueError("horizon must be 'noon' or 'close'")
    target_clock = time(12, 0) if horizon_norm == "noon" else time(16, 0)

    if isinstance(panel, pd.DataFrame):
        df = panel.copy()
    else:
        # Polars DataFrame or similar mapping.
        import polars as pl  # local import to avoid hard dependency in tests

        if isinstance(panel, pl.DataFrame):
            df = panel.to_pandas()
        else:
            df = pd.DataFrame(panel)

    if "timestamp_et" not in df.columns:
        raise ValueError("panel must contain a timestamp_et column in ET")
    df["timestamp_et"] = _ensure_timestamp(df["timestamp_et"])
    df["trading_day"] = df["timestamp_et"].dt.date

    if symbols is not None:
        want = {sym.upper() for sym in symbols}
        df = df[df["symbol"].str.upper().isin(want)]
    if df.empty:
        raise ValueError("panel contains no rows for requested symbols")

    grouped = df.groupby("symbol")
    results: dict[str, ModelParams] = {}
    for symbol, frame in grouped:
        scaled, minutes_list = _normalized_returns(frame, target_clock=target_clock)
        if scaled.size == 0:
            continue
        mean_norm = float(np.mean(scaled))
        std_norm = float(np.std(scaled, ddof=1)) or 1e-6
        # Light-touch heavy-tail allowance: df grows slowly with sample size.
        df_value = max(3.0, min(50.0, 2.0 + scaled.size / 10.0))
        params = ModelParams(
            symbol=str(symbol),
            horizon=horizon_norm,
            mean_per_sqrt_hour=mean_norm,
            std_per_sqrt_hour=std_norm,
            df=df_value,
            sample_count=int(scaled.size),
            minutes_range=(float(min(minutes_list)), float(max(minutes_list))),
            generated_at=datetime.now(tz=UTC).isoformat(),
        )
        results[str(symbol)] = params
    if not results:
        raise ValueError("no model parameters could be fit from the panel")
    return results


def predict_pmf(
    now_state: Mapping[str, float],
    params: Mapping[str, object],
    ladder_strikes: Sequence[float],
) -> list[LadderBinProbability]:
    """Generate a ladder PMF from fitted parameters."""

    symbol = str(now_state.get("symbol") or params.get("symbol") or "")
    if not symbol:
        raise ValueError("now_state must include a symbol")
    normalized_symbol = symbol.upper().replace("_", ":")
    minutes_to_target = float(now_state.get("minutes_to_target", 0.0))
    if minutes_to_target <= 0:
        raise ValueError("minutes_to_target must be positive")
    price_now = float(now_state.get("price") or now_state.get("current_price") or 0.0)
    if not np.isfinite(price_now):
        raise ValueError("current price is required")

    def _as_mapping(obj: object) -> Mapping[str, object]:
        if isinstance(obj, ModelParams):
            return obj.as_dict()
        if isinstance(obj, Mapping):
            return obj
        raise TypeError(f"Unsupported parameter object for {symbol}: {type(obj)}")

    # Params may already be narrowed to a single symbol; otherwise look up.
    param_obj: object | None = params
    if isinstance(params, Mapping):
        if "symbols" in params:
            symbols_map = params.get("symbols")  # type: ignore[assignment]
            if isinstance(symbols_map, Mapping):
                param_obj = symbols_map.get(normalized_symbol) or symbols_map.get(symbol)
        elif normalized_symbol in params:
            param_obj = params.get(normalized_symbol)
    if param_obj is None:
        raise KeyError(f"No parameters found for symbol {normalized_symbol}")
    symbol_params = _as_mapping(param_obj)

    mean_norm = float(symbol_params.get("mean_per_sqrt_hour", 0.0))
    std_norm = float(symbol_params.get("std_per_sqrt_hour", 1.0))
    df_value = symbol_params.get("df")

    dt_hours = max(minutes_to_target / 60.0, 1e-6)
    delta_loc = mean_norm * sqrt(dt_hours)
    delta_scale = max(std_norm * sqrt(dt_hours), 1e-6)
    distribution = (
        stats.t(df=float(df_value), loc=delta_loc, scale=delta_scale)
        if df_value is not None
        else stats.norm(loc=delta_loc, scale=delta_scale)
    )

    bins = base.ladder_bins(ladder_strikes)
    weights = []
    for lower, upper in bins:
        lower_delta = -np.inf if lower is None else float(lower) - price_now
        upper_delta = np.inf if upper is None else float(upper) - price_now
        prob = float(distribution.cdf(upper_delta) - distribution.cdf(lower_delta))
        weights.append(max(prob, 0.0))
    total = sum(weights)
    if total <= 0:
        raise ValueError("distribution produced zero probability mass")
    normalized = base.normalize(weights)
    return [
        LadderBinProbability(lower=lower, upper=upper, probability=prob)
        for (lower, upper), prob in zip(bins, normalized, strict=True)
    ]


def save_params(params: Mapping[str, ModelParams], path: Path) -> Path:
    """Persist parameter dict to JSON."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {symbol: model.as_dict() for symbol, model in params.items()}
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"symbols": payload}, handle, indent=2, sort_keys=True)
    return path


def load_params(path: Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload


def params_path(series: str, horizon: str, root: Path | None = None) -> Path:
    """Return default params.json path for a given series/horizon pair."""

    normalized = series.upper()
    resolve_series(normalized)  # validate
    slug = normalized.replace(":", "_")
    root_path = Path(root) if root is not None else PARAM_ROOT
    return root_path / slug / horizon.lower() / "params.json"


__all__ = [
    "ModelParams",
    "PARAM_ROOT",
    "fit_from_panel",
    "predict_pmf",
    "save_params",
    "load_params",
    "params_path",
]
