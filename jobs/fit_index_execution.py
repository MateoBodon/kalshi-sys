"""Fit index fill-alpha and slippage curves from the paper ledger."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

LEDGER_PATH = Path("data/proc/ledger_all.parquet")
DEFAULTS_PATH = Path("data/reference/index_execution_defaults.json")
OUTPUT_ROOT = Path("data/proc/index_exec")
SERIES_ORDER = ("INXU", "NASDAQ100U", "INX", "NASDAQ100")
MIN_ALPHA = 0.02
MAX_ALPHA = 0.95


@dataclass(frozen=True)
class SeriesDefaults:
    base_alpha: float


def _load_defaults() -> dict[str, SeriesDefaults]:
    if not DEFAULTS_PATH.exists():
        raise FileNotFoundError(
            f"Reference defaults missing at {DEFAULTS_PATH}. "
            "Cannot derive base execution parameters."
        )
    payload = json.loads(DEFAULTS_PATH.read_text(encoding="utf-8"))
    series_config = payload.get("series", {})
    defaults: dict[str, SeriesDefaults] = {}
    for series, config in series_config.items():
        try:
            base_alpha = float(config.get("alpha", 0.0))
        except (TypeError, ValueError):
            base_alpha = 0.0
        defaults[series.upper()] = SeriesDefaults(base_alpha=max(base_alpha, 0.0))
    return defaults


def _prepare_frame(frame: pl.DataFrame, series: str) -> pl.DataFrame:
    filtered = frame.filter(pl.col("series") == series)
    if filtered.is_empty():
        return filtered
    return filtered.with_columns(
        pl.col("depth_fraction")
        .fill_null(0.0)
        .clip(lower_bound=0.0, upper_bound=1.0)
        .alias("depth_fraction_clipped"),
        pl.col("delta_p").fill_null(0.0).abs().alias("abs_delta_p"),
        pl.col("spread").fill_null(0.0).abs().alias("spread_abs"),
        pl.col("minutes_to_event")
        .abs()
        .fill_null(0.0)
        .alias("tau_minutes"),
    )


def _design_matrix_alpha(frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    depth_fraction = frame.get_column("depth_fraction_clipped").to_numpy()
    abs_delta = frame.get_column("abs_delta_p").to_numpy()
    tau_minutes = frame.get_column("tau_minutes").to_numpy()
    log_tau = np.log1p(tau_minutes / 60.0)
    ones = np.ones_like(depth_fraction)
    X = np.column_stack((ones, depth_fraction, abs_delta, log_tau))
    y = frame.get_column("fill_ratio").fill_null(0.0).to_numpy()
    return X, y


def _design_matrix_slippage(frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    depth_fraction = frame.get_column("depth_fraction_clipped").to_numpy()
    spread = frame.get_column("spread_abs").to_numpy()
    tau_minutes = frame.get_column("tau_minutes").to_numpy()
    log_tau = np.log1p(tau_minutes / 60.0)
    ones = np.ones_like(depth_fraction)
    interaction = depth_fraction * spread
    X = np.column_stack((ones, depth_fraction, spread, interaction, log_tau))
    y = frame.get_column("slippage_ticks").fill_null(0.0).abs().to_numpy()
    return X, y


def _least_squares(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    if X.size == 0 or y.size == 0:
        raise ValueError("Empty design matrix supplied to regression.")
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def _clip(values: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    return np.clip(values, minimum, maximum)


def _fit_alpha(
    series: str,
    frame: pl.DataFrame,
    defaults: SeriesDefaults,
) -> dict[str, Any]:
    X, y = _design_matrix_alpha(frame)
    beta = _least_squares(X, y)
    predictions = _clip(X @ beta, MIN_ALPHA, MAX_ALPHA)
    base_pred = np.full_like(y, defaults.base_alpha)
    mse_model = float(np.mean((y - predictions) ** 2))
    mse_baseline = float(np.mean((y - base_pred) ** 2))
    result = {
        "series": series,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "features": ["intercept", "depth_fraction", "abs_delta_p", "log_tau_minutes"],
        "coefficients": {
            "intercept": float(beta[0]),
            "depth_fraction": float(beta[1]),
            "abs_delta_p": float(beta[2]),
            "log_tau_minutes": float(beta[3]),
        },
        "clip": {"min": MIN_ALPHA, "max": MAX_ALPHA},
        "baseline_alpha": defaults.base_alpha,
        "model_mse": mse_model,
        "baseline_mse": mse_baseline,
        "num_rows": int(len(y)),
    }
    return result


def _fit_slippage(frame: pl.DataFrame) -> dict[str, Any]:
    X, y = _design_matrix_slippage(frame)
    beta = _least_squares(X, y)
    predictions = np.clip(X @ beta, 0.0, None)
    mse_model = float(np.mean((y - predictions) ** 2))
    mse_baseline = float(np.mean(y**2))
    result = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "features": [
            "intercept",
            "depth_fraction",
            "spread",
            "depth_fraction_x_spread",
            "log_tau_minutes",
        ],
        "coefficients": {
            "intercept": float(beta[0]),
            "depth_fraction": float(beta[1]),
            "spread": float(beta[2]),
            "depth_fraction_x_spread": float(beta[3]),
            "log_tau_minutes": float(beta[4]),
        },
        "model_mse": mse_model,
        "baseline_mse": mse_baseline,
        "num_rows": int(len(y)),
    }
    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit index execution curves from ledger data.")
    parser.add_argument(
        "--ledger",
        type=Path,
        default=LEDGER_PATH,
        help=f"Ledger parquet path (default: {LEDGER_PATH}).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help=f"Output directory for curve JSON files (default: {OUTPUT_ROOT}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.ledger.exists():
        raise FileNotFoundError(f"Ledger parquet not found at {args.ledger}")
    frame = pl.read_parquet(args.ledger)
    defaults_map = _load_defaults()
    for series in SERIES_ORDER:
        series_defaults = defaults_map.get(series)
        if series_defaults is None:
            continue
        prepared = _prepare_frame(frame, series)
        if prepared.is_empty():
            continue
        alpha_payload = _fit_alpha(series, prepared, series_defaults)
        slippage_payload = _fit_slippage(prepared)

        series_dir = args.output_root / series
        _write_json(series_dir / "alpha.json", alpha_payload)
        slippage_payload["series"] = series
        _write_json(series_dir / "slippage.json", slippage_payload)
        print(
            f"[fit_index_execution] {series}: "
            f"rows={alpha_payload['num_rows']} "
            f"alpha_mse={alpha_payload['model_mse']:.4f} "
            f"slip_mse={slippage_payload['model_mse']:.4f}"
        )


if __name__ == "__main__":
    main()
