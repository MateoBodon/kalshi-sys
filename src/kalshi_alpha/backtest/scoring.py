"""Shared scoring utilities for index backtests."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import polars as pl

from kalshi_alpha.backtest.generate_dataset import DEFAULT_OUTPUT as DEFAULT_DATASET_PATH
from kalshi_alpha.strategies.index import (
    CLOSE_CALIBRATION_PATH,
    HOURLY_CALIBRATION_PATH,
)
from kalshi_alpha.strategies.index import cdf as index_cdf

DEFAULT_CONTRACTS = 100
PIT_BINS = 10


@dataclass(frozen=True)
class ScoreSample:
    symbol: str
    series: str
    target_timestamp: datetime
    minutes_to_target: int
    current_price: float
    realized: float
    mean: float
    std: float
    pit: float
    crps: float
    brier: float
    taker_price: float
    taker_fee: float
    taker_ev: float
    maker_yes_ev: float
    maker_no_ev: float
    contracts: int


@dataclass(frozen=True)
class ScoreReport:
    horizon: str
    samples: list[ScoreSample]
    start_timestamp: datetime
    end_timestamp: datetime
    summary: Mapping[str, Mapping[str, float]]
    pit_histogram: Sequence[Mapping[str, float]]


CalibrationLoader = Callable[[str, str], index_cdf.SigmaCalibration]


def evaluate_backtest(
    *,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    output_dir: Path,
    horizon: str,
    polygon_to_series: Mapping[str, str],
    contracts: int = DEFAULT_CONTRACTS,
    calibration_loader: CalibrationLoader | None = None,
) -> ScoreReport:
    if horizon not in {"hourly", "close"}:
        raise ValueError("horizon must be 'hourly' or 'close'")
    if not polygon_to_series:
        raise ValueError("polygon_to_series mapping cannot be empty")

    frame = _load_dataset(dataset_path)
    if frame.is_empty():
        raise RuntimeError("BLOCKED: backtest dataset empty — run generate_dataset first")
    selected_rows = _select_terminal_rows(frame, horizon)
    if not selected_rows:
        raise RuntimeError(f"BLOCKED: no {horizon} samples available in dataset")

    loader = calibration_loader or _default_calibration_loader
    calibrations = {
        (symbol, horizon): loader(symbol, horizon)
        for symbol in polygon_to_series
    }

    samples: list[ScoreSample] = []
    for row in selected_rows:
        polygon_symbol = str(row["symbol"])
        series = polygon_to_series.get(polygon_symbol)
        if series is None:
            continue
        realized = row["target_on_before"]
        if realized is None:
            continue
        realized_value = float(realized)
        minutes_to_target = int(row["minutes_to_target"])
        current_price = float(row["price_close"])
        calibration = calibrations[(polygon_symbol, horizon)]
        mean_value, std_value = _resolve_mean_std(
            calibration,
            horizon=horizon,
            minutes=minutes_to_target,
            current_price=current_price,
        )
        pit = _normal_cdf(realized_value, mean_value, std_value)
        crps = _normal_crps(realized_value, mean_value, std_value)
        brier = (pit - 1.0) ** 2  # event {X <= realized} always true
        taker_price = pit
        taker_fee = _index_taker_fee(contracts, taker_price)
        taker_ev = contracts * (1.0 - taker_price) - taker_fee
        maker_yes_ev = contracts * (taker_price - 1.0)
        maker_no_ev = contracts * (1.0 - taker_price)
        samples.append(
            ScoreSample(
                symbol=polygon_symbol,
                series=series,
                target_timestamp=row["target_timestamp"],
                minutes_to_target=minutes_to_target,
                current_price=current_price,
                realized=realized_value,
                mean=mean_value,
                std=std_value,
                pit=pit,
                crps=crps,
                brier=brier,
                taker_price=taker_price,
                taker_fee=taker_fee,
                taker_ev=taker_ev,
                maker_yes_ev=maker_yes_ev,
                maker_no_ev=maker_no_ev,
                contracts=contracts,
            )
        )

    if not samples:
        raise RuntimeError(f"BLOCKED: dataset missing realized prices for {horizon} horizon")

    samples.sort(key=lambda item: (item.symbol, item.target_timestamp))
    summary = _summaries(samples)
    histogram = _pit_histogram(samples, bins=PIT_BINS)
    start_ts = min(sample.target_timestamp for sample in samples)
    end_ts = max(sample.target_timestamp for sample in samples)

    report = ScoreReport(
        horizon=horizon,
        samples=samples,
        start_timestamp=start_ts,
        end_timestamp=end_ts,
        summary=summary,
        pit_histogram=histogram,
    )
    _write_outputs(report, output_dir)
    return report


def _load_dataset(path: Path) -> pl.DataFrame:
    resolved = path.resolve()
    if not resolved.exists():
        raise RuntimeError(f"BLOCKED: dataset not found at {resolved}")
    return pl.read_parquet(resolved)


def _select_terminal_rows(frame: pl.DataFrame, horizon: str) -> list[dict[str, object]]:
    iterator = frame.iter_rows(named=True)
    filtered: list[dict[str, object]] = [row for row in iterator if str(row["target_type"]).lower() == horizon]
    filtered.sort(key=lambda row: (str(row["symbol"]), row["target_timestamp"], int(row["minutes_to_target"])))
    best: dict[tuple[str, datetime], dict[str, object]] = {}
    for row in filtered:
        key = (str(row["symbol"]), row["target_timestamp"])
        current_best = best.get(key)
        if current_best is None or int(row["minutes_to_target"]) < int(current_best["minutes_to_target"]):
            best[key] = row
    return list(best.values())


def _default_calibration_loader(symbol: str, horizon: str) -> index_cdf.SigmaCalibration:
    if horizon == "hourly":
        return index_cdf.load_calibration(HOURLY_CALIBRATION_PATH, symbol, horizon="hourly")
    if horizon == "close":
        return index_cdf.load_calibration(CLOSE_CALIBRATION_PATH, symbol, horizon="close")
    raise ValueError(f"Unsupported horizon for calibration loader: {horizon}")


def _resolve_mean_std(
    calibration: index_cdf.SigmaCalibration,
    *,
    horizon: str,
    minutes: int,
    current_price: float,
) -> tuple[float, float]:
    drift = calibration.drift(minutes)
    if horizon == "hourly":
        sigma_curve = calibration.sigma(minutes)
        residual = calibration.residual_std or 0.0
        effective_sigma = max(float(sigma_curve), float(residual), 0.5)
        return float(current_price) + float(drift), float(effective_sigma)
    if horizon == "close":
        sigma_curve = calibration.sigma(minutes)
        residual = calibration.residual_std or 0.0
        base_sigma = max(float(sigma_curve), float(residual), 1.0)
        variance = base_sigma * base_sigma
        effective_sigma = max(math.sqrt(variance), 1.0)
        return float(current_price) + float(drift), float(effective_sigma)
    raise ValueError(f"Unsupported horizon for mean/std resolution: {horizon}")


def _normal_cdf(value: float, mean: float, std: float) -> float:
    if std <= 0.0:
        std = 1e-6
    z = (value - mean) / std
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _normal_pdf(value: float, mean: float, std: float) -> float:
    if std <= 0.0:
        std = 1e-6
    z = (value - mean) / std
    return (1.0 / (std * math.sqrt(2.0 * math.pi))) * math.exp(-0.5 * z * z)


def _normal_crps(value: float, mean: float, std: float) -> float:
    if std <= 0.0:
        std = 1e-6
    z = (value - mean) / std
    cdf = _normal_cdf(value, mean, std)
    pdf = _normal_pdf(value, mean, std)
    phi = pdf * std
    bracket = z * (2.0 * cdf - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi)
    return max(std * bracket, 0.0)


def _index_taker_fee(contracts: int, price: float) -> float:
    price_clamped = min(max(float(price), 0.0), 1.0)
    raw = 0.035 * float(contracts) * price_clamped * (1.0 - price_clamped)
    return math.ceil(raw * 100.0) / 100.0


def _summaries(samples: Sequence[ScoreSample]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    grouped: dict[str, list[ScoreSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.series, []).append(sample)
    grouped["ALL"] = samples

    for series, rows in grouped.items():
        if not rows:
            continue
        count = len(rows)
        mean_abs_error = sum(abs(row.realized - row.mean) for row in rows) / count
        mean_crps = sum(row.crps for row in rows) / count
        mean_brier = sum(row.brier for row in rows) / count
        mean_pit = sum(row.pit for row in rows) / count
        mean_taker_ev = sum(row.taker_ev for row in rows) / count
        summary[series] = {
            "count": float(count),
            "mean_abs_error": mean_abs_error,
            "mean_crps": mean_crps,
            "mean_brier": mean_brier,
            "mean_pit": mean_pit,
            "mean_taker_ev": mean_taker_ev,
        }
    return summary


def _pit_histogram(samples: Sequence[ScoreSample], *, bins: int) -> list[dict[str, float]]:
    if bins <= 0:
        raise ValueError("bins must be positive")
    counts = [0] * bins
    for sample in samples:
        pit = min(max(sample.pit, 0.0), 1.0)
        index = min(bins - 1, int(pit * bins))
        counts[index] += 1
    total = sum(counts) or 1
    histogram: list[dict[str, float]] = []
    for idx, count in enumerate(counts):
        start = idx / bins
        end = (idx + 1) / bins
        histogram.append(
            {
                "bin_start": start,
                "bin_end": end,
                "count": float(count),
                "frequency": count / total,
            }
        )
    return histogram


def _write_outputs(report: ScoreReport, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ev_table.csv"
    records = [
        {
            "symbol": sample.symbol,
            "series": sample.series,
            "target_timestamp": sample.target_timestamp.astimezone(UTC).isoformat(),
            "minutes_to_target": sample.minutes_to_target,
            "current_price": sample.current_price,
            "realized": sample.realized,
            "mean": sample.mean,
            "std": sample.std,
            "pit": sample.pit,
            "crps": sample.crps,
            "brier": sample.brier,
            "taker_price": sample.taker_price,
            "taker_fee": sample.taker_fee,
            "taker_ev": sample.taker_ev,
            "maker_yes_ev": sample.maker_yes_ev,
            "maker_no_ev": sample.maker_no_ev,
            "contracts": sample.contracts,
        }
        for sample in report.samples
    ]
    pl.DataFrame(records).write_csv(csv_path)

    markdown_path = output_dir / "metrics.md"
    lines: list[str] = []
    horizon_title = "Hourly" if report.horizon == "hourly" else "Close"
    lines.append(f"# {horizon_title} Backtest Metrics")
    lines.append("")
    lines.append(f"- Samples: {len(report.samples)}")
    lines.append(f"- Window: {report.start_timestamp.date()} → {report.end_timestamp.date()}")
    lines.append(f"- Contracts per trade: {report.samples[0].contracts}")
    lines.append(f"- Taker fee formula: ceil(0.035 × contracts × price × (1 - price) × 100)/100")
    lines.append("")
    lines.append("| Series | N | Mean Abs Error | Mean CRPS | Mean Brier | Mean PIT | Mean Taker EV ($) |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for series, stats in report.summary.items():
        count = int(stats["count"])
        lines.append(
            f"| {series} | {count} | {stats['mean_abs_error']:.4f} | {stats['mean_crps']:.4f} | "
            f"{stats['mean_brier']:.4f} | {stats['mean_pit']:.4f} | {stats['mean_taker_ev']:.2f} |"
        )
    lines.append("")
    lines.append("## PIT Histogram (deciles)")
    lines.append("")
    lines.append("| Bin | Count | Frequency |")
    lines.append("| --- | ---: | ---: |")
    for entry in report.pit_histogram:
        bin_label = f"{entry['bin_start']:.1f}–{entry['bin_end']:.1f}"
        lines.append(f"| {bin_label} | {int(entry['count'])} | {entry['frequency']:.3f} |")
    lines.append("")

    markdown_path.write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "ScoreReport",
    "ScoreSample",
    "evaluate_backtest",
]
