from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

import math

from kalshi_alpha.backtest.scoring import (
    _index_taker_fee,
    _normal_crps,
    _normal_cdf,
    evaluate_backtest,
)

ET = ZoneInfo("America/New_York")


def test_index_taker_fee_rounds_up() -> None:
    fee = _index_taker_fee(100, 0.50)
    assert fee == 0.88


def test_normal_crps_at_mean_matches_closed_form() -> None:
    crps = _normal_crps(0.0, 0.0, 1.0)
    expected = (2 ** 0.5 - 1.0) / (math.pi ** 0.5)
    assert abs(crps - expected) < 1e-6


@dataclass
class StubCalibration:
    sigma_value: float
    residual_std: float
    drift_value: float

    event_tail: None = None
    late_day_variance: None = None

    def sigma(self, minutes: int) -> float:
        return self.sigma_value

    def drift(self, minutes: int) -> float:
        return self.drift_value


def _build_dataset(tmp_path: Path) -> Path:
    target_et = datetime(2025, 3, 14, 12, 0, tzinfo=ET)
    rows = [
        {
            "symbol": "I:SPX",
            "trading_day": target_et.date(),
            "observation_timestamp": target_et.astimezone(UTC),
            "observation_timestamp_et": target_et,
            "target_timestamp": target_et,
            "target_type": "hourly",
            "minutes_to_target": 5,
            "price_close": 4995.0,
            "ewma_sigma_now": 10.0,
            "micro_drift": 0.5,
            "target_on_before": 5000.0,
        },
        {
            "symbol": "I:SPX",
            "trading_day": target_et.date(),
            "observation_timestamp": target_et.astimezone(UTC),
            "observation_timestamp_et": target_et,
            "target_timestamp": target_et,
            "target_type": "hourly",
            "minutes_to_target": 0,
            "price_close": 5005.0,
            "ewma_sigma_now": 9.0,
            "micro_drift": 0.4,
            "target_on_before": 5000.0,
        },
    ]
    frame = pl.DataFrame(rows)
    path = tmp_path / "dataset.parquet"
    frame.write_parquet(path)
    return path


def _stub_loader(symbol: str, horizon: str) -> StubCalibration:
    return StubCalibration(sigma_value=5.0, residual_std=1.0, drift_value=0.0)


def test_evaluate_backtest_writes_outputs(tmp_path: Path) -> None:
    dataset = _build_dataset(tmp_path)
    output_dir = tmp_path / "reports"
    report = evaluate_backtest(
        dataset_path=dataset,
        output_dir=output_dir,
        horizon="hourly",
        polygon_to_series={"I:SPX": "INXU"},
        calibration_loader=_stub_loader,
        contracts=100,
    )
    csv_path = output_dir / "ev_table.csv"
    md_path = output_dir / "metrics.md"
    assert csv_path.exists()
    assert md_path.exists()
    assert report.samples and report.summary
    sample = report.samples[0]
    pit = _normal_cdf(sample.realized, sample.mean, sample.std)
    assert abs(sample.pit - pit) < 1e-6
