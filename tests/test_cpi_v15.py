from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from kalshi_alpha.core.backtest import crps_from_pmf
from kalshi_alpha.strategies import base
from kalshi_alpha.strategies.cpi import (
    CPIInputs,
    map_to_ladder_bins,
    nowcast,
    nowcast_v15,
)


def test_cpi_v15_component_fixtures_improve_crps_and_shoulders(tmp_path: Path) -> None:
    fixtures_dir = tmp_path / "cpi"
    fixtures_dir.mkdir()
    _write_gas_fixture(fixtures_dir)
    _write_shelter_fixture(fixtures_dir)
    _write_used_car_fixture(fixtures_dir)

    inputs = CPIInputs(cleveland_nowcast=0.36, latest_release_mom=0.34, aaa_delta=0.0)
    actual_mom = 0.30

    distribution_v0 = nowcast(inputs)
    pmf_v0 = base.grid_distribution_to_pmf(distribution_v0)
    crps_v0 = crps_from_pmf(pmf_v0, actual_mom)

    distribution_v15 = nowcast_v15(inputs, fixtures_dir=fixtures_dir)
    pmf_v15 = base.grid_distribution_to_pmf(distribution_v15)
    crps_v15 = crps_from_pmf(pmf_v15, actual_mom)

    assert crps_v15 < crps_v0

    strikes = [0.25, 0.3, 0.35, 0.4, 0.45]
    bins_v0 = map_to_ladder_bins(strikes, distribution_v0)
    bins_v15 = map_to_ladder_bins(strikes, distribution_v15)

    shoulder_range = (0.3, 0.35)

    def _probability_for_range(bins: list, lower: float | None, upper: float | None) -> float:
        for entry in bins:
            if entry.lower == lower and entry.upper == upper:
                return entry.probability
        raise AssertionError("expected ladder range not present")

    assert _probability_for_range(bins_v15, *shoulder_range) > _probability_for_range(bins_v0, *shoulder_range)


def test_cpi_v15_fallback_to_v0_when_components_missing(tmp_path: Path) -> None:
    inputs = CPIInputs(cleveland_nowcast=0.33, latest_release_mom=0.31, aaa_delta=0.02)

    distribution_v0 = nowcast(inputs)
    distribution_v15 = nowcast_v15(inputs, fixtures_dir=tmp_path)

    assert set(distribution_v0.keys()) == set(distribution_v15.keys())
    for point, probability in distribution_v0.items():
        assert distribution_v15[point] == pytest.approx(probability, rel=1e-9, abs=1e-9)


def test_cpi_v15_respects_config_weights(tmp_path: Path) -> None:
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    _write_gas_fixture(fixtures_dir)
    _write_shelter_fixture(fixtures_dir)
    _write_used_car_fixture(fixtures_dir)

    config_path = tmp_path / "cpi.yaml"
    config_path.write_text(
        """
component_weights:
  gas: 0.0
  shelter: 0.0
  autos: 0.0
blend:
  cleveland: 1.0
  components: 0.0
variance:
  component_scale: 0.0
""",
        encoding="utf-8",
    )

    inputs = CPIInputs(cleveland_nowcast=0.42, latest_release_mom=0.38, aaa_delta=0.01)
    distribution_v0 = nowcast(inputs)
    distribution_cfg = nowcast_v15(
        inputs,
        fixtures_dir=fixtures_dir,
        config_path=config_path,
    )

    assert set(distribution_v0.keys()) == set(distribution_cfg.keys())
    for point, probability in distribution_v0.items():
        assert distribution_cfg[point] == pytest.approx(probability, rel=1e-9, abs=1e-9)


def _write_gas_fixture(base: Path) -> None:
    dates = [date(2024, 9, 1) + timedelta(days=offset) for offset in range(6)]
    prices = [3.40, 3.34, 3.30, 3.26, 3.22, 3.18]
    frame = pl.DataFrame({"date": dates, "price": prices})
    frame.write_parquet(base / "aaa_daily.parquet")


def _write_shelter_fixture(base: Path) -> None:
    frame = pl.DataFrame(
        {
            "period": ["2024-08", "2024-09", "2024-10"],
            "shelter_mom": [0.35, 0.33, 0.31],
            "lag_proxy": [0.34, 0.30, 0.25],
        }
    )
    frame.write_parquet(base / "cpi_shelter_components.parquet")


def _write_used_car_fixture(base: Path) -> None:
    frame = pl.DataFrame(
        {
            "period": ["2024-07", "2024-08", "2024-09", "2024-10"],
            "spread": [0.05, 0.01, -0.02, -0.08],
        }
    )
    frame.write_parquet(base / "used_car_spread.parquet")
