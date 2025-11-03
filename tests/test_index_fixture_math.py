from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from kalshi_alpha.core.backtest import crps_from_pmf
from kalshi_alpha.core.execution import fillratio, slippage
from kalshi_alpha.core.pricing import LadderBinProbability
from kalshi_alpha.exec.scanners.scan_index_close import evaluate_close
from kalshi_alpha.exec.scanners.scan_index_noon import evaluate_noon
from kalshi_alpha.exec.scanners.utils import pmf_to_survival
from kalshi_alpha.strategies.index import CloseInputs, NoonInputs

ET = ZoneInfo("America/New_York")


def _copy_calibration(fixture: Path, proc_root: Path, symbol: str, horizon: str) -> None:
    target = proc_root / "calib" / "index" / symbol / horizon / "params.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(fixture.read_bytes())


@pytest.mark.parametrize(
    "fixture_name", ["I_SPX_2024-10-21_noon.parquet", "I_SPX_2024-10-22_noon.parquet", "I_SPX_2024-10-23_noon.parquet"]
)
def test_spx_noon_survival_matches_fixture(
    fixture_name: str,
    fixtures_root: Path,
    isolated_data_roots: tuple[Path, Path],
) -> None:
    _, proc_root = isolated_data_roots
    _copy_calibration(Path("tests/fixtures/index/spx/noon/params.json"), proc_root, "spx", "noon")

    path = fixtures_root / "index" / fixture_name
    frame = pl.read_parquet(path)
    rows = frame.with_columns(pl.col("timestamp").dt.convert_time_zone("America/New_York"))
    sample_time = datetime.fromisoformat(f"{fixture_name[6:16]}T11:55:00").replace(tzinfo=ET)
    sample = rows.filter(pl.col("timestamp") == sample_time).row(0, named=True)
    strike = round(float(sample["close"]), 2)
    target_time = datetime.fromisoformat(f"{fixture_name[6:16]}T12:00:00").replace(tzinfo=ET)
    minutes_to_noon = int((target_time - sample["timestamp"]).total_seconds() // 60)
    inputs = NoonInputs(series="INXU", current_price=strike, minutes_to_noon=minutes_to_noon)
    result = evaluate_noon([strike], [0.5], inputs, contracts=1, min_ev=0.0)
    survival = pmf_to_survival(result.pmf, [strike])[0]
    assert survival == pytest.approx(0.5, abs=1e-6)


def test_ndx_close_range_mass_matches_fixture(
    fixtures_root: Path,
    isolated_data_roots: tuple[Path, Path],
) -> None:
    _, proc_root = isolated_data_roots
    _copy_calibration(Path("tests/fixtures/index/ndx/close/params.json"), proc_root, "ndx", "close")

    path = fixtures_root / "index" / "I_NDX_2024-10-21_close.parquet"
    frame = pl.read_parquet(path)
    rows = frame.with_columns(pl.col("timestamp").dt.convert_time_zone("America/New_York"))
    sample = rows.filter(pl.col("timestamp") == datetime(2024, 10, 21, 15, 55, tzinfo=ET)).row(0, named=True)
    close_target = datetime(2024, 10, 21, 16, 0, tzinfo=ET)
    minutes_to_close = int((close_target - sample["timestamp"]).total_seconds() // 60)
    inputs = CloseInputs(series="NASDAQ100", current_price=float(sample["close"]), minutes_to_close=minutes_to_close)
    strikes = [15000.0, 15050.0, 15100.0]
    result = evaluate_close(strikes, [0.4, 0.3, 0.2], inputs, contracts=1, min_ev=0.0)
    # Probability mass for the middle bin (between second strike pair)
    middle_mass = float(result.pmf[2].probability)
    assert middle_mass == pytest.approx(0.407329, rel=1e-6)
    total_mass = sum(prob.probability for prob in result.pmf)
    assert total_mass == pytest.approx(1.0, abs=1e-6)


def _brier_score(pmf: Sequence[LadderBinProbability], actual: float) -> float:
    total = 0.0
    for entry in pmf:
        lower = float("-inf") if entry.lower is None else float(entry.lower)
        upper = float("inf") if entry.upper is None else float(entry.upper)
        indicator = 1.0 if lower <= actual < upper else 0.0
        total += (float(entry.probability) - indicator) ** 2
    return total


def test_close_range_metrics_non_regressive(
    fixtures_root: Path,
    isolated_data_roots: tuple[Path, Path],
) -> None:
    _, proc_root = isolated_data_roots
    _copy_calibration(Path("tests/fixtures/index/ndx/close/params.json"), proc_root, "ndx", "close")

    fixture_names = [
        "I_NDX_2024-10-21_close.parquet",
        "I_NDX_2024-10-22_close.parquet",
        "I_NDX_2024-10-23_close.parquet",
    ]
    for name in fixture_names:
        frame = pl.read_parquet(fixtures_root / "index" / name)
        rows = frame.with_columns(pl.col("timestamp").dt.convert_time_zone("America/New_York"))
        sample_time = datetime.fromisoformat(f"{name[6:16]}T15:55:00").replace(tzinfo=ET)
        close_time = datetime.fromisoformat(f"{name[6:16]}T16:00:00").replace(tzinfo=ET)
        sample = rows.filter(pl.col("timestamp") == sample_time).row(0, named=True)
        close_row = rows.filter(pl.col("timestamp") == close_time).row(0, named=True)
        current_price = float(sample["close"])
        actual_close = float(close_row["close"])
        minutes_to_close = int((close_time - sample_time).total_seconds() // 60)
        strikes = [15000.0, 15050.0, 15100.0]

        updated_inputs = CloseInputs(series="NASDAQ100", current_price=current_price, minutes_to_close=minutes_to_close)
        baseline_inputs = CloseInputs(
            series="NASDAQ100",
            current_price=current_price,
            minutes_to_close=minutes_to_close,
            late_day_bump_override=0.0,
            event_multiplier_override=1.0,
        )

        pmf_updated = evaluate_close(strikes, [0.4, 0.3, 0.2], updated_inputs, contracts=1, min_ev=0.0).pmf
        pmf_baseline = evaluate_close(strikes, [0.4, 0.3, 0.2], baseline_inputs, contracts=1, min_ev=0.0).pmf

        crps_updated = crps_from_pmf(pmf_updated, actual_close)
        crps_baseline = crps_from_pmf(pmf_baseline, actual_close)
        assert crps_updated <= crps_baseline + 1e-6

        brier_updated = _brier_score(pmf_updated, actual_close)
        brier_baseline = _brier_score(pmf_baseline, actual_close)
        assert brier_updated <= brier_baseline + 5e-4

        event_inputs = CloseInputs(
            series="NASDAQ100",
            current_price=current_price,
            minutes_to_close=minutes_to_close,
            event_tags=("FOMC",),
        )
        pmf_event = evaluate_close(strikes, [0.4, 0.3, 0.2], event_inputs, contracts=1, min_ev=0.0).pmf
        event_tail = float(pmf_event[0].probability + pmf_event[-1].probability)
        updated_tail = float(pmf_updated[0].probability + pmf_updated[-1].probability)
        assert event_tail >= updated_tail


def test_default_alpha_slippage_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(fillratio, "STATE_PATH", state_dir / "fill_alpha.json")
    monkeypatch.setattr(slippage, "STATE_PATH", state_dir / "slippage.json")

    alpha_value = fillratio.load_alpha("INXU")
    assert alpha_value == pytest.approx(0.45, rel=1e-6)

    model = slippage.load_slippage_model("NASDAQ100", mode="depth")
    assert model is not None
    assert model.mode == "depth"
    assert model.impact_cap == pytest.approx(0.032, rel=1e-6)
