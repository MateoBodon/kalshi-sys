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
from kalshi_alpha.drivers.calendar.loader import calendar_tags_for
from kalshi_alpha.exec.scanners.scan_index_close import evaluate_close
from kalshi_alpha.exec.scanners.scan_index_noon import evaluate_noon
from kalshi_alpha.exec.scanners.utils import pmf_to_survival
from kalshi_alpha.strategies.index import CloseInputs, NoonInputs

ET = ZoneInfo("America/New_York")

_INDEX_FIXTURE_DIR = Path("tests/data_fixtures/index")
_SPX_NOON_FIXTURES = sorted(path.name for path in _INDEX_FIXTURE_DIR.glob("I_SPX_*_noon.parquet"))
_NDX_CLOSE_FIXTURES = sorted(path.name for path in _INDEX_FIXTURE_DIR.glob("I_NDX_*_close.parquet"))

assert _SPX_NOON_FIXTURES, "Expected at least one SPX noon fixture"
assert _NDX_CLOSE_FIXTURES, "Expected at least one NDX close fixture"


def _copy_calibration(fixture: Path, proc_root: Path, symbol: str, horizon: str) -> None:
    target = proc_root / "calib" / "index" / symbol / horizon / "params.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(fixture.read_bytes())


def _strike_grid(center: float, step: float = 50.0) -> list[float]:
    base = round(center / step) * step
    return [base - step, base, base + step]


@pytest.mark.parametrize("fixture_name", _SPX_NOON_FIXTURES)
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


@pytest.mark.parametrize("fixture_name", _NDX_CLOSE_FIXTURES)
def test_ndx_close_range_mass_matches_fixture(
    fixture_name: str,
    fixtures_root: Path,
    isolated_data_roots: tuple[Path, Path],
) -> None:
    _, proc_root = isolated_data_roots
    _copy_calibration(Path("tests/fixtures/index/ndx/close/params.json"), proc_root, "ndx", "close")

    path = fixtures_root / "index" / fixture_name
    frame = pl.read_parquet(path)
    rows = frame.with_columns(pl.col("timestamp").dt.convert_time_zone("America/New_York"))
    date_label = fixture_name[6:16]
    sample_time = datetime.fromisoformat(f"{date_label}T15:55:00").replace(tzinfo=ET)
    close_time = datetime.fromisoformat(f"{date_label}T16:00:00").replace(tzinfo=ET)
    sample = rows.filter(pl.col("timestamp") == sample_time).row(0, named=True)
    close_row = rows.filter(pl.col("timestamp") == close_time).row(0, named=True)
    minutes_to_close = int((close_time - sample["timestamp"]).total_seconds() // 60)
    current_price = float(sample["close"])
    strikes = _strike_grid(current_price, step=50.0)
    inputs = CloseInputs(series="NASDAQ100", current_price=current_price, minutes_to_close=minutes_to_close)
    result = evaluate_close(strikes, [0.4, 0.3, 0.2], inputs, contracts=1, min_ev=0.0)
    total_mass = sum(prob.probability for prob in result.pmf)
    assert total_mass == pytest.approx(1.0, abs=1e-6)

    actual_close = float(close_row["close"])

    def _contains(entry: LadderBinProbability, value: float) -> bool:
        lower = float("-inf") if entry.lower is None else float(entry.lower)
        upper = float("inf") if entry.upper is None else float(entry.upper)
        return lower <= value < upper

    target_bin = next(entry for entry in result.pmf if _contains(entry, actual_close))
    assert float(target_bin.probability) > 0.0
    for entry in result.pmf:
        probability = float(entry.probability)
        assert 0.0 <= probability <= 1.0


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

    for name in _NDX_CLOSE_FIXTURES:
        frame = pl.read_parquet(fixtures_root / "index" / name)
        rows = frame.with_columns(pl.col("timestamp").dt.convert_time_zone("America/New_York"))
        sample_time = datetime.fromisoformat(f"{name[6:16]}T15:55:00").replace(tzinfo=ET)
        close_time = datetime.fromisoformat(f"{name[6:16]}T16:00:00").replace(tzinfo=ET)
        sample = rows.filter(pl.col("timestamp") == sample_time).row(0, named=True)
        close_row = rows.filter(pl.col("timestamp") == close_time).row(0, named=True)
        current_price = float(sample["close"])
        actual_close = float(close_row["close"])
        minutes_to_close = int((close_time - sample_time).total_seconds() // 60)
        strikes = _strike_grid(current_price, step=50.0)

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
        assert crps_updated <= crps_baseline + 0.02

        brier_updated = _brier_score(pmf_updated, actual_close)
        brier_baseline = _brier_score(pmf_baseline, actual_close)
        assert brier_updated <= brier_baseline + 5e-3

        event_inputs = CloseInputs(
            series="NASDAQ100",
            current_price=current_price,
            minutes_to_close=minutes_to_close,
            event_tags=calendar_tags_for(close_time),
        )
        pmf_event = evaluate_close(strikes, [0.4, 0.3, 0.2], event_inputs, contracts=1, min_ev=0.0).pmf
        event_tail = float(pmf_event[0].probability + pmf_event[-1].probability)
        updated_tail = float(pmf_updated[0].probability + pmf_updated[-1].probability)
        if event_inputs.event_tags:
            assert event_tail >= updated_tail
        else:
            assert event_tail == pytest.approx(updated_tail, abs=1e-6)


def test_close_variance_bump_window_only(
    fixtures_root: Path,
    isolated_data_roots: tuple[Path, Path],
) -> None:
    _, proc_root = isolated_data_roots
    _copy_calibration(Path("tests/fixtures/index/ndx/close/params.json"), proc_root, "ndx", "close")

    frame = pl.read_parquet(fixtures_root / "index" / "I_NDX_2024-10-21_close.parquet")
    rows = frame.with_columns(pl.col("timestamp").dt.convert_time_zone("America/New_York"))
    sample_time = datetime(2024, 10, 21, 15, 45, tzinfo=ET)
    close_time = datetime(2024, 10, 21, 16, 0, tzinfo=ET)
    sample = rows.filter(pl.col("timestamp") == sample_time).row(0, named=True)
    minutes_to_close = int((close_time - sample_time).total_seconds() // 60)
    current_price = float(sample["close"])
    strikes = _strike_grid(current_price, step=50.0)
    default_inputs = CloseInputs(series="NASDAQ100", current_price=current_price, minutes_to_close=minutes_to_close)
    baseline_inputs = CloseInputs(
        series="NASDAQ100",
        current_price=current_price,
        minutes_to_close=minutes_to_close,
        late_day_bump_override=0.0,
    )
    pmf_default = evaluate_close(strikes, [0.4, 0.3, 0.2], default_inputs, contracts=1, min_ev=0.0).pmf
    pmf_baseline = evaluate_close(strikes, [0.4, 0.3, 0.2], baseline_inputs, contracts=1, min_ev=0.0).pmf
    for default_prob, baseline_prob in zip(pmf_default, pmf_baseline, strict=True):
        assert float(default_prob.probability) == pytest.approx(float(baseline_prob.probability), abs=1e-9)


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
