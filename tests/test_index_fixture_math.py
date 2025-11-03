from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from kalshi_alpha.core.execution import fillratio, slippage
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
    assert middle_mass == pytest.approx(0.407273, rel=1e-6)


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
