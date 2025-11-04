from __future__ import annotations

from pathlib import Path

from kalshi_alpha.core.pricing import OrderSide
from kalshi_alpha.exec.scanners.scan_index_close import evaluate_close
from kalshi_alpha.exec.scanners.scan_index_hourly import evaluate_hourly
from kalshi_alpha.strategies.index import CloseInputs, HourlyInputs


def _copy_calibration(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(src.read_bytes())


def test_evaluate_hourly_yields_opportunities(isolated_data_roots: tuple[Path, Path]) -> None:
    _, proc_root = isolated_data_roots
    _copy_calibration(
        Path("tests/fixtures/index/spx/hourly/params.json"),
        proc_root / "calib" / "index" / "spx" / "hourly" / "params.json",
    )
    strikes = [5000.0, 5020.0, 5040.0]
    yes_prices = [0.45, 0.35, 0.25]
    inputs = HourlyInputs(series="INXU", current_price=5035.0, minutes_to_target=30)
    result = evaluate_hourly(strikes, yes_prices, inputs, contracts=1, min_ev=0.0)
    assert result.opportunities
    first = result.opportunities[0]
    assert first.side is OrderSide.YES
    assert first.range_mass >= 0.0
    assert result.rule is not None
    assert result.rule.series == "INXU"
    assert "12:00" in result.rule.evaluation_time_et


def test_evaluate_close_range_mass(isolated_data_roots: tuple[Path, Path]) -> None:
    _, proc_root = isolated_data_roots
    _copy_calibration(
        Path("tests/fixtures/index/ndx/close/params.json"),
        proc_root / "calib" / "index" / "ndx" / "close" / "params.json",
    )
    strikes = [17800.0, 17900.0]
    yes_prices = [0.4, 0.25]
    inputs = CloseInputs(series="NASDAQ100", current_price=17850.0, minutes_to_close=120)
    result = evaluate_close(strikes, yes_prices, inputs, contracts=1, min_ev=0.0)
    assert result.tail_mass > 0.0
    assert any(op.maker_ev >= 0.0 for op in result.opportunities)
    assert result.rule is not None
    assert result.rule.series == "NASDAQ100"
    assert "16:00" in result.rule.evaluation_time_et
