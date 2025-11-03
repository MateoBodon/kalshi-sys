from __future__ import annotations

from pathlib import Path

from kalshi_alpha.core.pricing import OrderSide
from kalshi_alpha.exec.scanners.scan_index_close import evaluate_close
from kalshi_alpha.exec.scanners.scan_index_noon import evaluate_noon
from kalshi_alpha.strategies.index import CloseInputs, NoonInputs


def _copy_calibration(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(src.read_bytes())


def test_evaluate_noon_yields_opportunities(isolated_data_roots: tuple[Path, Path]) -> None:
    _, proc_root = isolated_data_roots
    _copy_calibration(
        Path("tests/fixtures/index/index_noon_calibration.parquet"),
        proc_root / "index_noon_calibration.parquet",
    )
    strikes = [5000.0, 5020.0, 5040.0]
    yes_prices = [0.45, 0.35, 0.25]
    inputs = NoonInputs(series="INXU", current_price=5035.0, minutes_to_noon=30)
    result = evaluate_noon(strikes, yes_prices, inputs, contracts=1, min_ev=0.0)
    assert result.opportunities
    first = result.opportunities[0]
    assert first.side is OrderSide.YES
    assert first.range_mass >= 0.0


def test_evaluate_close_range_mass(isolated_data_roots: tuple[Path, Path]) -> None:
    _, proc_root = isolated_data_roots
    _copy_calibration(
        Path("tests/fixtures/index/index_close_calibration.parquet"),
        proc_root / "index_close_calibration.parquet",
    )
    strikes = [17800.0, 17900.0]
    yes_prices = [0.4, 0.25]
    inputs = CloseInputs(series="NASDAQ100", current_price=17850.0, minutes_to_close=120)
    result = evaluate_close(strikes, yes_prices, inputs, contracts=1, min_ev=0.0)
    assert result.tail_mass > 0.0
    assert any(op.maker_ev >= 0.0 for op in result.opportunities)
