from __future__ import annotations

import json
from pathlib import Path

from kalshi_alpha.core.pricing import LadderRung
from kalshi_alpha.exec.scanners.scan_index_close import evaluate_close
from kalshi_alpha.strategies.index import CloseInputs
from kalshi_alpha.structures import build_range_structures


def _copy_calibration(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(src.read_bytes())


def test_range_structures_from_close_fixture(isolated_data_roots: tuple[Path, Path]) -> None:
    _, proc_root = isolated_data_roots
    _copy_calibration(
        Path("tests/fixtures/index/ndx/close/params.json"),
        proc_root / "calib" / "index" / "ndx" / "close" / "params.json",
    )
    payload = json.loads(Path("tests/data_fixtures/kalshi/markets_EVT_NDQ100_H1600.json").read_text())
    market = payload["markets"][0]
    strikes = [float(value) for value in market["ladder_strikes"]]
    yes_prices = [float(value) for value in market["ladder_yes_prices"]]
    rungs = [LadderRung(strike=strike, yes_price=price) for strike, price in zip(strikes, yes_prices, strict=True)]
    inputs = CloseInputs(
        series="NASDAQ100",
        current_price=15125.0,
        minutes_to_close=120,
    )
    result = evaluate_close(strikes, yes_prices, inputs, contracts=1, min_ev=0.0)
    strategy_survival = [float(result.survival[strike]) for strike in strikes]
    structures = build_range_structures(
        series="NASDAQ100",
        market_id=market["id"],
        market_ticker=market["ticker"],
        rungs=rungs,
        strategy_survival=strategy_survival,
        contracts=5,
    )
    assert len(structures) == len(rungs) - 1
    first = structures[0]
    assert first.range_probability > 0.0
    assert first.synthetic_price >= 0.0
    assert first.max_loss > 0.0
    assert first.sigma >= 0.0
