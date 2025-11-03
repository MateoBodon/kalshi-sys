from __future__ import annotations

from pathlib import Path

import pytest

from kalshi_alpha.core.pricing import LadderBinProbability
from kalshi_alpha.strategies.index import CloseInputs, NoonInputs, close_pmf, noon_pmf
from kalshi_alpha.strategies.index import cdf as index_cdf


def _copy_calibration(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(src.read_bytes())


def test_noon_strategy_uses_calibration(isolated_data_roots: tuple[Path, Path]) -> None:
    _, proc_root = isolated_data_roots
    fixture = Path("tests/fixtures/index/spx/noon/params.json")
    target = proc_root / "calib" / "index" / "spx" / "noon" / "params.json"
    _copy_calibration(fixture, target)
    strikes = [5000.0, 5020.0, 5040.0]
    inputs = NoonInputs(series="INXU", current_price=5035.0, minutes_to_noon=30)
    pmf = noon_pmf(strikes, inputs)
    assert len(pmf) == len(strikes) + 1
    probabilities = [bin_prob.probability for bin_prob in pmf]
    assert pytest.approx(sum(probabilities), rel=1e-6) == 1.0


def test_close_strategy_tail_mass(isolated_data_roots: tuple[Path, Path]) -> None:
    _, proc_root = isolated_data_roots
    fixture = Path("tests/fixtures/index/ndx/close/params.json")
    target = proc_root / "calib" / "index" / "ndx" / "close" / "params.json"
    _copy_calibration(fixture, target)
    strikes = [17800.0, 17900.0]
    inputs = CloseInputs(series="NASDAQ100", current_price=17850.0, minutes_to_close=120)
    pmf = close_pmf(strikes, inputs)
    assert len(pmf) == len(strikes) + 1
    tail_mass = pmf[-1].probability
    assert tail_mass > 0

def test_survival_and_range_mass() -> None:
    strikes = [1.0, 2.0, 3.0]
    # simple pmf where bins increase linearly
    pmf = [
        LadderBinProbability(lower=None, upper=1.0, probability=0.1),
        LadderBinProbability(lower=1.0, upper=2.0, probability=0.3),
        LadderBinProbability(lower=2.0, upper=3.0, probability=0.4),
        LadderBinProbability(lower=3.0, upper=None, probability=0.2),
    ]
    survival = index_cdf.survival_map(strikes, pmf)
    assert survival[1.0] == pytest.approx(0.9)
    assert survival[2.0] == pytest.approx(0.6)
    mass = index_cdf.probability_between(1.0, 2.0, strikes, pmf)
    assert mass == pytest.approx(0.3)
