"""CPI ladder scanner utilities."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from kalshi_alpha.core.pricing import LadderBinProbability
from kalshi_alpha.drivers.bls_cpi import load_latest_release
from kalshi_alpha.strategies.cpi import CPIInputs, map_to_ladder_bins, nowcast


def strategy_pmf(strikes: Sequence[float], *, fixtures_dir: Path) -> list[LadderBinProbability]:
    release_path = fixtures_dir / "bls_cpi_latest.json"
    release = load_latest_release(offline_path=release_path)
    inputs = CPIInputs(latest_release_mom=release.seasonally_adjusted_mom)
    distribution = nowcast(inputs)
    return map_to_ladder_bins(strikes, distribution)
