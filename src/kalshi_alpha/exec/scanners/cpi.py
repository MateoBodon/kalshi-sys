"""CPI ladder scanner utilities."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from kalshi_alpha.core.pricing import LadderBinProbability
from kalshi_alpha.drivers.bls_cpi import fetch_latest_release
from kalshi_alpha.drivers.cleveland_nowcast import fetch_nowcast
from kalshi_alpha.strategies.cpi import CPIInputs, map_to_ladder_bins, nowcast, nowcast_v15


def strategy_pmf(
    strikes: Sequence[float],
    *,
    fixtures_dir: Path,
    offline: bool = False,
    model_version: str = "v15",
) -> list[LadderBinProbability]:
    bls_fixtures = fixtures_dir / "bls_cpi"
    release = fetch_latest_release(
        offline=offline,
        fixtures_dir=bls_fixtures if bls_fixtures.exists() else fixtures_dir,
    )
    cleveland_dir = fixtures_dir / "cleveland_nowcast"
    nowcast_series = fetch_nowcast(
        offline=offline,
        fixtures_dir=cleveland_dir if cleveland_dir.exists() else fixtures_dir,
    )
    headline = nowcast_series.get("headline")
    inputs = CPIInputs(
        cleveland_nowcast=headline.value if headline else release.mom_sa,
        latest_release_mom=release.mom_sa,
        aaa_delta=0.0,
    )
    version = (model_version or "v15").lower()
    if version == "v15":
        distribution = nowcast_v15(inputs, fixtures_dir=fixtures_dir, offline=offline)
    else:
        distribution = nowcast(inputs)
    return map_to_ladder_bins(strikes, distribution)
