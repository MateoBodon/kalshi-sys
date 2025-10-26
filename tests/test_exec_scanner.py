from __future__ import annotations

import json
from pathlib import Path

from kalshi_alpha.core.kalshi_api import KalshiPublicClient
from kalshi_alpha.core.risk import PALGuard, PALPolicy
from kalshi_alpha.exec.runners.scan_ladders import scan_series, write_proposals
from kalshi_alpha.exec.scanners import cpi


def test_scanner_generates_positive_maker_ev(fixtures_root: Path, tmp_path: Path) -> None:
    client = KalshiPublicClient(offline_dir=fixtures_root / "kalshi", use_offline=True)
    policy = PALPolicy(series="CPI", default_max_loss=10_000.0)
    guard = PALGuard(policy)

    proposals = scan_series(
        series="CPI",
        client=client,
        min_ev=0.01,
        contracts=5,
        pal_guard=guard,
        driver_fixtures=fixtures_root / "drivers",
        strategy_name="auto",
        maker_only=True,
        allow_tails=False,
    )
    assert proposals, "Expected scanner to produce proposals with positive EV"

    for proposal in proposals:
        assert proposal.taker_ev == 0.0
        assert proposal.maker_ev_per_contract >= 0.01

    exposures = guard.exposure_snapshot()
    assert all(value <= guard.policy.default_max_loss + 1e-6 for value in exposures.values())

    ladder = client.get_markets(client.get_events(client.get_series()[0].id)[0].id)[0]
    strikes = ladder.ladder_strikes
    strategy_pmf = cpi.strategy_pmf(strikes, fixtures_dir=fixtures_root / "drivers")
    max_prob = max(bin_prob.probability for bin_prob in strategy_pmf[: len(strikes)])
    mode_indices = [
        idx
        for idx, bin_prob in enumerate(strategy_pmf[: len(strikes)])
        if bin_prob.probability >= max_prob - 1e-9
    ]
    allowed_indices: set[int] = set()
    for idx in mode_indices:
        allowed_indices.update(range(max(idx - 1, 0), min(idx + 1, len(strikes) - 1) + 1))
    strike_to_index = {float(strike): index for index, strike in enumerate(strikes)}
    for proposal in proposals:
        assert strike_to_index[proposal.strike] in allowed_indices

    output_path = write_proposals(series="CPI", proposals=proposals, output_dir=tmp_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["series"] == "CPI"
    assert payload["proposals"]
