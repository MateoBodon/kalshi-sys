from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from kalshi_alpha.core.kalshi_api import KalshiPublicClient
from kalshi_alpha.core.risk import PALGuard, PALPolicy
from kalshi_alpha.exec.runners import scan_ladders


@pytest.mark.usefixtures("isolated_data_roots")
def test_snapshot_parity(
    tmp_path: Path,
    fixtures_root: Path,
    offline_fixtures_root: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    client = KalshiPublicClient(offline_dir=fixtures_root / "kalshi", use_offline=True)
    guard = PALGuard(PALPolicy(series="CPI", default_max_loss=1_000.0, per_strike={}))

    outcome = scan_ladders.scan_series(
        series="CPI",
        client=client,
        min_ev=0.01,
        contracts=5,
        pal_guard=guard,
        driver_fixtures=offline_fixtures_root,
        strategy_name="auto",
        maker_only=True,
        allow_tails=False,
        risk_manager=None,
        max_var=None,
        offline=True,
        sizing_mode="fixed",
        kelly_cap=0.25,
    )

    proposals = outcome.proposals
    assert proposals, "expected scan_series to generate proposals"

    proposals_path = scan_ladders.write_proposals(
        series="CPI",
        proposals=proposals,
        output_dir=tmp_path / "exec/proposals",
    )

    manifest_path, replay_path = scan_ladders._archive_and_replay(
        client=client,
        series=outcome.series,
        events=outcome.events,
        markets=outcome.markets,
        orderbooks=outcome.books_at_scan,
        proposals_path=proposals_path,
        driver_fixtures=offline_fixtures_root,
        scanner_fixtures=fixtures_root,
        model_metadata=outcome.model_metadata,
    )

    assert manifest_path is not None and manifest_path.exists()
    assert replay_path is not None and replay_path.exists()

    replay_df = pl.read_parquet(replay_path)
    assert not replay_df.is_empty()
    lookup = {
        (str(row["market_id"]), float(row["strike"])): row
        for row in replay_df.to_dicts()
    }

    for proposal in proposals:
        key = (proposal.market_id, float(proposal.strike))
        replay_row = lookup.get(key)
        assert replay_row is not None, f"missing replay row for {key}"
        replay_per = replay_row.get("maker_ev_per_contract_replay", replay_row.get("maker_ev_replay", 0.0))
        assert abs(proposal.maker_ev_per_contract - float(replay_per)) <= 1e-6
