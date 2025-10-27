from __future__ import annotations

from pathlib import Path

import polars as pl

from kalshi_alpha.core.kalshi_api import KalshiPublicClient, Orderbook
from kalshi_alpha.exec.runners import scan_ladders


def test_archive_and_replay_creates_artifacts(
    tmp_path: Path,
    fixtures_root: Path,
    offline_fixtures_root: Path,
    isolated_data_roots: tuple[Path, Path],
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    client = KalshiPublicClient(
        offline_dir=fixtures_root / "kalshi",
        use_offline=True,
    )

    pal_guard = scan_ladders.PALGuard(
        scan_ladders.PALPolicy(series="CPI", default_max_loss=1_000.0, per_strike={})
    )

    outcome = scan_ladders.scan_series(
        series="CPI",
        client=client,
        min_ev=0.01,
        contracts=5,
        pal_guard=pal_guard,
        driver_fixtures=offline_fixtures_root,
        strategy_name="auto",
        maker_only=True,
        allow_tails=False,
        risk_manager=None,
        max_var=None,
        offline=True,
        sizing_mode="kelly",
        kelly_cap=0.25,
    )

    proposals = outcome.proposals
    proposals_path = scan_ladders.write_proposals(
        series="CPI",
        proposals=proposals,
        output_dir=tmp_path / "exec/proposals",
    )

    orderbooks: dict[str, Orderbook] = {}
    for market in outcome.markets:
        try:
            orderbooks[market.id] = client.get_orderbook(market.id)
        except Exception:
            continue

    manifest_path = scan_ladders._archive_and_replay(
        client=client,
        series=outcome.series,
        events=outcome.events,
        markets=outcome.markets,
        orderbooks=orderbooks,
        proposals_path=proposals_path,
        driver_fixtures=offline_fixtures_root,
        scanner_fixtures=fixtures_root,
    )

    assert manifest_path is not None
    assert manifest_path.exists()

    replay_path = Path("reports/_artifacts/replay_ev.parquet")
    assert replay_path.exists()
    frame = pl.read_parquet(replay_path)
    assert "maker_ev_replay" in frame.columns

    manifest = manifest_path.read_text(encoding="utf-8")
    assert "proposals_path" in manifest
