from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl

from kalshi_alpha.core.archive.archiver import archive_scan
from kalshi_alpha.core.archive.scorecards import build_replay_scorecard
from kalshi_alpha.core.kalshi_api import Event, Market, Orderbook, Series


def _series() -> Series:
    return Series(id="S1", ticker="CPI", name="CPI Headline")


def _event(series_id: str) -> Event:
    return Event(id="E1", series_id=series_id, ticker="CPI-E1", title="CPI Release")


def _market(event_id: str) -> Market:
    return Market(
        id="M1",
        event_id=event_id,
        ticker="CPI-M1",
        title="CPI Bin",
        ladder_strikes=[270.0, 275.0, 280.0],
        ladder_yes_prices=[0.45, 0.35, 0.25],
    )


def _orderbook(market_id: str) -> Orderbook:
    return Orderbook(
        market_id=market_id,
        bids=[{"price": 0.44, "size": 20}, {"price": 0.34, "size": 10}],
        asks=[{"price": 0.45, "size": 15}, {"price": 0.36, "size": 12}],
    )


def test_build_replay_scorecard(tmp_path: Path) -> None:
    series = _series()
    event = _event(series.id)
    market = _market(event.id)
    client_stub = SimpleNamespace(base_url="https://example", use_offline=True)
    manifest_path = archive_scan(
        series,
        client_stub,
        [event],
        [market],
        {market.id: _orderbook(market.id)},
        out_dir=tmp_path / "raw",
    )

    proposals_path = tmp_path / "proposals.json"
    proposals_payload = {
        "proposals": [
            {
                "market_id": market.id,
                "market_ticker": market.ticker,
                "strike": market.ladder_strikes[0],
                "side": "YES",
                "contracts": 5,
                "maker_ev": 1.0,
                "taker_ev": -1.0,
            }
        ]
    }
    proposals_path.write_text(json.dumps(proposals_payload, indent=2), encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["proposals_path"] = str(proposals_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    fixtures_root = Path(__file__).resolve().parent / "fixtures"
    scorecard = build_replay_scorecard(
        manifest_path=manifest_path,
        model_version="v15",
        driver_fixtures=fixtures_root,
    )

    assert isinstance(scorecard.summary, pl.DataFrame)
    assert set(scorecard.summary.columns) >= {
        "market_id",
        "mean_abs_cdf_delta",
        "max_abs_cdf_delta",
        "prob_sum_gap",
        "max_kink",
    }
    assert isinstance(scorecard.cdf_deltas, pl.DataFrame)
    assert set(scorecard.cdf_deltas.columns) >= {
        "market_id",
        "strike",
        "delta",
    }
