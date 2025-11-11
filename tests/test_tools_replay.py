from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from tools import replay as tools_replay

from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def _et(year: int, month: int, day: int, hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=ET)


def _write_series_run(
    *,
    root: Path,
    trading_day: date,
    timestamp_label: str,
    series: str,
    market_id: str,
    market_ticker: str,
    hour_et: int,
    generated_at_et: datetime,
    fixtures_root: Path,
) -> None:
    series_dir = root / "data" / "raw" / "kalshi" / trading_day.isoformat() / timestamp_label / series
    series_dir.mkdir(parents=True, exist_ok=True)
    orderbooks_dir = series_dir / "orderbooks"
    orderbooks_dir.mkdir(parents=True, exist_ok=True)
    orderbook_path = orderbooks_dir / f"{market_id}.json"
    orderbook_payload = {
        "market_id": market_id,
        "bids": [{"price": 0.48, "size": 10}, {"price": 0.47, "size": 8}],
        "asks": [{"price": 0.52, "size": 9}, {"price": 0.53, "size": 7}],
    }
    orderbook_path.write_text(json.dumps(orderbook_payload), encoding="utf-8")
    markets_payload = [
        {
            "event_id": f"EVT_{series}_H{hour_et:04d}",
            "id": market_id,
            "ladder_strikes": [5000.0, 5100.0],
            "ladder_yes_prices": [0.48, 0.35],
            "ticker": market_ticker,
            "title": f"{series} window",
        }
    ]
    (series_dir / "markets.json").write_text(json.dumps(markets_payload), encoding="utf-8")
    (series_dir / "series.json").write_text(
        json.dumps({"id": f"{series}_SERIES", "ticker": series, "name": f"{series} name"}),
        encoding="utf-8",
    )
    (series_dir / "events.json").write_text(
        json.dumps([{"id": f"EVT_{series}_H{hour_et:04d}", "series_id": f"{series}_SERIES"}]),
        encoding="utf-8",
    )
    proposals_dir = root / "exec" / "proposals" / "index_live" / series
    proposals_dir.mkdir(parents=True, exist_ok=True)
    proposals_path = proposals_dir / f"{trading_day.isoformat()}_{timestamp_label}.json"
    proposals_payload = {
        "series": series,
        "generated_at": generated_at_et.astimezone(UTC).isoformat(),
        "proposals": [
            {
                "market_id": market_id,
                "market_ticker": market_ticker,
                "strike": 5000.0,
                "side": "YES",
                "contracts": 1,
                "maker_ev": 0.25,
                "taker_ev": -0.25,
                "maker_ev_per_contract": 0.25,
                "taker_ev_per_contract": -0.25,
                "strategy_probability": 0.75,
                "market_yes_price": 0.48,
                "survival_market": 0.48,
                "survival_strategy": 0.75,
                "max_loss": 0.52,
                "strategy": series,
                "series": series,
            }
        ],
    }
    proposals_path.write_text(json.dumps(proposals_payload), encoding="utf-8")
    manifest = {
        "generated_at": generated_at_et.astimezone(UTC).isoformat(),
        "paths": {
            "markets": "markets.json",
            "orderbooks": [f"orderbooks/{market_id}.json"],
            "events": "events.json",
            "series": "series.json",
        },
        "proposals_path": str(proposals_path.relative_to(root)),
        "driver_fixtures": str(fixtures_root),
        "scanner_fixtures": str(fixtures_root),
        "series": {"id": f"{series}_SERIES", "ticker": series, "name": f"{series}"},
    }
    (series_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def test_tools_replay_generates_parquet_and_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    fixtures_root = Path(__file__).resolve().parents[1] / "tests" / "data_fixtures"
    trading_day = date(2025, 11, 10)
    close_generated = _et(2025, 11, 10, 15, 56)
    hourly_generated = _et(2025, 11, 10, 11, 56)
    _write_series_run(
        root=tmp_path,
        trading_day=trading_day,
        timestamp_label="155600",
        series="INX",
        market_id="MKT_INX_H1600_RANGE",
        market_ticker="KXINX-TEST-H1600",
        hour_et=16,
        generated_at_et=close_generated,
        fixtures_root=fixtures_root,
    )
    _write_series_run(
        root=tmp_path,
        trading_day=trading_day,
        timestamp_label="115600",
        series="INXU",
        market_id="MKT_INXU_H1200_A",
        market_ticker="KXINXU-TEST-H1200",
        hour_et=12,
        generated_at_et=hourly_generated,
        fixtures_root=fixtures_root,
    )
    # Add an out-of-window run (same hour but generated too early) that should be ignored
    early_generated = _et(2025, 11, 10, 11, 40)
    _write_series_run(
        root=tmp_path,
        trading_day=trading_day,
        timestamp_label="114000",
        series="INXU",
        market_id="MKT_INXU_H1200_B",
        market_ticker="KXINXU-TEST-H1200-B",
        hour_et=12,
        generated_at_et=early_generated,
        fixtures_root=fixtures_root,
    )

    out_dir = tmp_path / "reports" / "_artifacts"
    args = [
        "--date",
        trading_day.isoformat(),
        "--families",
        "SPX",
        "--hours",
        "12,16",
        "--epsilon",
        "0.2",
        "--out",
        str(out_dir),
    ]
    tools_replay.main(args)

    replay_path = out_dir / "replay_ev.parquet"
    assert replay_path.exists(), "expected aggregate replay parquet"
    frame = pl.read_parquet(replay_path)
    assert set(["window_type", "family", "window_label"]).issubset(frame.columns)
    assert frame["window_label"].n_unique() == 2

    summary_path = out_dir / "replay" / f"replay_summary_{trading_day.isoformat()}.json"
    assert summary_path.exists(), "expected replay summary JSON"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["runs"] == 2
    assert {window["window_type"] for window in summary["windows"]} == {"hourly", "close"}
    assert "hourly" in summary["window_type_max"]
    assert "close" in summary["window_type_max"]

    plot_path = out_dir / "replay" / f"replay_delta_{trading_day.isoformat()}.png"
    assert plot_path.exists(), "expected replay plot artifact"
