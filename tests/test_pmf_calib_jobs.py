from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import polars as pl
import pytest

from jobs import calib_eod, calib_hourly
from kalshi_alpha.drivers.polygon_index.client import MinuteBar, PolygonIndicesClient

_FIXTURE_ROOT = Path("tests/data_fixtures/index")
_SPX_NOON_FIXTURE = _FIXTURE_ROOT / "I_SPX_2024-10-21_noon.parquet"
_SPX_CLOSE_FIXTURE = _FIXTURE_ROOT / "I_SPX_2024-10-21_close.parquet"
_SPX_EVENT_CLOSE_FIXTURE = _FIXTURE_ROOT / "I_SPX_2024-09-11_close.parquet"


def _load_bars(path: Path) -> list[MinuteBar]:
    frame = pl.read_parquet(path)
    bars: list[MinuteBar] = []
    for row in frame.iter_rows(named=True):
        bars.append(
            MinuteBar(
                timestamp=row["timestamp"],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                vwap=row.get("vwap"),
                trades=int(row["trades"]) if row.get("trades") is not None else None,
            )
        )
    return bars


def _patch_fetch(monkeypatch: pytest.MonkeyPatch, mapping: dict[str, list[MinuteBar]]) -> None:
    def _fake_fetch(self: PolygonIndicesClient, symbol: str, *_args, **_kwargs) -> list[MinuteBar]:
        return mapping.get(symbol.upper(), [])

    monkeypatch.setattr(PolygonIndicesClient, "fetch_minute_bars", _fake_fetch)


def test_calib_hourly_generates_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bars = _load_bars(_SPX_NOON_FIXTURE)
    _patch_fetch(monkeypatch, {"I:SPX": bars})
    output = tmp_path / "pmf"
    reports = tmp_path / "reports"
    args = [
        "--series",
        "INXU",
        "--target-hours",
        "12",
        "--start",
        "2024-10-21",
        "--end",
        "2024-10-21",
        "--output",
        str(output),
        "--reports-dir",
        str(reports),
        "--skip-plots",
        "--skip-snapshots",
    ]
    calib_hourly.main(args)
    payload_path = output / "INXU" / "hourly" / "1200.json"
    assert payload_path.exists()
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    assert payload["target_type"] == "hourly"
    assert payload["series"] == "INXU"
    assert payload["minutes_to_target"], "minutes_to_target should not be empty"


def test_calib_eod_generates_payload_with_optional_bump(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bars = _load_bars(_SPX_CLOSE_FIXTURE) + _load_bars(_SPX_EVENT_CLOSE_FIXTURE)
    _patch_fetch(monkeypatch, {"I:SPX": bars})
    output = tmp_path / "pmf"
    reports = tmp_path / "reports"
    args = [
        "--series",
        "INX",
        "--start",
        "2024-09-01",
        "--end",
        "2024-10-21",
        "--output",
        str(output),
        "--reports-dir",
        str(reports),
        "--skip-plots",
        "--skip-snapshots",
    ]
    calib_eod.main(args)
    payload_path = output / "INX" / "close" / "close.json"
    assert payload_path.exists()
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    assert payload["target_type"] == "close"
    assert payload["series"] == "INX"
    if payload.get("eod_bump"):
        assert payload["eod_bump"]["minutes_threshold"] >= 0
        assert payload["eod_bump"]["variance"] >= 0.0
