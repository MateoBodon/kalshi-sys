from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from kalshi_alpha.drivers import kalshi_index_history as kih


FIX_ROOT = Path("tests/data_fixtures/kalshi_index_quotes")
ET = ZoneInfo("America/New_York")


def test_loads_and_normalizes_quotes():
    trading_day = date(2025, 11, 3)
    frame = kih.load_quotes_for_day("INX", trading_day, root=FIX_ROOT)
    assert frame.height == 3
    assert set(frame["strike"].to_list()) == {4490.0, 4500.0, 4510.0}
    assert "mid" in frame.columns
    assert frame.select(pl.col("mid").mean()).item() > 0.3
    assert frame.schema["timestamp_et"].time_zone == "America/New_York"


def test_latest_snapshot_filters_by_horizon_and_time():
    trading_day = date(2025, 11, 4)
    as_of = datetime(2025, 11, 4, 15, 59, tzinfo=ET)
    snapshot = kih.latest_snapshot("INX", trading_day, "close", as_of=as_of, root=FIX_ROOT)
    assert snapshot is not None
    assert snapshot.horizon == "close"
    assert snapshot.as_of <= as_of
    assert snapshot.strikes == [4495.0, 4505.0, 4515.0]
    assert snapshot.quotes.select(pl.max("spread_cents")).item() > 0.0


def test_noon_snapshot_uses_intraday_file():
    trading_day = date(2025, 11, 4)
    as_of = datetime(2025, 11, 4, 11, 59, tzinfo=ET)
    snapshot = kih.latest_snapshot("INXU", trading_day, "noon", as_of=as_of, root=FIX_ROOT)
    assert snapshot is not None
    assert snapshot.horizon == "noon"
    assert snapshot.strikes == [4495.0, 4505.0, 4515.0]
    mids = snapshot.quotes["mid"].to_list()
    assert mids == sorted(mids, reverse=True)  # monotone with strike for these fixes
