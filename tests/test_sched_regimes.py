from __future__ import annotations

from datetime import UTC, datetime, date
from pathlib import Path

import polars as pl

from kalshi_alpha.sched import regimes


def test_regime_flags_from_calendar(tmp_path: Path) -> None:
    calendar_path = tmp_path / "calendar.parquet"
    frame = pl.DataFrame(
        {
            "date": [date(2025, 1, 29), date(2025, 2, 12)],
            "is_fomc": [True, False],
            "is_cpi": [False, True],
        }
    )
    calendar_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(calendar_path)
    fomc_flags = regimes.regime_for(datetime(2025, 1, 29, tzinfo=UTC), calendar_path=calendar_path)
    assert fomc_flags.is_fomc
    assert not fomc_flags.is_cpi
    assert fomc_flags.size_multiplier < 1.0
    assert fomc_flags.label == "fomc"
    cpi_flags = regimes.regime_for(datetime(2025, 2, 12, tzinfo=UTC), calendar_path=calendar_path)
    assert cpi_flags.is_cpi
    assert cpi_flags.label == "cpi"
