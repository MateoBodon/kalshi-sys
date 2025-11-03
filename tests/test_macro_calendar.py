from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from kalshi_alpha.drivers.macro_calendar import emit_day_dummies


def _load_column(frame: pl.DataFrame, target: date, column: str) -> bool:
    series = frame.filter(pl.col("date") == target).select(column).to_series()
    assert not series.is_empty(), f"missing row for {target}"
    value = series.item()
    return bool(value)


def test_emit_day_dummies_offline(
    tmp_path: Path,
    offline_fixtures_root: Path,
) -> None:
    output = tmp_path / "macro.parquet"
    path = emit_day_dummies(
        date(2025, 10, 1),
        date(2025, 10, 31),
        output,
        offline=True,
        fixtures_dir=offline_fixtures_root,
    )
    assert path == output
    frame = pl.read_parquet(path)
    assert set(frame.columns) == {"date", "is_fomc", "is_cpi", "is_jobs", "is_claims"}
    assert frame.height == 31

    assert _load_column(frame, date(2025, 10, 10), "is_cpi") is True
    assert _load_column(frame, date(2025, 10, 3), "is_jobs") is True
    assert _load_column(frame, date(2025, 10, 9), "is_claims") is True
    assert _load_column(frame, date(2025, 10, 29), "is_fomc") is True

    # Non-event day stays empty.
    assert _load_column(frame, date(2025, 10, 8), "is_cpi") is False


def test_emit_day_dummies_invalid_range(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        emit_day_dummies(date(2025, 10, 31), date(2025, 10, 1), tmp_path / "x.parquet")
