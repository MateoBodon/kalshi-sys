from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from kalshi_alpha.drivers import macro_calendar
from kalshi_alpha.drivers.macro_calendar import cli


def test_macro_calendar_cli_offline(monkeypatch, tmp_path: Path, offline_fixtures_root: Path) -> None:
    monkeypatch.chdir(tmp_path)
    target_root = tmp_path / "data" / "proc" / "macro_calendar"
    target_root.mkdir(parents=True, exist_ok=True)

    default_output = target_root / "latest.parquet"
    monkeypatch.setattr(macro_calendar, "PROC_ROOT", target_root)
    monkeypatch.setattr(macro_calendar, "DEFAULT_OUTPUT", default_output)
    monkeypatch.setattr(cli, "DEFAULT_OUTPUT", default_output)

    result = cli.main(
        [
            "--offline",
            "--fixtures-root",
            str(offline_fixtures_root),
            "--as-of",
            "2025-10-31",
            "--days",
            "60",
            "--quiet",
        ]
    )
    assert result == default_output
    frame = pl.read_parquet(result)
    assert set(frame.columns) == {"date", "is_fomc", "is_cpi", "is_jobs", "is_claims"}
    assert frame.height == 60

    def _flag(day: date, column: str) -> bool:
        series = frame.filter(pl.col("date") == day).select(column).to_series()
        return bool(series.item()) if not series.is_empty() else False

    assert _flag(date(2025, 10, 10), "is_cpi") is True
    assert _flag(date(2025, 10, 31), "is_jobs") is True
    assert _flag(date(2025, 9, 17), "is_fomc") is True
    assert _flag(date(2025, 10, 2), "is_claims") is True
