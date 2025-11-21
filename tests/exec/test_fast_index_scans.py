from __future__ import annotations

from pathlib import Path

import polars as pl

from kalshi_alpha.exec.scanners import scan_index_close, scan_index_hourly


def _csv_rows(output_root: Path) -> list[Path]:
    return list(output_root.rglob("*.csv"))


def test_fast_hourly_scan(tmp_path: Path) -> None:
    output = tmp_path / "hourly"
    scan_index_hourly.main(
        [
            "--offline",
            "--fast-fixtures",
            "--fixtures-root",
            "tests/data_fixtures",
            "--output-root",
            str(output),
            "--series",
            "INXU",
            "NASDAQ100U",
            "--target-hour",
            "12",
        ]
    )
    files = _csv_rows(output)
    assert files, "expected at least one hourly CSV output"
    frame = pl.read_csv(files[0])
    assert frame.height > 0


def test_fast_close_scan(tmp_path: Path) -> None:
    output = tmp_path / "close"
    scan_index_close.main(
        [
            "--offline",
            "--fast-fixtures",
            "--fixtures-root",
            "tests/data_fixtures",
            "--output-root",
            str(output),
            "--series",
            "INX",
            "NASDAQ100",
        ]
    )
    files = _csv_rows(output)
    assert files, "expected at least one close CSV output"
    frame = pl.read_csv(files[0])
    assert frame.height > 0
