from __future__ import annotations

from pathlib import Path

import polars as pl

from kalshi_alpha.exec.scanners import scan_index_close, scan_index_hourly


def _install_calibration(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(src.read_bytes())


def test_scan_index_hourly_cli(
    tmp_path: Path,
    fixtures_root: Path,
    isolated_data_roots: tuple[Path, Path],
) -> None:
    _, proc_root = isolated_data_roots
    _install_calibration(
        Path("tests/fixtures/index/spx/hourly/params.json"),
        proc_root / "calib" / "index" / "spx" / "hourly" / "params.json",
    )
    output_root = tmp_path / "reports"
    scan_index_hourly.main(
        [
            "--offline",
            "--fixtures-root",
            str(fixtures_root),
            "--output-root",
            str(output_root),
            "--series",
            "INXU",
            "--now",
            "2025-11-03T12:05:00+00:00",
        ]
    )

    csv_files = sorted(output_root.glob("*/INXU/*.csv"))
    assert csv_files, "expected hourly scanner to emit CSV output"
    frame = pl.read_csv(csv_files[-1])
    assert "delta_bps" in frame.columns


def test_scan_index_close_cli(
    tmp_path: Path,
    fixtures_root: Path,
    isolated_data_roots: tuple[Path, Path],
) -> None:
    _, proc_root = isolated_data_roots
    _install_calibration(
        Path("tests/fixtures/index/ndx/close/params.json"),
        proc_root / "calib" / "index" / "ndx" / "close" / "params.json",
    )

    output_root = tmp_path / "reports"
    scan_index_close.main(
        [
            "--offline",
            "--fixtures-root",
            str(fixtures_root),
            "--output-root",
            str(output_root),
            "--series",
            "NASDAQ100",
            "--now",
            "2025-11-03T20:05:00+00:00",
        ]
    )

    csv_files = sorted((output_root / "NASDAQ100").glob("*.csv"))
    assert csv_files, "expected close scanner to emit CSV output"
    frame = pl.read_csv(csv_files[-1])
    assert "delta_bps" in frame.columns
