from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from kalshi_alpha.core.datastore import DuckDBCatalog, ProcessedWriter, SnapshotWriter


def test_snapshot_writer_creates_unique_files(tmp_path: Path) -> None:
    writer = SnapshotWriter(tmp_path / "raw")
    timestamp = datetime(2025, 10, 25, 15, 30, tzinfo=UTC)
    first = writer.write_json("cpi", {"value": 1}, timestamp=timestamp)
    second = writer.write_json("cpi", {"value": 2}, timestamp=timestamp)
    assert first.exists()
    assert second.exists()
    assert first != second
    assert second.name.endswith("_1.json")


def test_processed_writer_registers_with_duckdb(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.duckdb"
    catalog = DuckDBCatalog(catalog_path)
    writer = ProcessedWriter(tmp_path / "proc", catalog=catalog)
    frame = pl.DataFrame({"strike": [0.1, 0.2], "prob": [0.3, 0.7]})
    timestamp = datetime(2025, 10, 25, 12, 0, tzinfo=UTC)
    parquet_path = writer.write_parquet("cpi", frame, timestamp=timestamp)
    assert parquet_path.exists()
    registrations = catalog.list_registered("cpi")
    assert registrations and registrations[0][1] == str(parquet_path)
    catalog.close()
