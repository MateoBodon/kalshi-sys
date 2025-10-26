"""Datastore helpers for raw snapshots, processed tables, and DuckDB cataloging."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import polars as pl

TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S"


def _timestamp_slug(ts: datetime | None = None) -> str:
    moment = ts or datetime.now(tz=UTC)
    return moment.strftime(TIMESTAMP_FORMAT)


def _next_available(path: Path) -> Path:
    if not path.exists():
        return path
    index = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


@dataclass
class SnapshotWriter:
    root: Path

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def write_json(
        self,
        namespace: str,
        payload: dict,
        *,
        timestamp: datetime | None = None,
    ) -> Path:
        target_dir = self.root / namespace
        target_dir.mkdir(parents=True, exist_ok=True)
        slug = _timestamp_slug(timestamp)
        path = _next_available(target_dir / f"{slug}.json")
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path


@dataclass
class ProcessedWriter:
    root: Path
    catalog: DuckDBCatalog | None = None

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def write_parquet(
        self,
        namespace: str,
        data: pl.DataFrame | Sequence[dict],
        *,
        timestamp: datetime | None = None,
    ) -> Path:
        frame = data if isinstance(data, pl.DataFrame) else pl.DataFrame(data)
        target_dir = self.root / namespace
        target_dir.mkdir(parents=True, exist_ok=True)
        slug = _timestamp_slug(timestamp)
        path = _next_available(target_dir / f"{slug}.parquet")
        frame.write_parquet(path)
        if self.catalog:
            self.catalog.register(namespace, path)
        return path


class DuckDBCatalog:
    """Lightweight registry of processed snapshots inside DuckDB."""

    def __init__(self, database_path: Path) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self.database_path = database_path
        self._conn = duckdb.connect(str(database_path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS parquet_snapshots (
                namespace TEXT,
                path TEXT,
                recorded_at TIMESTAMP DEFAULT now()
            )
            """
        )

    def register(self, namespace: str, parquet_path: Path) -> None:
        self._conn.execute(
            "INSERT INTO parquet_snapshots (namespace, path) VALUES (?, ?)",
            [namespace, str(parquet_path)],
        )

    def list_registered(self, namespace: str | None = None) -> list[tuple[str, str]]:
        if namespace is None:
            result = self._conn.execute("SELECT namespace, path FROM parquet_snapshots").fetchall()
        else:
            result = self._conn.execute(
                "SELECT namespace, path FROM parquet_snapshots WHERE namespace = ?", [namespace]
            ).fetchall()
        return [(row[0], row[1]) for row in result]

    def close(self) -> None:
        self._conn.close()
