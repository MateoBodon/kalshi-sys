from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from kalshi_alpha.exec.ingest import polygon_index


def test_polygon_ingest_cli_invokes_client(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, datetime, datetime, Path]] = []

    class FakeClient:
        def download_minute_history(self, symbol: str, start: datetime, end: datetime, *, output_root: Path, chunk_limit: int = 50000, adjusted: bool = True):
            calls.append((symbol, start, end, output_root))
            return [output_root / symbol.replace(":", "_").upper() / "2025-11-03.parquet"]

    monkeypatch.setattr(polygon_index, "PolygonIndicesClient", FakeClient)

    output_root = tmp_path / "polygon"
    polygon_index.main(
        [
            "--start",
            "2025-11-01",
            "--end",
            "2025-11-03",
            "--symbols",
            "I:SPX",
            "--output-root",
            str(output_root),
        ]
    )

    assert calls
    symbol, start, end, root = calls[0]
    assert symbol == "I:SPX"
    assert start.tzinfo is UTC and end.tzinfo is UTC
    assert root == output_root
