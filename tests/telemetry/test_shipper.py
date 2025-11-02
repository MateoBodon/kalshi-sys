from __future__ import annotations

import gzip
from datetime import UTC, datetime
from pathlib import Path

from kalshi_alpha.exec.telemetry.shipper import main


def test_telemetry_shipper_compresses(tmp_path: Path) -> None:
    day = datetime(2025, 11, 1, tzinfo=UTC).date()
    source = tmp_path / "data" / "raw" / "kalshi" / f"{day.year:04d}" / f"{day.month:02d}" / f"{day.day:02d}"
    source.mkdir(parents=True, exist_ok=True)
    exec_path = source / "exec.jsonl"
    exec_path.write_text("{\"timestamp\": \"...\"}\n", encoding="utf-8")

    dest = tmp_path / "reports" / "_artifacts" / "telemetry"
    exit_code = main(
        [
            "--source",
            str(tmp_path / "data" / "raw" / "kalshi"),
            "--dest",
            str(dest),
            "--day",
            day.isoformat(),
            "--compress",
        ]
    )
    assert exit_code == 0
    archive = dest / f"{day.isoformat()}.exec.jsonl.gz"
    assert archive.exists()
    with gzip.open(archive, "rt", encoding="utf-8") as handle:
        assert handle.read().strip() == "{\"timestamp\": \"...\"}"
