from __future__ import annotations

import json
from pathlib import Path

from kalshi_alpha.exec.collectors import kalshi_tob


def test_kalshi_tob_recorder_offline(tmp_path: Path) -> None:
    output_dir = tmp_path / "tob"
    kalshi_tob.main(
        [
            "--series",
            "INXU",
            "--fixtures-root",
            "tests/data_fixtures",
            "--offline",
            "--iterations",
            "1",
            "--output",
            str(output_dir),
        ]
    )
    files = list(output_dir.glob("*.jsonl"))
    assert files, "expected a snapshot file"
    payload = files[0].read_text(encoding="utf-8").strip().splitlines()
    assert payload, "snapshot file should not be empty"
    record = json.loads(payload[0])
    assert record["series"] == "INXU"
