from __future__ import annotations

import json
from pathlib import Path

from kalshi_alpha.exec.runners import scan_ladders


def test_load_honesty_clamp(tmp_path: Path, monkeypatch) -> None:
    clamp_payload = {
        "window_days": 7,
        "series": {
            "INXU": {"clamp": 0.75},
            "NASDAQ100U": {"clamp": 0.5},
        },
    }
    clamp_path = tmp_path / "honesty_clamp.json"
    clamp_path.write_text(json.dumps(clamp_payload), encoding="utf-8")
    monkeypatch.setattr(scan_ladders, "HONESTY_CLAMP_PATH", clamp_path)
    value = scan_ladders._load_honesty_clamp("inxu")
    assert value == 0.75
    assert scan_ladders._load_honesty_clamp("UNKNOWN") is None
