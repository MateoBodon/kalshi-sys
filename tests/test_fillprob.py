from __future__ import annotations

import json
from pathlib import Path

import pytest

from kalshi_alpha.core.execution import fillprob


def test_adjust_alpha_uses_curve(tmp_path: Path, monkeypatch) -> None:
    payload = {
        "series": {
            "INXU": {
                "default_probability": 0.7,
                "late_probability": 0.5,
                "late_threshold_seconds": 120,
            }
        }
    }
    curve_path = tmp_path / "curve.json"
    curve_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(fillprob, "FILL_CURVE_PATH", curve_path)
    fillprob._load_payload.cache_clear()
    adjusted = fillprob.adjust_alpha("INXU", 0.9)
    assert adjusted == pytest.approx(0.7)
    adjusted_late = fillprob.adjust_alpha("INXU", 0.9, seconds_to_event=60)
    assert adjusted_late == pytest.approx(0.5)
