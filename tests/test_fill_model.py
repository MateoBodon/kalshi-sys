from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from kalshi_alpha.replay import fill_model


def _write_snapshot(tmp_path: Path, series: str = "INXU") -> Path:
    path = tmp_path / "tob.jsonl"
    captured = datetime.now(tz=UTC).isoformat()
    entries = [
        {
            "captured_at": captured,
            "series": series,
            "market_id": "MKT_INXU_H1200_A",
            "market_ticker": "KXINXU-TEST",
            "best_bid_price": 0.48,
            "best_bid_size": 12,
            "best_ask_price": 0.52,
            "best_ask_size": 15,
            "seconds_to_close": 300,
        },
        {
            "captured_at": captured,
            "series": series,
            "market_id": "MKT_INXU_H1200_A",
            "market_ticker": "KXINXU-TEST",
            "best_bid_price": 0.49,
            "best_bid_size": 6,
            "best_ask_price": 0.51,
            "best_ask_size": 4,
            "seconds_to_close": 90,
        },
    ]
    path.write_text("\n".join(json.dumps(entry) for entry in entries), encoding="utf-8")
    return path


def test_fill_model_builds_curve(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path)
    output = tmp_path / "curve.json"
    fill_model.main(["--snapshots", str(snapshot), "--output", str(output), "--contracts", "10"])
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["series"]["INXU"]["default_probability"]
