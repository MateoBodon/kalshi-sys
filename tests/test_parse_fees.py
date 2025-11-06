from __future__ import annotations

import json
from pathlib import Path

from kalshi_alpha.core import fees as fees_module
from kalshi_alpha.dev.parse_fees import main as parse_main
from kalshi_alpha.dev.parse_fees import parse_fee_schedule


def test_parse_fee_schedule_extracts_brackets(tmp_path: Path) -> None:
    pdf_path = tmp_path / "fee.pdf"
    pdf_content = """%PDF-1.4
Effective Date: 2026-01-15
Maker Fee Rate 0.018
Taker Fee Rate 0.065
Half Rate Series: SP500, NASDAQ
Half Rate Keywords: SP500, NASDAQ
$0 - $100,000 -> 0.05
$100,000 - $500,000 -> 0.045
$500,000 - + -> 0.040
%%EOF
"""
    pdf_path.write_bytes(pdf_content.encode("latin-1"))
    base_config = {
        "effective_date": "2025-10-01",
        "maker_rate": 0.0175,
        "taker_rate": 0.07,
        "series_half_rate": [],
        "half_rate_keywords": [],
        "series_overrides": [],
        "maker_series": ["CPI"],
    }
    result = parse_fee_schedule(pdf_path, base_config=base_config)
    assert result["effective_date"] == "2026-01-15"
    assert result["maker_rate"] == 0.018
    assert result["taker_rate"] == 0.065
    assert result["series_half_rate"] == ["SP500", "NASDAQ"]
    assert result["half_rate_keywords"] == ["SP500", "NASDAQ"]
    assert result["maker_series"] == ["CPI"]
    assert result["fee_brackets"]
    assert result["fee_brackets"][0] == {"lower": 0, "upper": 100000, "rate": 0.05}
    assert result["fee_brackets"][2] == {"lower": 500000, "upper": None, "rate": 0.04}


def test_parse_main_updates_reference(tmp_path: Path) -> None:
    pdf_path = tmp_path / "kalshi-fee-schedule.pdf"
    pdf_content = """%PDF-1.4
Effective Date: 2025-11-01
Maker Fee Rate 0.0175
Taker Fee Rate 0.070
Half Rate Series: INX, INXU, NASDAQ100, NASDAQ100U
Half Rate Keywords: INX, NASDAQ100
$0 - $100,000 -> 7.00%
%%EOF
"""
    pdf_path.write_bytes(pdf_content.encode("latin-1"))

    reference_path = tmp_path / "data" / "reference" / "kalshi_fee_schedule.json"
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    base_config = {
        "effective_date": "2025-10-01",
        "maker_rate": 0.0175,
        "taker_rate": 0.07,
        "series_half_rate": [],
        "half_rate_keywords": [],
        "series_overrides": [],
        "maker_series": ["CPI"],
    }
    reference_path.write_text(json.dumps(base_config), encoding="utf-8")

    output_path = tmp_path / "data" / "proc" / "state" / "fees.json"
    fees_module._load_fee_config.cache_clear()
    parse_main(
        [
            "--pdf",
            str(pdf_path),
            "--output",
            str(output_path),
            "--reference",
            str(reference_path),
            "--quiet",
        ]
    )

    generated = json.loads(reference_path.read_text(encoding="utf-8"))
    assert generated["effective_date"] == "2025-11-01"
    assert set(generated["series_half_rate"]) == {"INX", "INXU", "NASDAQ100", "NASDAQ100U"}
    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8") == reference_path.read_text(encoding="utf-8")
    fees_module._load_fee_config.cache_clear()
