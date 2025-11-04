from __future__ import annotations

from pathlib import Path

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
