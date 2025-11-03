#!/usr/bin/env python3
"""Convert the Kalshi fee schedule PDF into JSON consumed by ``kalshi_alpha.core.fees``.

This script is intentionally lightweight: if ``pdfplumber`` is available and the
official PDF exists, the table is parsed directly. Otherwise we fall back to the
hand-maintained defaults that ship with the repository.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import pdfplumber  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pdfplumber = None

ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "docs" / "kalshi-fee-schedule.pdf"
OUTPUT_PATH = ROOT / "data" / "reference" / "kalshi_fee_schedule.json"


def _fallback_config() -> dict[str, Any]:
    return {
        "effective_date": "2025-10-01",
        "maker_rate": 0.0175,
        "taker_rate": 0.07,
        "series_half_rate": ["SP500", "SPX", "NASDAQ", "NDX", "NAS100"],
        "half_rate_keywords": ["S&P", "SP500", "SPX", "NASDAQ", "NDX", "NAS100"],
        "series_overrides": [],
    }


def _parse_pdf() -> dict[str, Any]:  # pragma: no cover - exercised manually
    if pdfplumber is None or not PDF_PATH.exists():
        return _fallback_config()

    with pdfplumber.open(PDF_PATH) as pdf:
        first_page = pdf.pages[0]
        table = first_page.extract_table() or []

    maker_rate = 0.0175
    taker_rate = 0.07
    keywords: list[str] = []

    for row in table:
        if not row:
            continue
        cells = [cell.strip() for cell in row if isinstance(cell, str)]
        line = " ".join(cells).upper()
        if "MAKER" in line and "FEE" in line and "%" in line:
            parts = [segment for segment in line.replace("%", "").split() if segment.replace(".", "", 1).isdigit()]
            if len(parts) >= 2:
                maker_rate = float(parts[0]) / 100
                taker_rate = float(parts[1]) / 100
        if "S&P" in line or "NASDAQ" in line or "HALF" in line:
            keywords.extend(cell for cell in cells if cell)

    keywords = sorted({keyword.upper() for keyword in keywords if keyword})

    return {
        "effective_date": "2025-10-01",
        "maker_rate": maker_rate,
        "taker_rate": taker_rate,
        "series_half_rate": ["SP500", "SPX", "NASDAQ", "NDX", "NAS100"],
        "half_rate_keywords": keywords or ["S&P", "SP500", "SPX", "NASDAQ", "NDX", "NAS100"],
        "series_overrides": [],
    }


def main() -> None:
    config = _parse_pdf()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Wrote fee config to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
