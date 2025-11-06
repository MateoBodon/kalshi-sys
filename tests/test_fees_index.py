import json
import re
import warnings
import zlib
from decimal import Decimal
from pathlib import Path

import pytest

import kalshi_alpha.core.fees as fees_module
from kalshi_alpha.core.fees import FeeSchedule, fee_index_taker
from kalshi_alpha.core.pricing import Liquidity, OrderSide, expected_value_after_fees


def test_index_taker_fee_matches_pdf() -> None:
    schedule = FeeSchedule()
    fee = schedule.taker_fee(100, 0.50, series="INXU")
    assert fee == Decimal("0.88")


def test_index_maker_expected_value_has_no_fee() -> None:
    schedule = FeeSchedule()
    ev = expected_value_after_fees(
        contracts=100,
        yes_price=0.50,
        event_probability=0.50,
        side=OrderSide.YES,
        liquidity=Liquidity.MAKER,
        schedule=schedule,
        series="INXU",
    )
    assert ev == 0.0


@pytest.mark.parametrize(
    ("contracts", "price", "expected"),
    [
        (100, Decimal("0.50"), Decimal("0.88")),
        (50, Decimal("0.25"), Decimal("0.33")),
    ],
)
def test_fee_index_taker_golden(contracts: int, price: Decimal, expected: Decimal) -> None:
    fee = fee_index_taker(price=price, contracts=contracts)
    assert fee == expected


def test_index_maker_fee_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OVERRIDE_INDEX_MAKER_FEE", "1")
    schedule = FeeSchedule()
    with pytest.raises(RuntimeError):
        schedule.maker_fee(contracts=100, price=0.5, series="INXU")


def test_fee_schedule_pdf_effective_date_matches_reference() -> None:
    fees_module._load_fee_config.cache_clear()
    root = Path(__file__).resolve().parents[1]
    pdf_path = root / "docs" / "kalshi-fee-schedule.pdf"
    json_path = root / "data" / "reference" / "kalshi_fee_schedule.json"

    if not pdf_path.exists():
        warnings.warn(
            "kalshi-fee-schedule.pdf missing; skipped effective_date drift check",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    config = json.loads(json_path.read_text(encoding="utf-8"))
    reference_date = config.get("effective_date")
    if not reference_date:
        pytest.fail("Reference kalshi_fee_schedule.json missing effective_date")

    pdf_date = _extract_effective_date(pdf_path)

    assert pdf_date == reference_date


def _extract_effective_date(pdf_path: Path) -> str:
    def _decode_segment(segment: bytes) -> str:
        try:
            return zlib.decompress(segment).decode("utf-8", errors="ignore")
        except zlib.error:
            return segment.decode("utf-8", errors="ignore")

    content_parts: list[str] = []
    raw = pdf_path.read_bytes()
    for match in re.finditer(rb"stream\r?\n(.*?)endstream", raw, re.S):
        segment = match.group(1).strip()
        if not segment:
            continue
        decoded = _decode_segment(segment)
        if "Effective Date" in decoded:
            content_parts.append(decoded)
    if not content_parts:
        # Fallback: decode raw bytes if stream markers missing.
        content_parts.append(raw.decode("latin-1", errors="ignore"))

    combined = "\n".join(content_parts)
    match = re.search(r"Effective(?:\s+Date)?(?:\s*[-:])?\s*(\d{4}-\d{2}-\d{2})", combined)
    if not match:
        pytest.fail("Unable to locate Effective Date in kalshi-fee-schedule.pdf")
    return match.group(1)
