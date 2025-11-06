from decimal import Decimal

import pytest

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
