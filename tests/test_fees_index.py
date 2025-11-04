from decimal import Decimal

from kalshi_alpha.core.fees import FeeSchedule
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
