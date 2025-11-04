from __future__ import annotations

from decimal import Decimal

import pytest

from kalshi_alpha.core.fees import DEFAULT_FEE_SCHEDULE


@pytest.mark.parametrize(
    "series,price,expected",
    [
        ("INXU", 0.45, Decimal("0.01")),
        ("NASDAQ100", 0.35, Decimal("0.01")),
    ],
)
def test_index_maker_fee_matches_formula(series: str, price: float, expected: Decimal) -> None:
    fee = DEFAULT_FEE_SCHEDULE.maker_fee(contracts=1, price=price, series=series)
    assert fee == expected


@pytest.mark.parametrize(
    "contracts,price,expected",
    [
        (100, 0.50, Decimal("0.88")),
        (100, 0.30, Decimal("0.74")),
        (250, 0.55, Decimal("2.17")),
    ],
)
def test_index_maker_fee_golden_rows(contracts: int, price: float, expected: Decimal) -> None:
    fee = DEFAULT_FEE_SCHEDULE.maker_fee(contracts=contracts, price=price, series="INX")
    assert fee == expected


def test_index_fee_monotonicity() -> None:
    previous = Decimal("0.00")
    for p in [x / 100 for x in range(1, 99)]:
        fee = DEFAULT_FEE_SCHEDULE.maker_fee(contracts=1, price=p, series="INX")
        assert fee >= previous
        previous = fee
