from decimal import Decimal

import pytest

from kalshi_alpha.exec import fees as exec_fees


@pytest.mark.parametrize(
    ("contracts", "expected"),
    [
        (1, Decimal("0.01")),
        (2, Decimal("0.02")),
        (100, Decimal("0.88")),
        (1000, Decimal("8.75")),
    ],
)
def test_order_fee_rounding_across_sizes(contracts: int, expected: Decimal) -> None:
    fee = exec_fees.order_fee(
        series="INXU",
        price=0.50,
        contracts=contracts,
        liquidity="taker",
    )
    assert fee == expected


def test_fee_breakdown_matches_per_order() -> None:
    details = exec_fees.fee_breakdown(
        series="NASDAQ100",
        price=0.42,
        contracts=4,
        liquidity="taker",
    )
    per_order = details["per_order"]
    per_contract = details["per_contract_effective"]
    assert per_order == exec_fees.order_fee(series="NASDAQ100", price=0.42, contracts=4, liquidity="taker")
    assert per_contract.quantize(Decimal("0.0001")) == (per_order / Decimal(4)).quantize(Decimal("0.0001"))
