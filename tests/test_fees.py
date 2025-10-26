from decimal import ROUND_DOWN, Decimal

import pytest

from kalshi_alpha.core.fees import FeeSchedule, maker_fee, taker_fee


def _raw_fee(rate: Decimal, contracts: int, price: float) -> Decimal:
    price_dec = Decimal(str(price))
    contracts_dec = Decimal(str(contracts))
    return rate * contracts_dec * price_dec * (Decimal("1") - price_dec)


def test_taker_fee_examples() -> None:
    assert taker_fee(100, 0.35) == Decimal("1.60")
    assert taker_fee(100, 0.50) == Decimal("1.75")
    assert taker_fee(100, 0.99) == Decimal("0.07")


def test_maker_fee_examples() -> None:
    assert maker_fee(100, 0.35) == Decimal("0.40")


def test_half_rate_path() -> None:
    schedule = FeeSchedule()
    half = schedule.taker_fee(100, 0.35, market_name="S&P 500")
    assert half == Decimal("0.80")


@pytest.mark.parametrize("contracts", [1, 10, 100, 1000])
@pytest.mark.parametrize("price", [round(p / 100, 2) for p in range(1, 100)])
def test_taker_fee_round_up_behavior(contracts: int, price: float) -> None:
    schedule = FeeSchedule()
    fee = schedule.taker_fee(contracts, price)
    raw = _raw_fee(schedule.taker_rate, contracts, price)
    assert fee >= raw.quantize(Decimal("0.0001"))
    assert (fee - raw) <= Decimal("0.01")
    if (raw / Decimal("0.01")) == (raw / Decimal("0.01")).to_integral_value():
        assert fee == raw


@pytest.mark.parametrize("price", [0.4999, 0.5001, 0.7499, 0.2504])
def test_round_up_edge_cases(price: float) -> None:
    contracts = 10
    schedule = FeeSchedule()
    fee = schedule.taker_fee(contracts, price)
    raw = _raw_fee(schedule.taker_rate, contracts, price)
    floored = raw.quantize(Decimal("0.01"), rounding=ROUND_DOWN)
    if raw == floored:
        assert fee == raw
    else:
        assert fee == floored + Decimal("0.01")
