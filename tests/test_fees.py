from decimal import ROUND_DOWN, Decimal

import pytest

from kalshi_alpha.core.fees import FeeSchedule, maker_fee, round_up_to_cent, taker_fee


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


def test_series_override_rates() -> None:
    config = {
        "effective_date": "2025-10-01",
        "maker_rate": 0.02,
        "taker_rate": 0.08,
        "series_half_rate": [],
        "half_rate_keywords": [],
        "series_overrides": [
            {"series": "TNEY", "maker_rate": 0.0125, "taker_rate": 0.05},
        ],
    }
    schedule = FeeSchedule(config=config)
    maker_fee_value = schedule.maker_fee(50, 0.45, series="TNEY")
    expected = round_up_to_cent(_raw_fee(Decimal("0.0125"), 50, 0.45))
    assert maker_fee_value == expected


def test_series_override_half_rate() -> None:
    config = {
        "effective_date": "2025-10-01",
        "maker_rate": 0.02,
        "taker_rate": 0.08,
        "series_half_rate": [],
        "half_rate_keywords": [],
        "series_overrides": [
            {"series": "CPI", "half_rate": True},
        ],
    }
    schedule = FeeSchedule(config=config)
    baseline = schedule.taker_fee(100, 0.4, series="SP500")
    override = schedule.taker_fee(100, 0.4, series="CPI")
    expected_baseline = round_up_to_cent(_raw_fee(schedule.taker_rate, 100, 0.4))
    expected_override = round_up_to_cent(_raw_fee(schedule.taker_rate / 2, 100, 0.4))
    assert baseline == expected_baseline
    assert override == expected_override


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
