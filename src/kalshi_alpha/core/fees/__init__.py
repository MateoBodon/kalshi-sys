"""Kalshi fee schedule utilities (effective October 1, 2025).

This module implements the official Kalshi trading fee formulas with round-up-to-the-next-cent
semantics and includes the reduced-rate path for S&P and Nasdaq index markets.

References:
- Kalshi fee schedule (Oct 1, 2025).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import ROUND_UP, Decimal, getcontext

getcontext().prec = 16

CENT = Decimal("0.01")
ONE = Decimal("1")

HALF_RATE_KEYWORDS = {
    "S&P",
    "SP500",
    "SPX",
    "NASDAQ",
    "NDX",
    "NAS100",
}


def _to_decimal(value: float | int | Decimal) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def round_up_to_cent(amount: float | Decimal) -> Decimal:
    """Round *amount* up to the next cent, matching Kalshi settlement semantics."""
    dec_amount = _to_decimal(amount)
    if dec_amount <= 0:
        return Decimal("0.00")
    return dec_amount.quantize(CENT, rounding=ROUND_UP)


def _should_apply_half_rate(
    market: str | None,
    keywords: Iterable[str] = HALF_RATE_KEYWORDS,
) -> bool:
    if market is None:
        return False
    upper_name = market.upper()
    return any(keyword in upper_name for keyword in keywords)


@dataclass(frozen=True)
class FeeSchedule:
    """Encapsulates Kalshi maker/taker fee logic."""

    taker_rate: Decimal = Decimal("0.07")
    maker_rate: Decimal = Decimal("0.0175")
    half_rate_keywords: tuple[str, ...] = tuple(sorted(HALF_RATE_KEYWORDS))

    def taker_fee(
        self,
        contracts: float | int,
        price: float,
        *,
        market_name: str | None = None,
        half_rate: bool | None = None,
    ) -> Decimal:
        """Compute taker trading fees for a YES or NO order."""
        return self._fee(
            contracts=contracts,
            price=price,
            base_rate=self.taker_rate,
            market_name=market_name,
            half_rate=half_rate,
        )

    def maker_fee(
        self,
        contracts: float | int,
        price: float,
        *,
        market_name: str | None = None,
        half_rate: bool | None = None,
    ) -> Decimal:
        """Compute maker trading fees for a YES or NO order."""
        return self._fee(
            contracts=contracts,
            price=price,
            base_rate=self.maker_rate,
            market_name=market_name,
            half_rate=half_rate,
        )

    def _fee(
        self,
        *,
        contracts: float | int,
        price: float,
        base_rate: Decimal,
        market_name: str | None,
        half_rate: bool | None,
    ) -> Decimal:
        price_dec = _to_decimal(price)
        if price_dec < 0 or price_dec > ONE:
            raise ValueError("price must be expressed as a probability in [0, 1]")

        contracts_dec = _to_decimal(contracts)
        if contracts_dec <= 0:
            raise ValueError("contracts must be positive")

        apply_half_rate = (
            half_rate
            if half_rate is not None
            else _should_apply_half_rate(market_name, self.half_rate_keywords)
        )
        rate = base_rate / Decimal("2") if apply_half_rate else base_rate

        raw_fee = rate * contracts_dec * price_dec * (ONE - price_dec)
        return round_up_to_cent(raw_fee)


DEFAULT_FEE_SCHEDULE = FeeSchedule()


def taker_fee(
    contracts: float | int,
    price: float,
    *,
    market_name: str | None = None,
    half_rate: bool | None = None,
    schedule: FeeSchedule = DEFAULT_FEE_SCHEDULE,
) -> Decimal:
    """Convenience wrapper using the default fee schedule."""
    return schedule.taker_fee(
        contracts=contracts,
        price=price,
        market_name=market_name,
        half_rate=half_rate,
    )


def maker_fee(
    contracts: float | int,
    price: float,
    *,
    market_name: str | None = None,
    half_rate: bool | None = None,
    schedule: FeeSchedule = DEFAULT_FEE_SCHEDULE,
) -> Decimal:
    """Convenience wrapper using the default fee schedule."""
    return schedule.maker_fee(
        contracts=contracts,
        price=price,
        market_name=market_name,
        half_rate=half_rate,
    )
