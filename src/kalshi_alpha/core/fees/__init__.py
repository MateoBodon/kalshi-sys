"""Kalshi fee schedule utilities loaded from the canonical JSON configuration."""

from __future__ import annotations

import json
from collections.abc import Iterable
from decimal import ROUND_UP, Decimal, getcontext
from functools import lru_cache
from pathlib import Path
from typing import Any

getcontext().prec = 16

CENT = Decimal("0.01")
ONE = Decimal("1")

ROOT = Path(__file__).resolve().parents[4]
FEE_CONFIG_PATH = ROOT / "data" / "reference" / "kalshi_fee_schedule.json"
FEE_OVERRIDE_PATH = ROOT / "data" / "proc" / "state" / "fees.json"


@lru_cache(maxsize=4)
def _load_fee_config(config_path: Path | None = None) -> dict[str, Any]:
    if config_path is not None:
        path = Path(config_path)
    else:
        path = FEE_OVERRIDE_PATH if FEE_OVERRIDE_PATH.exists() else FEE_CONFIG_PATH
    if not path.exists():  # pragma: no cover - configuration check
        raise FileNotFoundError(f"Fee configuration not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _to_decimal(value: float | int | Decimal) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _contains_keyword(name: str | None, keywords: Iterable[str]) -> bool:
    if not name:
        return False
    upper = name.upper()
    return any(keyword in upper for keyword in keywords)


def round_up_to_cent(amount: float | Decimal) -> Decimal:
    """Round *amount* up to the next cent, matching Kalshi settlement semantics."""

    dec_amount = _to_decimal(amount)
    if dec_amount <= 0:
        return Decimal("0.00")
    return dec_amount.quantize(CENT, rounding=ROUND_UP)


class FeeSchedule:
    """Encapsulates Kalshi maker/taker fee logic with JSON-backed configuration."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        config_path: Path | None = None,
    ) -> None:
        data = config or _load_fee_config(config_path)
        self.effective_date = data.get("effective_date")
        self.maker_rate = Decimal(str(data["maker_rate"]))
        self.taker_rate = Decimal(str(data["taker_rate"]))

        self.series_half_rate = tuple(str(series).upper() for series in data.get("series_half_rate", []))
        self.half_rate_keywords = tuple(str(keyword).upper() for keyword in data.get("half_rate_keywords", []))

        overrides: dict[str, dict[str, Any]] = {}
        for entry in data.get("series_overrides", []):
            series = str(entry.get("series", "")).upper()
            if not series:
                continue
            overrides[series] = {
                "maker_rate": Decimal(str(entry.get("maker_rate", data["maker_rate"]))),
                "taker_rate": Decimal(str(entry.get("taker_rate", data["taker_rate"]))),
                "half_rate": bool(entry.get("half_rate", False)),
            }
        self.series_overrides = overrides
        self.fee_brackets = list(data.get("fee_brackets", []))

    # Public API -----------------------------------------------------------------

    def taker_fee(
        self,
        contracts: float | int,
        price: float,
        *,
        series: str | None = None,
        market_name: str | None = None,
        half_rate: bool | None = None,
    ) -> Decimal:
        return self._fee(
            contracts=contracts,
            price=price,
            base_rate=self.taker_rate,
            series=series,
            market_name=market_name,
            half_rate=half_rate,
            rate_key="taker_rate",
        )

    def maker_fee(
        self,
        contracts: float | int,
        price: float,
        *,
        series: str | None = None,
        market_name: str | None = None,
        half_rate: bool | None = None,
    ) -> Decimal:
        return self._fee(
            contracts=contracts,
            price=price,
            base_rate=self.maker_rate,
            series=series,
            market_name=market_name,
            half_rate=half_rate,
            rate_key="maker_rate",
        )

    # Internal helpers -----------------------------------------------------------

    def _fee(  # noqa: PLR0913
        self,
        *,
        contracts: float | int,
        price: float,
        base_rate: Decimal,
        series: str | None,
        market_name: str | None,
        half_rate: bool | None,
        rate_key: str,
    ) -> Decimal:
        price_dec = _to_decimal(price)
        if price_dec < 0 or price_dec > ONE:
            raise ValueError("price must be expressed as a probability in [0, 1]")

        contracts_dec = _to_decimal(contracts)
        if contracts_dec <= 0:
            raise ValueError("contracts must be positive")

        rate = self._resolve_rate(
            base_rate,
            series=series,
            market_name=market_name,
            half_rate=half_rate,
            rate_key=rate_key,
        )

        raw_fee = rate * contracts_dec * price_dec * (ONE - price_dec)
        return round_up_to_cent(raw_fee)

    def _resolve_rate(
        self,
        base_rate: Decimal,
        *,
        series: str | None,
        market_name: str | None,
        half_rate: bool | None,
        rate_key: str,
    ) -> Decimal:
        rate = base_rate
        series_key = series.upper() if isinstance(series, str) else None

        if series_key and series_key in self.series_overrides:
            override = self.series_overrides[series_key]
            rate = override.get(rate_key, rate)
            if override.get("half_rate"):
                return rate / Decimal("2")

        apply_half_rate = half_rate
        if apply_half_rate is None:
            if series_key and series_key in self.series_half_rate:
                apply_half_rate = True
            else:
                apply_half_rate = _contains_keyword(market_name, self.half_rate_keywords)

        if apply_half_rate:
            rate = rate / Decimal("2")
        return rate


DEFAULT_FEE_SCHEDULE = FeeSchedule()


def taker_fee(  # noqa: PLR0913
    contracts: float | int,
    price: float,
    *,
    series: str | None = None,
    market_name: str | None = None,
    half_rate: bool | None = None,
    schedule: FeeSchedule = DEFAULT_FEE_SCHEDULE,
) -> Decimal:
    """Convenience wrapper using the default fee schedule."""

    return schedule.taker_fee(
        contracts=contracts,
        price=price,
        series=series,
        market_name=market_name,
        half_rate=half_rate,
    )


def maker_fee(  # noqa: PLR0913
    contracts: float | int,
    price: float,
    *,
    series: str | None = None,
    market_name: str | None = None,
    half_rate: bool | None = None,
    schedule: FeeSchedule = DEFAULT_FEE_SCHEDULE,
) -> Decimal:
    """Convenience wrapper using the default fee schedule."""

    return schedule.maker_fee(
        contracts=contracts,
        price=price,
        series=series,
        market_name=market_name,
        half_rate=half_rate,
    )


__all__ = [
    "FeeSchedule",
    "DEFAULT_FEE_SCHEDULE",
    "maker_fee",
    "taker_fee",
    "round_up_to_cent",
]
