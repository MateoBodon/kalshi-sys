"""Shared symbol metadata for Polygon-powered index ladders."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class IndexSymbol:
    kalshi_series: str
    polygon_ticker: str
    description: str
    window: str
    fallback_ticker: str | None = None


_SYMBOLS: dict[str, IndexSymbol] = {
    "INX": IndexSymbol(
        kalshi_series="INX",
        polygon_ticker="I:SPX",
        description="S&P 500 index close (Polygon I:SPX)",
        window="daily-close",
        fallback_ticker="SPY",
    ),
    "INXU": IndexSymbol(
        kalshi_series="INXU",
        polygon_ticker="I:SPX",
        description="S&P 500 index noon snapshot (Polygon I:SPX)",
        window="intraday-noon",
        fallback_ticker="SPY",
    ),
    "NASDAQ100": IndexSymbol(
        kalshi_series="NASDAQ100",
        polygon_ticker="I:NDX",
        description="NASDAQ 100 index close (Polygon I:NDX)",
        window="daily-close",
        fallback_ticker="QQQ",
    ),
    "NASDAQ100U": IndexSymbol(
        kalshi_series="NASDAQ100U",
        polygon_ticker="I:NDX",
        description="NASDAQ 100 index noon snapshot (Polygon I:NDX)",
        window="intraday-noon",
        fallback_ticker="QQQ",
    ),
}


def supported_series() -> list[IndexSymbol]:
    return list(_SYMBOLS.values())


def resolve_series(series: str) -> IndexSymbol:
    key = series.upper()
    try:
        return _SYMBOLS[key]
    except KeyError as exc:
        supported = ", ".join(sorted(_SYMBOLS))
        raise KeyError(f"Unsupported index series '{series}'. Supported: {supported}") from exc


def polygon_tickers(series: Iterable[str]) -> list[str]:
    results: list[str] = []
    for name in series:
        try:
            results.append(resolve_series(name).polygon_ticker)
        except KeyError:
            continue
    return results


__all__ = ["IndexSymbol", "supported_series", "resolve_series", "polygon_tickers"]
