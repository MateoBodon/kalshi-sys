"""Minimal Polygon-only backtest harness for index ladders with Kalshi quotes."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo

import polars as pl

from kalshi_alpha.core.pricing import Liquidity, OrderSide, expected_value_after_fees
from kalshi_alpha.exec.scanners.utils import pmf_to_survival
from kalshi_alpha.drivers import kalshi_index_history
from kalshi_alpha.drivers.kalshi_index_history import QuoteSnapshot
from kalshi_alpha.strategies.index.fill_model import estimate_maker_fill_prob
from kalshi_alpha.strategies.index.model_polygon import load_params, params_path, predict_pmf
from kalshi_alpha.drivers.polygon_index.symbols import resolve_series

ET = ZoneInfo("America/New_York")

DEFAULT_STRIKE_STEP = {
    "I:SPX": 25.0,
    "I:NDX": 50.0,
}


@dataclass(frozen=True)
class BacktestConfig:
    series: Sequence[str]
    start_date: date | None = None
    end_date: date | None = None
    ev_threshold: float = 0.02
    max_bins: int = 2
    maker_edge: float = 0.02
    contracts: int = 1
    params_root: Path | None = None
    panel_path: Path | None = None
    horizons: Sequence[str] | None = None
    use_kalshi_quotes: bool = False
    quotes_root: Path | None = None
    use_fill_model: bool = True


@dataclass(frozen=True)
class TradeResult:
    series: str
    symbol: str
    trading_day: date
    horizon: str
    timestamp: datetime
    strike: float
    yes_price: float
    model_prob: float
    market_prob: float
    fill_prob: float
    maker_ev: float
    ev_if_filled: float
    pnl: float
    pnl_if_filled: float
    settlement_price: float
    minutes_to_target: float


def _strike_grid(price: float, symbol: str, half_width: int = 6) -> list[float]:
    step = DEFAULT_STRIKE_STEP.get(symbol, 25.0)
    center = round(price / step) * step
    strikes = [center + step * offset for offset in range(-half_width, half_width + 1)]
    return sorted(set(strikes))


def _target_price(frame: pl.DataFrame, target_dt: datetime) -> float | None:
    before = frame.filter(pl.col("timestamp_et") <= target_dt).sort("timestamp_et")
    if not before.is_empty():
        return float(before.tail(1).item(0, "price"))
    after = frame.filter(pl.col("timestamp_et") >= target_dt).sort("timestamp_et")
    if not after.is_empty():
        return float(after.head(1).item(0, "price"))
    return None


def _now_snapshot(frame: pl.DataFrame, target_dt: datetime) -> tuple[datetime, float] | None:
    before = frame.filter(pl.col("timestamp_et") < target_dt).sort("timestamp_et")
    if before.is_empty():
        return None
    latest = before.tail(1)
    return latest.item(0, "timestamp_et"), float(latest.item(0, "price"))


def _minutes_to_target(now_ts: datetime, target_dt: datetime) -> float:
    return (target_dt - now_ts).total_seconds() / 60.0


def _simulate_day(
    frame_day: pl.DataFrame,
    *,
    series: str,
    symbol: str,
    horizon: str,
    params: dict[str, object],
    config: BacktestConfig,
) -> list[TradeResult]:
    target_clock = time(12, 0) if horizon == "noon" else time(16, 0)
    target_dt = datetime.combine(frame_day.item(0, "trading_day"), target_clock, tzinfo=ET)
    snapshot = _now_snapshot(frame_day, target_dt)
    if snapshot is None:
        return []
    now_ts, price_now = snapshot
    quotes: QuoteSnapshot | None = None
    if config.use_kalshi_quotes:
        try:
            quotes = kalshi_index_history.latest_snapshot(
                series=series,
                trading_day=frame_day.item(0, "trading_day"),
                horizon=horizon,
                as_of=now_ts,
                root=config.quotes_root,
            )
        except FileNotFoundError:
            quotes = None

    # Use the quote timestamp when present so time-to-target aligns with Kalshi data.
    reference_now = quotes.as_of if quotes is not None else now_ts
    minutes_to_target = _minutes_to_target(reference_now, target_dt)
    if minutes_to_target <= 0:
        return []

    target_price = _target_price(frame_day, target_dt)
    if target_price is None:
        return []

    strikes = quotes.strikes if quotes is not None else _strike_grid(price_now, symbol)
    quote_map = {float(row["strike"]): row for row in quotes.quotes.to_dicts()} if quotes else {}
    pmf = predict_pmf(
        {"symbol": symbol, "price": price_now, "minutes_to_target": minutes_to_target},
        params,
        strikes,
    )
    survival = pmf_to_survival(pmf, strikes)
    trades: list[TradeResult] = []
    candidates = []
    for strike, event_prob in zip(strikes, survival, strict=True):
        quoted_yes = max(min(event_prob - config.maker_edge, 0.99), 0.01)
        quote_row = quote_map.get(float(strike))
        market_mid = float(quote_row["mid"]) if quote_row else max(min(event_prob, 0.99), 0.01)
        spread_cents = float(quote_row["spread_cents"]) if quote_row else 0.0
        distance_to_mid_cents = abs(quoted_yes - market_mid) * 100.0
        if quote_row and config.use_fill_model:
            fill_prob = estimate_maker_fill_prob(distance_to_mid_cents, minutes_to_target, spread_cents)
        else:
            fill_prob = 1.0

        ev_if_filled = expected_value_after_fees(
            contracts=config.contracts,
            yes_price=quoted_yes,
            event_probability=event_prob,
            side=OrderSide.YES,
            liquidity=Liquidity.MAKER,
            series=series,
        )
        maker_ev = fill_prob * ev_if_filled
        event_occurs = target_price >= strike
        pnl_if_filled = (1.0 - quoted_yes) if event_occurs else -quoted_yes
        expected_pnl = fill_prob * pnl_if_filled
        candidates.append(
            (
                maker_ev,
                strike,
                event_prob,
                quoted_yes,
                market_mid,
                fill_prob,
                ev_if_filled,
                expected_pnl,
                pnl_if_filled,
            )
        )

    candidates.sort(key=lambda row: row[0], reverse=True)
    selected = [entry for entry in candidates if entry[0] >= config.ev_threshold][: config.max_bins]
    for (
        maker_ev,
        strike,
        event_prob,
        quoted_yes,
        market_mid,
        fill_prob,
        ev_if_filled,
        expected_pnl,
        pnl_if_filled,
    ) in selected:
        trades.append(
            TradeResult(
                series=series,
                symbol=symbol,
                trading_day=frame_day.item(0, "trading_day"),
                horizon=horizon,
                timestamp=reference_now,
                strike=float(strike),
                yes_price=float(quoted_yes),
                model_prob=float(event_prob),
                market_prob=float(market_mid),
                fill_prob=float(fill_prob),
                maker_ev=float(maker_ev),
                ev_if_filled=float(ev_if_filled),
                pnl=float(expected_pnl),
                pnl_if_filled=float(pnl_if_filled),
                settlement_price=float(target_price),
                minutes_to_target=float(minutes_to_target),
            )
        )
    return trades


def _horizons_for_series(series: str, override: Sequence[str] | None) -> list[str]:
    if override:
        return [h.lower() for h in override]
    if series.upper().endswith("U"):
        return ["noon"]
    return ["close"]


def run_backtest(panel: pl.DataFrame, config: BacktestConfig) -> list[TradeResult]:
    if panel.is_empty():
        return []

    trades: list[TradeResult] = []
    for series in config.series:
        meta = resolve_series(series)
        symbol = meta.polygon_ticker.upper()
        frame_series = panel.filter(pl.col("symbol") == symbol)
        if config.start_date:
            frame_series = frame_series.filter(pl.col("trading_day") >= config.start_date)
        if config.end_date:
            frame_series = frame_series.filter(pl.col("trading_day") <= config.end_date)
        if frame_series.is_empty():
            continue
        for horizon in _horizons_for_series(series, config.horizons):
            params_file = params_path(series, horizon, root=config.params_root)
            series_params = load_params(params_file)
            for day in frame_series.get_column("trading_day").unique():
                frame_day = frame_series.filter(pl.col("trading_day") == day).sort("timestamp_et")
                trades.extend(
                    _simulate_day(
                        frame_day,
                        series=series.upper(),
                        symbol=symbol,
                        horizon=horizon,
                        params=series_params,
                        config=config,
                    )
                )
    return trades


def summarize(trades: Iterable[TradeResult]) -> dict[str, float]:
    trades_list = list(trades)
    total_pnl = sum(t.pnl for t in trades_list)
    total_ev = sum(t.maker_ev for t in trades_list)
    count = len(trades_list)
    return {
        "trades": count,
        "total_pnl": total_pnl,
        "total_ev": total_ev,
        "avg_ev": (total_ev / count) if count else 0.0,
    }


def write_trades_csv(trades: Iterable[TradeResult], path: Path) -> None:
    rows = list(trades)
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "series",
                "symbol",
                "trading_day",
                "horizon",
                "timestamp_et",
                "minutes_to_target",
                "strike",
                "yes_price",
                "model_prob",
                "market_prob",
                "fill_prob",
                "maker_ev",
                "ev_if_filled",
                "pnl",
                "pnl_if_filled",
                "settlement_price",
            ]
        )
        for trade in rows:
            writer.writerow(
                [
                    trade.series,
                    trade.symbol,
                    trade.trading_day,
                    trade.horizon,
                    trade.timestamp.astimezone(UTC).isoformat(),
                    f"{trade.minutes_to_target:.2f}",
                    f"{trade.strike:.2f}",
                    f"{trade.yes_price:.4f}",
                    f"{trade.model_prob:.6f}",
                    f"{trade.market_prob:.6f}",
                    f"{trade.fill_prob:.4f}",
                    f"{trade.maker_ev:.6f}",
                    f"{trade.ev_if_filled:.6f}",
                    f"{trade.pnl:.6f}",
                    f"{trade.pnl_if_filled:.6f}",
                    f"{trade.settlement_price:.2f}",
                ]
            )


def write_report(summary: dict[str, float], path: Path) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Polygon Index Backtest\n\n")
        handle.write(f"- Trades: {int(summary.get('trades', 0))}\n")
        handle.write(f"- Total EV (USD): {summary.get('total_ev', 0.0):.4f}\n")
        handle.write(f"- Total PnL (USD): {summary.get('total_pnl', 0.0):.4f}\n")
        handle.write(f"- Avg EV / trade (USD): {summary.get('avg_ev', 0.0):.6f}\n")


__all__ = [
    "BacktestConfig",
    "TradeResult",
    "run_backtest",
    "summarize",
    "write_trades_csv",
    "write_report",
]
