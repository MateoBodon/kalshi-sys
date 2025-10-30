"""Replay archived Kalshi data to recompute proposal EVs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import polars as pl

from kalshi_alpha.core.fees import DEFAULT_FEE_SCHEDULE
from kalshi_alpha.core.kalshi_api import Market, Orderbook, Series
from kalshi_alpha.core.pricing import LadderBinProbability, LadderRung, pmf_from_quotes
from kalshi_alpha.exec.scanners import cpi as cpi_scanner
from kalshi_alpha.exec.scanners.utils import expected_value_summary, pmf_to_survival
from kalshi_alpha.strategies import claims as claims_strategy
from kalshi_alpha.strategies import teny as teny_strategy
from kalshi_alpha.strategies import weather as weather_strategy


@dataclass
class _ReplayContext:
    manifest_path: Path
    manifest: dict[str, Any]
    base_dir: Path
    artifacts_dir: Path


def replay_manifest(
    manifest_path: Path | str,
    *,
    model_version: str = "v15",
    orderbooks_override: Mapping[str, Orderbook] | None = None,
) -> Path:
    """Recompute expected values for archived proposals and write parquet output."""

    ctx = _load_context(Path(manifest_path))
    proposals = _load_proposals(ctx)
    if not proposals:
        return _write_output([], ctx.artifacts_dir)

    markets = _load_markets(ctx)
    if orderbooks_override is not None:
        orderbooks = dict(orderbooks_override)
        if len(orderbooks) < len(markets):
            disk_books = _load_orderbooks(ctx)
            for market_id, book in disk_books.items():
                orderbooks.setdefault(market_id, book)
    else:
        orderbooks = _load_orderbooks(ctx)
    driver_fixtures = _resolve_driver_fixtures(ctx)
    series_info = ctx.manifest.get("series", {})
    series = Series(
        id=str(series_info.get("id", "")),
        ticker=str(series_info.get("ticker", "")),
        name=str(series_info.get("name", "")),
    )

    records: list[dict[str, Any]] = []
    for proposal in proposals:
        market = markets.get(proposal["market_id"])
        if market is None:
            continue
        rungs = _build_rungs(market)
        if not rungs:
            continue
        market_pmf = pmf_from_quotes(rungs)
        market_survival = pmf_to_survival(market_pmf, [rung.strike for rung in rungs])

        try:
            strategy_pmf = _strategy_pmf_for_series(
                series=series.ticker,
                strikes=[rung.strike for rung in rungs],
                fixtures_dir=driver_fixtures,
                override=series.ticker,
                offline=True,
                model_version=model_version,
            )
            strategy_survival = pmf_to_survival(strategy_pmf, [rung.strike for rung in rungs])
        except Exception:  # pragma: no cover - fallback when drivers unavailable
            strategy_survival = market_survival

        strike = float(proposal["strike"])
        idx = _resolve_rung_index([rung.strike for rung in rungs], strike)
        if idx is None:
            continue
        event_probability = float(strategy_survival[idx])
        yes_price = float(rungs[idx].yes_price)
        contracts = int(proposal.get("contracts", 0))
        if contracts <= 0:
            continue

        summary_total = expected_value_summary(
            contracts=contracts,
            yes_price=yes_price,
            event_probability=event_probability,
            schedule=DEFAULT_FEE_SCHEDULE,
            market_name=market.ticker,
        )
        summary_per = expected_value_summary(
            contracts=1,
            yes_price=yes_price,
            event_probability=event_probability,
            schedule=DEFAULT_FEE_SCHEDULE,
            market_name=market.ticker,
        )

        side = str(proposal.get("side", "YES")).upper()
        maker_key = "maker_yes" if side == "YES" else "maker_no"
        taker_key = "taker_yes" if side == "YES" else "taker_no"

        ob = orderbooks.get(market.id)
        depth = (len(ob.bids) + len(ob.asks)) if ob else 0

        records.append(
            {
                "series": series.ticker,
                "market_id": market.id,
                "market_ticker": market.ticker,
                "strike": strike,
                "side": side,
                "contracts": contracts,
                "strategy_probability": event_probability,
                "market_survival": float(market_survival[idx]),
                "maker_ev_replay": float(summary_total[maker_key]),
                "maker_ev_per_contract_replay": float(summary_per[maker_key]),
                "taker_ev_replay": float(summary_total[taker_key]),
                "taker_ev_per_contract_replay": float(summary_per[taker_key]),
                "orderbook_depth": depth,
                "maker_ev_original": float(proposal.get("maker_ev", 0.0)),
                "manifest": str(ctx.manifest_path),
            }
        )

    return _write_output(records, ctx.artifacts_dir)


def _load_context(manifest_path: Path) -> _ReplayContext:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_dir = manifest_path.parent
    artifacts_dir = Path("reports/_artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return _ReplayContext(manifest_path=manifest_path, manifest=manifest, base_dir=base_dir, artifacts_dir=artifacts_dir)


def _load_markets(ctx: _ReplayContext) -> dict[str, Market]:
    markets_path = ctx.base_dir / ctx.manifest["paths"]["markets"]
    payload = json.loads(markets_path.read_text(encoding="utf-8"))
    result: dict[str, Market] = {}
    for item in payload:
        market = Market.from_payload(item)
        result[market.id] = market
    return result


def _load_orderbooks(ctx: _ReplayContext) -> dict[str, Orderbook]:
    entries = ctx.manifest["paths"].get("orderbooks", [])
    result: dict[str, Orderbook] = {}
    for entry in entries:
        path = ctx.base_dir / entry
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        orderbook = Orderbook.from_payload(payload)
        result[orderbook.market_id] = orderbook
    return result


def _load_proposals(ctx: _ReplayContext) -> list[dict[str, Any]]:
    proposals_path = ctx.manifest.get("proposals_path")
    if not proposals_path:
        return []
    path = Path(proposals_path)
    if not path.exists():
        # try relative to manifest
        candidate = ctx.base_dir / proposals_path
        if candidate.exists():
            path = candidate
        else:
            return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("proposals", []))


def _resolve_driver_fixtures(ctx: _ReplayContext) -> Path:
    raw = ctx.manifest.get("driver_fixtures")
    if raw:
        path = Path(raw)
        if not path.exists():
            candidate = ctx.base_dir / raw
            if candidate.exists():
                path = candidate
        return path
    default = Path("tests/fixtures")
    return default if default.exists() else Path.cwd()


def _strategy_pmf_for_series(
    *,
    series: str,
    strikes: Sequence[float],
    fixtures_dir: Path,
    override: str,
    offline: bool,
    model_version: str = "v15",
) -> list[LadderBinProbability]:
    pick = override.lower()
    ticker = series.upper()
    if pick == "auto":
        pick = ticker.lower()

    version = (model_version or "v15").lower()
    if version not in {"v0", "v15"}:
        version = "v15"

    if pick == "cpi" and ticker == "CPI":
        return cpi_scanner.strategy_pmf(
            strikes,
            fixtures_dir=fixtures_dir,
            offline=offline,
            model_version=version,
        )
    if pick in {"claims", "jobless"} and ticker == "CLAIMS":
        history = _load_history(fixtures_dir, "claims")
        claims_history = [int(item["claims"]) for item in history[-6:]] if history else None
        latest_claims = claims_history[-1] if claims_history else None
        holiday_flag = bool(history[-1].get("holiday")) if history else False
        continuing_seq = None
        if history:
            continuing_seq = [
                int(item["continuing_claims"])
                for item in history[-3:]
                if "continuing_claims" in item
            ] or None
        short_week_flag = bool(history[-1].get("short_week")) if history else holiday_flag
        inputs = claims_strategy.ClaimsInputs(
            history=claims_history,
            holiday_next=holiday_flag,
            freeze_active=claims_strategy.freeze_window(),
            latest_initial_claims=latest_claims,
            four_week_avg=None,
            continuing_claims=continuing_seq,
            short_week=short_week_flag,
        )
        if version == "v15":
            return claims_strategy.pmf_v15(strikes, inputs=inputs)
        return claims_strategy.pmf(strikes, inputs=inputs)
    if pick in {"tney", "rates"} and ticker == "TNEY":
        history = _load_history(fixtures_dir, "teny")
        if history:
            closes = [float(entry["actual_close"]) for entry in history]
            latest = history[-1]
            prior_close = float(latest.get("prior_close", closes[-1]))
            macro_shock = float(latest.get("macro_shock", 0.0))
            trailing = closes[:-1] if len(closes) > 1 else closes
        else:
            prior_close = None
            macro_shock = 0.0
            trailing = None
        inputs = teny_strategy.TenYInputs(
            prior_close=prior_close,
            macro_shock=macro_shock,
            trailing_history=trailing,
        )
        if version == "v15":
            return teny_strategy.pmf_v15(strikes, inputs=inputs)
        return teny_strategy.pmf(strikes, inputs=inputs)
    if pick in {"weather"} and ticker == "WEATHER":
        history = _load_history(fixtures_dir, "weather")
        if history:
            latest = history[-1]
            inputs = weather_strategy.WeatherInputs(
                forecast_high=float(latest.get("forecast_high", 70.0)),
                bias=float(latest.get("bias", 0.0)),
                spread=float(latest.get("spread", 3.0)),
                station=str(latest.get("station", "")),
            )
        else:
            inputs = weather_strategy.WeatherInputs(forecast_high=70.0)
        return weather_strategy.pmf(strikes, inputs=inputs)
    raise ValueError(f"No strategy PMF implemented for series {series}")


def _load_history(fixtures_dir: Path, namespace: str) -> list[dict[str, Any]]:
    path = fixtures_dir / namespace / "history.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    history = payload.get("history")
    if isinstance(history, list):
        return [item for item in history if isinstance(item, dict)]
    return []


def _build_rungs(market: Market) -> list[LadderRung]:
    return [
        LadderRung(strike=float(strike), yes_price=float(price))
        for strike, price in zip(market.ladder_strikes, market.ladder_yes_prices, strict=True)
    ]


def _resolve_rung_index(strikes: list[float], strike: float) -> int | None:
    if not strikes:
        return None
    best_idx = min(range(len(strikes)), key=lambda idx: abs(strikes[idx] - strike))
    if abs(strikes[best_idx] - strike) > 1e-6:
        return None
    return best_idx


def _write_output(records: list[dict[str, Any]], artifacts_dir: Path) -> Path:
    if records:
        frame = pl.DataFrame(records)
    else:
        frame = pl.DataFrame(
            {
                "series": pl.Series(dtype=pl.Utf8, name="series", values=[]),
                "market_id": pl.Series(dtype=pl.Utf8, name="market_id", values=[]),
                "market_ticker": pl.Series(dtype=pl.Utf8, name="market_ticker", values=[]),
                "strike": pl.Series(dtype=pl.Float64, name="strike", values=[]),
                "side": pl.Series(dtype=pl.Utf8, name="side", values=[]),
                "contracts": pl.Series(dtype=pl.Int64, name="contracts", values=[]),
                "strategy_probability": pl.Series(dtype=pl.Float64, name="strategy_probability", values=[]),
                "market_survival": pl.Series(dtype=pl.Float64, name="market_survival", values=[]),
                "maker_ev_replay": pl.Series(dtype=pl.Float64, name="maker_ev_replay", values=[]),
                "maker_ev_per_contract_replay": pl.Series(dtype=pl.Float64, name="maker_ev_per_contract_replay", values=[]),
                "taker_ev_replay": pl.Series(dtype=pl.Float64, name="taker_ev_replay", values=[]),
                "taker_ev_per_contract_replay": pl.Series(dtype=pl.Float64, name="taker_ev_per_contract_replay", values=[]),
                "orderbook_depth": pl.Series(dtype=pl.Int64, name="orderbook_depth", values=[]),
                "maker_ev_original": pl.Series(dtype=pl.Float64, name="maker_ev_original", values=[]),
                "manifest": pl.Series(dtype=pl.Utf8, name="manifest", values=[]),
            }
        )
    output_path = artifacts_dir / "replay_ev.parquet"
    frame.write_parquet(output_path)
    return output_path
