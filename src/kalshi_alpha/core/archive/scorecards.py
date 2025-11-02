"""Replay scorecard computation from archived manifests and proposals."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from kalshi_alpha.core.archive.replay import (
    _build_rungs,
    _load_context,
    _load_markets,
    _load_orderbooks,
    _load_proposals,
    _resolve_driver_fixtures,
    _strategy_pmf_for_series,
)
from kalshi_alpha.core.kalshi_api import Orderbook
from kalshi_alpha.core.pricing import implied_cdf_kinks, pmf_from_quotes, prob_sum_gap
from kalshi_alpha.exec.scanners.utils import pmf_to_survival


@dataclass(frozen=True)
class ReplayScorecard:
    """Container for replay scorecard metrics."""

    summary: pl.DataFrame
    cdf_deltas: pl.DataFrame


def build_replay_scorecard(
    *,
    manifest_path: Path | str,
    model_version: str = "v15",
    driver_fixtures: Path | None = None,
) -> ReplayScorecard:
    """Compute replay scorecard metrics for archived manifest and proposals."""

    ctx = _load_context(Path(manifest_path))
    proposals = _load_proposals(ctx)
    markets = _load_markets(ctx)
    orderbooks = _load_orderbooks(ctx)
    fixtures_dir = driver_fixtures or _resolve_driver_fixtures(ctx)

    series_info = ctx.manifest.get("series", {})
    series_ticker = str(series_info.get("ticker", "")).upper()

    target_markets = {
        str(proposal.get("market_id"))
        for proposal in proposals
        if proposal.get("market_id") is not None
    }
    if not target_markets:
        target_markets = set(markets.keys())

    summary_records: list[dict[str, object]] = []
    delta_records: list[dict[str, object]] = []

    for market_id, market in markets.items():
        if target_markets and market_id not in target_markets:
            continue
        rungs = _build_rungs(market)
        if not rungs:
            continue
        strikes = [rung.strike for rung in rungs]
        market_pmf = pmf_from_quotes(rungs)
        market_survival = pmf_to_survival(market_pmf, strikes)

        try:
            strategy_pmf = _strategy_pmf_for_series(
                series=series_ticker,
                strikes=strikes,
                fixtures_dir=fixtures_dir,
                override=series_ticker,
                offline=True,
                model_version=model_version,
            )
        except Exception:
            strategy_pmf = market_pmf
        strategy_survival = pmf_to_survival(strategy_pmf, strikes)

        deltas = _compute_deltas(strategy_survival, market_survival)
        if not deltas:
            continue

        mean_abs_delta = float(sum(abs(delta) for delta in deltas) / len(deltas))
        max_abs_delta = float(max((abs(delta) for delta in deltas), default=0.0))
        kink_metrics = implied_cdf_kinks(strategy_survival)
        prob_gap = prob_sum_gap(strategy_pmf)

        summary_records.append(
            {
                "market_id": market_id,
                "market_ticker": market.ticker,
                "mean_abs_cdf_delta": mean_abs_delta,
                "max_abs_cdf_delta": max_abs_delta,
                "prob_sum_gap": prob_gap,
                "max_kink": kink_metrics.max_kink,
                "mean_abs_kink": kink_metrics.mean_abs_kink,
                "kink_count": kink_metrics.kink_count,
                "monotonicity_penalty": kink_metrics.monotonicity_penalty,
                "model_version": model_version,
            }
        )

        for strike, strat_surv, market_surv, delta in zip(
            strikes,
            strategy_survival,
            market_survival,
            deltas,
            strict=True,
        ):
            depth = _orderbook_depth(orderbooks.get(market_id))
            delta_records.append(
                {
                    "market_id": market_id,
                    "market_ticker": market.ticker,
                    "strike": strike,
                    "strategy_survival": float(strat_surv),
                    "market_survival": float(market_surv),
                    "delta": float(delta),
                    "orderbook_depth": depth,
                }
            )

    summary_df = pl.DataFrame(summary_records) if summary_records else _empty_summary_df(model_version)
    delta_df = pl.DataFrame(delta_records) if delta_records else _empty_delta_df()
    return ReplayScorecard(summary=summary_df, cdf_deltas=delta_df)


def _compute_deltas(strategy_survival: Sequence[float], market_survival: Sequence[float]) -> list[float]:
    length = min(len(strategy_survival), len(market_survival))
    return [
        float(strategy_survival[idx]) - float(market_survival[idx])
        for idx in range(length)
    ]


def _orderbook_depth(orderbook: Orderbook | None) -> int:
    if orderbook is None:
        return 0
    bids = getattr(orderbook, "bids", []) or []
    asks = getattr(orderbook, "asks", []) or []
    return len(bids) + len(asks)


def _empty_summary_df(model_version: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "market_id": pl.Series(dtype=pl.Utf8, name="market_id", values=[]),
            "market_ticker": pl.Series(dtype=pl.Utf8, name="market_ticker", values=[]),
            "mean_abs_cdf_delta": pl.Series(dtype=pl.Float64, name="mean_abs_cdf_delta", values=[]),
            "max_abs_cdf_delta": pl.Series(dtype=pl.Float64, name="max_abs_cdf_delta", values=[]),
            "prob_sum_gap": pl.Series(dtype=pl.Float64, name="prob_sum_gap", values=[]),
            "max_kink": pl.Series(dtype=pl.Float64, name="max_kink", values=[]),
            "mean_abs_kink": pl.Series(dtype=pl.Float64, name="mean_abs_kink", values=[]),
            "kink_count": pl.Series(dtype=pl.Int64, name="kink_count", values=[]),
            "monotonicity_penalty": pl.Series(dtype=pl.Float64, name="monotonicity_penalty", values=[]),
            "model_version": pl.Series(dtype=pl.Utf8, name="model_version", values=[]),
        }
    )


def _empty_delta_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "market_id": pl.Series(dtype=pl.Utf8, name="market_id", values=[]),
            "market_ticker": pl.Series(dtype=pl.Utf8, name="market_ticker", values=[]),
            "strike": pl.Series(dtype=pl.Float64, name="strike", values=[]),
            "strategy_survival": pl.Series(dtype=pl.Float64, name="strategy_survival", values=[]),
            "market_survival": pl.Series(dtype=pl.Float64, name="market_survival", values=[]),
            "delta": pl.Series(dtype=pl.Float64, name="delta", values=[]),
            "orderbook_depth": pl.Series(dtype=pl.Int64, name="orderbook_depth", values=[]),
        }
    ).head(0)
