"""Shared helpers for index ladder scanner CLIs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Sequence

import polars as pl

from kalshi_alpha.core.execution.fillratio import FillRatioEstimator, load_alpha
from kalshi_alpha.core.execution.index_models import load_alpha_curve, load_slippage_curve
from kalshi_alpha.core.execution.slippage import SlippageModel
from kalshi_alpha.core.kalshi_api import KalshiPublicClient
from kalshi_alpha.core.pricing import OrderSide
from kalshi_alpha.core.risk import PALGuard, PALPolicy
from kalshi_alpha.exec.ledger import PaperLedger, simulate_fills
from kalshi_alpha.exec.reports import write_markdown_report
from kalshi_alpha.exec.runners import scan_ladders


DEFAULT_CONTRACTS = 1
DEFAULT_KELLY_CAP = 0.25
DEFAULT_OUTPUT_ROOT = Path("reports/index_ladders")
MIN_ALPHA_FALLBACK = 0.4


@dataclass(frozen=True)
class ScannerConfig:
    series: Sequence[str]
    min_ev: float = 0.05
    max_bins: int = 2
    contracts: int = DEFAULT_CONTRACTS
    kelly_cap: float = DEFAULT_KELLY_CAP
    offline: bool = False
    fixtures_root: Path = Path("tests/data_fixtures")
    output_root: Path = DEFAULT_OUTPUT_ROOT
    run_label: str = "index"
    timestamp: datetime | None = None


@dataclass(frozen=True)
class OpportunityRow:
    series: str
    market: str
    strike: float
    side: str
    event: str | None
    yes_price: float
    model_probability: float
    market_probability: float
    ev_after_fees: float
    ev_per_contract: float
    contracts: int
    alpha: float
    slippage: float
    delta_bps: float


def _build_client(fixtures_root: Path, *, offline: bool) -> KalshiPublicClient:
    fixtures = fixtures_root / "kalshi"
    return KalshiPublicClient(offline_dir=fixtures, use_offline=offline)


def _pal_guard(series: str) -> PALGuard:
    policy = PALPolicy(series=series.upper(), default_max_loss=1_000_000.0)
    return PALGuard(policy)


def _simulate_execution(
    *,
    proposals: Sequence[scan_ladders.Proposal],
    client: KalshiPublicClient,
    orderbooks: dict[str, scan_ladders.Orderbook],
    series_label: str,
    events: Sequence[scan_ladders.Event],
    markets: Sequence[scan_ladders.Market],
) -> PaperLedger:
    alpha_value = load_alpha(series_label) or MIN_ALPHA_FALLBACK
    estimator = FillRatioEstimator(alpha=alpha_value)
    alpha_curve = load_alpha_curve(series_label)
    slippage_curve = load_slippage_curve(series_label)
    event_lookup = {market.id: market.ticker for market in markets}
    orderbook_cache = dict(orderbooks)
    for proposal in proposals:
        if proposal.market_id in orderbook_cache:
            continue
        try:
            orderbook_cache[proposal.market_id] = client.get_orderbook(proposal.market_id)
        except Exception:  # pragma: no cover - robustness for missing books
            continue

    ledger = simulate_fills(
        proposals,
        orderbook_cache,
        fill_estimator=estimator if alpha_curve is None else None,
        alpha_curve=alpha_curve,
        slippage_curve=slippage_curve,
        ledger_series=series_label.upper(),
        market_event_lookup=event_lookup,
        mode="top",
        slippage_model=SlippageModel(mode="top"),
    )
    return ledger


def _pair_rows(
    *,
    proposals: Sequence[scan_ladders.Proposal],
    ledger: PaperLedger,
    min_ev: float,
) -> list[OpportunityRow]:
    if not proposals or not ledger.records:
        return []
    rows: list[OpportunityRow] = []
    for record in ledger.records:
        proposal = record.proposal
        if proposal.side.upper() != "YES":
            continue
        ev_per_contract = float(proposal.maker_ev_per_contract)
        if ev_per_contract < min_ev:
            continue
        alpha = float(record.alpha_row or record.fill_ratio or 0.0)
        slippage = float(record.slippage or 0.0)
        delta_component = ev_per_contract * alpha - slippage
        delta_bps = delta_component * 10000.0
        rows.append(
            OpportunityRow(
                series=proposal.series.upper(),
                market=proposal.market_ticker,
                strike=float(proposal.strike),
                side=proposal.side.upper(),
                event=record.event_ticker or None,
                yes_price=float(proposal.market_yes_price),
                model_probability=float(proposal.strategy_probability),
                market_probability=float(proposal.survival_market),
                ev_after_fees=float(proposal.maker_ev),
                ev_per_contract=ev_per_contract,
                contracts=int(proposal.contracts),
                alpha=alpha,
                slippage=slippage,
                delta_bps=delta_bps,
            )
        )
    return rows


def _select_top_bins(rows: Sequence[OpportunityRow], *, max_bins: int) -> list[OpportunityRow]:
    if max_bins <= 0:
        return []
    sorted_rows = sorted(rows, key=lambda row: (row.delta_bps, row.ev_per_contract), reverse=True)
    unique_bins: list[OpportunityRow] = []
    seen: set[tuple[str, float]] = set()
    for row in sorted_rows:
        key = (row.market, row.strike)
        if key in seen:
            continue
        unique_bins.append(row)
        seen.add(key)
        if len(unique_bins) >= max_bins:
            break
    return unique_bins


def _write_outputs(
    *,
    rows: Sequence[OpportunityRow],
    series: str,
    output_dir: Path,
    timestamp: datetime,
    monitors: dict[str, object],
    ledger: PaperLedger,
) -> tuple[Path | None, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = timestamp.strftime("%Y%m%dT%H%M%SZ")
    csv_path = output_dir / f"{stamp}.csv"
    selected_keys = {(row.market, row.strike) for row in rows}
    filtered_records = [
        record
        for record in ledger.records
        if (record.proposal.market_ticker, float(record.proposal.strike)) in selected_keys
    ]
    filtered_ledger = PaperLedger(
        records=filtered_records,
        series=ledger.series,
        event_lookup=ledger.event_lookup,
    )
    if rows:
        frame = pl.DataFrame(
            {
                "series": [row.series for row in rows],
                "market": [row.market for row in rows],
                "strike": [row.strike for row in rows],
                "side": [row.side for row in rows],
                "event": [row.event for row in rows],
                "q_yes": [row.yes_price for row in rows],
                "model_probability": [row.model_probability for row in rows],
                "market_probability": [row.market_probability for row in rows],
                "ev_after_fees": [row.ev_after_fees for row in rows],
                "ev_per_contract": [row.ev_per_contract for row in rows],
                "contracts": [row.contracts for row in rows],
                "alpha": [row.alpha for row in rows],
                "slippage": [row.slippage for row in rows],
                "delta_bps": [row.delta_bps for row in rows],
            }
        )
        frame.write_csv(csv_path, float_precision=6)
    else:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "series",
                    "market",
                    "strike",
                    "side",
                    "event",
                    "q_yes",
                    "model_probability",
                    "market_probability",
                    "ev_after_fees",
                    "ev_per_contract",
                    "contracts",
                    "alpha",
                    "slippage",
                    "delta_bps",
                ]
            )

    proposals_for_report = [record.proposal for record in filtered_records]
    md_path = write_markdown_report(
        series=series,
        proposals=proposals_for_report,
        ledger=filtered_ledger,
        output_dir=output_dir,
        monitors=monitors,
        exposure_summary={},
        manifest_path=None,
        go_status=True,
        pilot_metadata={},
    )
    return csv_path, md_path


def run_index_scan(config: ScannerConfig) -> dict[str, dict[str, Path | None]]:
    timestamp = config.timestamp or datetime.now(tz=UTC)
    client = _build_client(config.fixtures_root, offline=config.offline)
    results: dict[str, dict[str, Path | None]] = {}
    for series in config.series:
        pal_guard = _pal_guard(series)
        driver_fixtures = config.fixtures_root / "drivers"
        outcome = scan_ladders.scan_series(
            series=series,
            client=client,
            min_ev=config.min_ev,
            contracts=config.contracts,
            pal_guard=pal_guard,
            driver_fixtures=driver_fixtures,
            strategy_name="auto",
            maker_only=True,
            allow_tails=False,
            risk_manager=None,
            max_var=None,
            offline=config.offline,
            sizing_mode="kelly",
            kelly_cap=config.kelly_cap,
        )
        if not outcome.proposals:
            series_dir = config.output_root / series.upper()
            series_dir.mkdir(parents=True, exist_ok=True)
            empty_csv = series_dir / f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}.csv"
            empty_csv.write_text("series,market,strike,side,event,q_yes,model_probability,market_probability,ev_after_fees,ev_per_contract,contracts,alpha,slippage,delta_bps\n", encoding="utf-8")
            md_path = write_markdown_report(
                series=series,
                proposals=[],
                ledger=PaperLedger(records=[], series=series.upper()),
                output_dir=series_dir,
                monitors=outcome.monitors,
                exposure_summary={},
                manifest_path=None,
                go_status=False,
                pilot_metadata={},
            )
            results[series.upper()] = {"csv": empty_csv, "markdown": md_path}
            continue
        ledger = _simulate_execution(
            proposals=outcome.proposals,
            client=client,
            orderbooks=outcome.books_at_scan,
            series_label=series,
            events=outcome.events,
            markets=outcome.markets,
        )
        rows = _pair_rows(proposals=outcome.proposals, ledger=ledger, min_ev=config.min_ev)
        selected = _select_top_bins(rows, max_bins=config.max_bins)
        series_dir = config.output_root / series.upper()
        csv_path, md_path = _write_outputs(
            rows=selected,
            series=series,
            output_dir=series_dir,
            timestamp=timestamp,
            monitors=outcome.monitors,
            ledger=ledger,
        )
        results[series.upper()] = {"csv": csv_path, "markdown": md_path}
    return results


def build_parser(default_series: Sequence[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan index ladders and emit paper reports.")
    parser.add_argument("--series", nargs="+", default=list(default_series), help="Series tickers to scan.")
    parser.add_argument("--min-ev", type=float, default=0.05, help="Minimum EV_after_fees per contract (USD).")
    parser.add_argument("--max-bins", type=int, default=2, help="Maximum number of bins to include per series.")
    parser.add_argument("--contracts", type=int, default=DEFAULT_CONTRACTS, help="Contracts per quote (default: 1).")
    parser.add_argument("--kelly-cap", type=float, default=DEFAULT_KELLY_CAP, help="Truncated Kelly cap.")
    parser.add_argument("--offline", action="store_true", help="Use offline Kalshi fixtures.")
    parser.add_argument("--fixtures-root", type=Path, default=Path("tests/data_fixtures"), help="Fixture root directory.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Report output root directory.")
    parser.add_argument("--now", type=str, help="Override timestamp (ISO-8601).")
    return parser


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid ISO timestamp") from None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)
