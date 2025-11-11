"""Shared helpers for index ladder scanner CLIs."""

from __future__ import annotations

import argparse
import csv
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from datetime import UTC, datetime, time
from functools import lru_cache
from pathlib import Path

import polars as pl

from kalshi_alpha.config import load_index_ops_config
from kalshi_alpha.core.execution.fillratio import FillRatioEstimator, load_alpha
from kalshi_alpha.core.execution.index_models import load_alpha_curve, load_slippage_curve
from kalshi_alpha.core.execution.slippage import SlippageModel
from kalshi_alpha.core.kalshi_api import KalshiPublicClient
from kalshi_alpha.core.risk import PALGuard, PALPolicy
from kalshi_alpha.exec.ledger import PaperLedger, simulate_fills
from kalshi_alpha.exec.reports import write_markdown_report
from kalshi_alpha.exec.runners import scan_ladders

INDEX_OPS_CONFIG = load_index_ops_config()
LOGGER = logging.getLogger(__name__)

DEFAULT_CONTRACTS = 1
DEFAULT_KELLY_CAP = 0.25
DEFAULT_OUTPUT_ROOT = Path("reports/index_ladders")
MIN_ALPHA_FALLBACK = 0.4
PAL_POLICY_PATH = Path("configs/pal_policy.yaml")
PAL_POLICY_FALLBACK_PATH = Path("configs/pal_policy.example.yaml")
STRUCTURE_ARTIFACT_ROOT = Path("reports/_artifacts/structures")


@dataclass(frozen=True)
class ScannerConfig:
    series: Sequence[str]
    min_ev: float = float(INDEX_OPS_CONFIG.min_ev_usd)
    max_bins: int = int(INDEX_OPS_CONFIG.max_bins_per_series)
    contracts: int = DEFAULT_CONTRACTS
    kelly_cap: float = DEFAULT_KELLY_CAP
    offline: bool = False
    fixtures_root: Path = Path("tests/data_fixtures")
    output_root: Path = DEFAULT_OUTPUT_ROOT
    run_label: str = "index"
    timestamp: datetime | None = None
    now_override: datetime | None = None
    target_time_et: time | None = None
    paper_ledger: bool = True
    maker_only: bool = True
    emit_report: bool = True


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


@lru_cache(maxsize=16)
def _load_pal_policy(series: str) -> PALPolicy:
    normalized = series.upper()
    config_path = PAL_POLICY_PATH if PAL_POLICY_PATH.exists() else PAL_POLICY_FALLBACK_PATH
    try:
        return PALPolicy.from_yaml(config_path, series=normalized)
    except KeyError:
        return PALPolicy.from_yaml(PAL_POLICY_FALLBACK_PATH, series=normalized)


def _pal_guard(series: str) -> PALGuard:
    policy = _load_pal_policy(series)
    return PALGuard(policy)


def _simulate_execution(  # noqa: PLR0913
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
        except Exception as exc:  # pragma: no cover - robustness for missing books
            LOGGER.debug("failed to load orderbook %s: %s", proposal.market_id, exc)
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


def _write_outputs(  # noqa: PLR0913
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
    scan_timestamp = config.timestamp or datetime.now(tz=UTC)
    now_override = config.now_override or scan_timestamp
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
            maker_only=config.maker_only,
            allow_tails=False,
            risk_manager=None,
            max_var=None,
            offline=config.offline,
            sizing_mode="kelly",
            kelly_cap=config.kelly_cap,
            now_override=now_override,
            target_time_override=config.target_time_et,
        )
        if not outcome.proposals:
            series_dir = config.output_root / series.upper()
            series_dir.mkdir(parents=True, exist_ok=True)
            empty_csv = series_dir / f"{scan_timestamp.strftime('%Y%m%dT%H%M%SZ')}.csv"
            header = (
                "series,market,strike,side,event,q_yes,model_probability,"
                "market_probability,ev_after_fees,ev_per_contract,contracts,"
                "alpha,slippage,delta_bps\n"
            )
            empty_csv.write_text(header, encoding="utf-8")
            md_path = None
            if config.emit_report:
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
            timestamp=scan_timestamp,
            monitors=outcome.monitors,
            ledger=ledger,
        )
        ev_table = outcome.monitors.get("ev_honesty_table") if outcome.monitors else None
        if isinstance(ev_table, list):
            artifacts_dir = Path("reports/_artifacts")
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = artifacts_dir / f"ev_honesty_{series.upper()}.csv"
            with artifact_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "market",
                        "strike",
                        "maker_ev_per_contract_original",
                        "maker_ev_per_contract_replay",
                        "maker_ev_original",
                        "maker_ev_replay",
                        "delta",
                    ]
                )
                for entry in ev_table:
                    writer.writerow(
                        [
                            entry.get("market_ticker", "-"),
                            entry.get("strike", ""),
                            entry.get("maker_ev_per_contract_original", ""),
                            entry.get("maker_ev_per_contract_replay", ""),
                            entry.get("maker_ev_original", ""),
                            entry.get("maker_ev_replay", ""),
                            entry.get("delta", ""),
                        ]
                    )
            LOGGER.debug("wrote ev honesty table for %s to %s", series, artifact_path)
        else:
            artifacts_dir = Path("reports/_artifacts")
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = artifacts_dir / f"ev_honesty_{series.upper()}.csv"
            with artifact_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "market",
                        "strike",
                        "maker_ev_per_contract_original",
                        "maker_ev_per_contract_replay",
                        "maker_ev_original",
                        "maker_ev_replay",
                        "delta",
                        "note",
                    ]
                )
                writer.writerow(["-", "", "", "", "", "", "", "no_ev_honesty_data"])
            LOGGER.debug("wrote placeholder ev honesty table for %s to %s", series, artifact_path)
    if not config.emit_report:
        md_path = None
    results[series.upper()] = {"csv": csv_path, "markdown": md_path}
    _persist_structure_artifact(series, outcome.monitors)
    return results


def _persist_structure_artifact(series: str, monitors: Mapping[str, object] | None) -> None:
    if not monitors:
        return
    keys = (
        "range_ab_structures",
        "range_ab_sigma",
        "replacement_throttle",
        "contracts_per_quote",
        "regime",
        "regime_size_multiplier",
        "regime_slo_multiplier",
    )
    payload = {key: monitors.get(key) for key in keys if key in monitors}
    if not payload:
        return
    payload["series"] = series.upper()
    payload["generated_at"] = datetime.now(tz=UTC).isoformat()
    STRUCTURE_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    path = STRUCTURE_ARTIFACT_ROOT / f"{series.upper()}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_parser(default_series: Sequence[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan index ladders and emit paper reports.")
    parser.add_argument("--series", nargs="+", default=list(default_series), help="Series tickers to scan.")
    parser.add_argument(
        "--min-ev",
        type=float,
        default=float(INDEX_OPS_CONFIG.min_ev_usd),
        help="Minimum EV_after_fees per contract (USD).",
    )
    parser.add_argument(
        "--max-bins",
        type=int,
        default=int(INDEX_OPS_CONFIG.max_bins_per_series),
        help="Maximum number of bins to include per series.",
    )
    parser.add_argument("--contracts", type=int, default=DEFAULT_CONTRACTS, help="Contracts per quote (default: 1).")
    parser.add_argument("--kelly-cap", type=float, default=DEFAULT_KELLY_CAP, help="Truncated Kelly cap.")
    parser.add_argument("--offline", action="store_true", help="Use offline Kalshi fixtures.")
    parser.add_argument("--report", action="store_true", help="Generate markdown report output (default on).")
    parser.add_argument("--no-report", action="store_true", help="Skip markdown report output.")
    parser.add_argument("--paper-ledger", action="store_true", help="Simulate paper ledger fills (default on).")
    parser.add_argument("--no-paper-ledger", action="store_true", help="Skip paper ledger simulation.")
    parser.add_argument("--maker-only", action="store_true", help="Restrict to maker-side proposals (default on).")
    parser.add_argument("--no-maker-only", action="store_true", help="Allow taker-side proposals.")
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=Path("tests/data_fixtures"),
        help="Fixture root directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Report output root directory.",
    )
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
