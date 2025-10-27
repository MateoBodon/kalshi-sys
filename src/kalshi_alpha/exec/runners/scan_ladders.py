"""CLI scanner that produces dry-run order proposals for Kalshi ladders."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from kalshi_alpha.core.fees import DEFAULT_FEE_SCHEDULE
from kalshi_alpha.core.kalshi_api import KalshiPublicClient, Orderbook, Series
from kalshi_alpha.core.pricing import (
    LadderBinProbability,
    LadderRung,
    Liquidity,
    OrderSide,
    pmf_from_quotes,
)
from kalshi_alpha.core.risk import (
    OrderProposal,
    PALGuard,
    PALPolicy,
    PortfolioConfig,
    PortfolioRiskManager,
    max_loss_for_order,
)
from kalshi_alpha.core.sizing import apply_caps, kelly_yes_no, truncate_kelly
from kalshi_alpha.drivers.aaa_gas import fetch as aaa_fetch
from kalshi_alpha.drivers.aaa_gas import ingest as aaa_ingest
from kalshi_alpha.exec.ledger import PaperLedger, simulate_fills
from kalshi_alpha.exec.reports import write_markdown_report
from kalshi_alpha.exec.scanners import cpi
from kalshi_alpha.exec.scanners.utils import expected_value_summary, pmf_to_survival

DEFAULT_MIN_EV = 0.05  # USD per contract after maker fees
DEFAULT_CONTRACTS = 10


@dataclass
class Proposal:
    market_id: str
    market_ticker: str
    strike: float
    side: str
    contracts: int
    maker_ev: float
    taker_ev: float
    maker_ev_per_contract: float
    taker_ev_per_contract: float
    strategy_probability: float
    market_yes_price: float
    survival_market: float
    survival_strategy: float
    max_loss: float
    strategy: str
    metadata: dict[str, object] | None = None


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    fixtures_root = Path(args.fixtures_root).resolve()
    driver_fixtures = fixtures_root / "drivers"

    client = _build_client(fixtures_root)
    pal_guard = _build_pal_guard(args)
    risk_manager = _build_risk_manager(args)
    proposals = scan_series(
        series=args.series,
        client=client,
        min_ev=args.min_ev,
        contracts=args.contracts,
        pal_guard=pal_guard,
        driver_fixtures=driver_fixtures,
        strategy_name=args.strategy,
        maker_only=args.maker_only,
        allow_tails=args.allow_tails,
        risk_manager=risk_manager,
        max_var=args.max_var,
        offline=args.offline,
        sizing_mode=args.sizing,
        kelly_cap=args.kelly_cap,
    )

    ledger = _maybe_simulate_ledger(args, proposals, client)
    if proposals:
        _attach_series_metadata(
            proposals=proposals,
            series=args.series,
            driver_fixtures=driver_fixtures,
            offline=args.offline,
        )
    output_path = write_proposals(
        series=args.series,
        proposals=proposals,
        output_dir=Path(args.output_dir),
    )
    if not args.quiet:
        print(f"Wrote {len(proposals)} proposals to {output_path}")

    _maybe_write_report(args, proposals, ledger)


def _build_client(fixtures_root: Path) -> KalshiPublicClient:
    api_fixtures = fixtures_root / "kalshi"
    return KalshiPublicClient(
        offline_dir=api_fixtures,
        use_offline=True,
    )


def _build_pal_guard(args: argparse.Namespace) -> PALGuard:
    policy_path = Path(args.pal_policy) if args.pal_policy else Path("configs/pal_policy.yaml")
    if not policy_path.exists():
        policy_path = Path("configs/pal_policy.example.yaml")
    policy = PALPolicy.from_yaml(policy_path)
    if args.max_loss_per_strike is not None:
        policy = PALPolicy(
            series=policy.series,
            default_max_loss=args.max_loss_per_strike,
            per_strike=dict(policy.per_strike),
        )
    return PALGuard(policy)


def _build_risk_manager(args: argparse.Namespace) -> PortfolioRiskManager | None:
    if args.portfolio_config:
        config = PortfolioConfig.from_yaml(Path(args.portfolio_config))
        return PortfolioRiskManager(config)
    if args.max_var is not None:
        fallback_config = PortfolioConfig(factor_vols={"TOTAL": 1.0}, strategy_betas={})
        return PortfolioRiskManager(fallback_config)
    return None


def _maybe_simulate_ledger(
    args: argparse.Namespace,
    proposals: Sequence[Proposal],
    client: KalshiPublicClient,
) -> PaperLedger | None:
    if not proposals or not (args.paper_ledger or args.report):
        return None
    orderbooks: dict[str, Orderbook] = {}
    for proposal in proposals:
        if proposal.market_id not in orderbooks:
            orderbooks[proposal.market_id] = client.get_orderbook(proposal.market_id)
    ledger = simulate_fills(
        proposals,
        orderbooks,
        artifacts_dir=Path("reports/_artifacts") if (args.paper_ledger or args.report) else None,
    )
    if args.paper_ledger and not args.quiet:
        stats = ledger.to_dict()
        print(
            f"Paper ledger trades={stats['trades']} "
            f"expected_pnl={stats['expected_pnl']:.2f} max_loss={stats['max_loss']:.2f}"
        )
    return ledger


def _attach_series_metadata(
    *,
    proposals: Sequence[Proposal],
    series: str,
    driver_fixtures: Path,
    offline: bool,
) -> None:
    if not proposals or series.upper() not in {"CPI", "GAS"}:
        return
    fixtures_aaa = driver_fixtures / "aaa"
    if offline and fixtures_aaa.exists() and not aaa_fetch.DAILY_PATH.exists():
        sample_csv = fixtures_aaa / "AAA_daily_gas_price_regular_sample.csv"
        if sample_csv.exists():
            aaa_ingest.bootstrap_from_csv(sample_csv)
    try:
        latest = aaa_fetch.fetch_latest(
            offline=offline,
            fixtures_dir=fixtures_aaa if fixtures_aaa.exists() else None,
        )
    except Exception:  # pragma: no cover - robustness
        latest = None
    mtd_avg = aaa_fetch.mtd_average(latest.as_of_date if latest else None)
    delta = (latest.price - mtd_avg) if latest and mtd_avg is not None else None
    suspicious = abs(delta) > 0.25 if delta is not None else False
    metadata = {
        "aaa_price": latest.price if latest else None,
        "aaa_as_of": latest.as_of_date.isoformat() if latest else None,
        "aaa_mtd_average": mtd_avg,
        "aaa_delta": delta,
        "stale": latest is None,
        "suspicious": suspicious,
    }
    for proposal in proposals:
        existing = dict(proposal.metadata) if proposal.metadata else {}
        existing.setdefault("aaa", metadata.copy())
        proposal.metadata = existing


def _maybe_write_report(
    args: argparse.Namespace,
    proposals: Sequence[Proposal],
    ledger: PaperLedger | None,
) -> None:
    if not args.report:
        return
    report_path = write_markdown_report(
        series=args.series,
        proposals=proposals,
        ledger=ledger,
        output_dir=Path("reports") / args.series.upper(),
    )
    if not args.quiet:
        print(f"Wrote report to {report_path}")


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan Kalshi ladders and output dry-run proposals."
    )
    parser.add_argument(
        "--series",
        required=True,
        help="Kalshi series ticker, e.g. CPI",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Alias for producing proposals only (default).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use offline fixtures for driver data.",
    )
    parser.add_argument(
        "--fixtures-root",
        default="tests/data_fixtures",
        help="Offline fixtures root directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="exec/proposals",
        help="Directory for proposal JSON outputs.",
    )
    parser.add_argument(
        "--min-ev",
        type=float,
        default=DEFAULT_MIN_EV,
        help="Minimum maker EV per contract (USD).",
    )
    parser.add_argument(
        "--contracts",
        type=int,
        default=DEFAULT_CONTRACTS,
        help="Target contracts per proposal.",
    )
    parser.add_argument(
        "--sizing",
        default="fixed",
        choices=["fixed", "kelly"],
        help="Position sizing methodology.",
    )
    parser.add_argument(
        "--kelly-cap",
        type=float,
        default=0.25,
        help="Maximum Kelly fraction when sizing via Kelly.",
    )
    parser.add_argument(
        "--strategy",
        default="auto",
        choices=["auto", "cpi"],
        help="Override strategy module selection.",
    )
    parser.add_argument(
        "--maker-only",
        action="store_true",
        help="Only consider maker-side executions.",
    )
    parser.add_argument(
        "--max-loss-per-strike",
        type=float,
        help="Override PAL default max loss per strike (USD).",
    )
    parser.add_argument(
        "--pal-policy",
        help="Path to PAL policy YAML overriding default.",
    )
    parser.add_argument(
        "--allow-tails",
        action="store_true",
        help="Permit proposals outside adjacent bins to the model mode.",
    )
    parser.add_argument(
        "--max-var",
        type=float,
        help="Maximum portfolio VaR allowed (USD).",
    )
    parser.add_argument(
        "--portfolio-config",
        type=Path,
        help="Path to portfolio factor configuration YAML file.",
    )
    parser.add_argument(
        "--paper-ledger",
        action="store_true",
        help="Simulate paper fills using top-of-book data.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Write markdown report for the scan.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress stdout summary.",
    )
    return parser.parse_args(argv)


def scan_series(  # noqa: PLR0913
    *,
    series: str,
    client: KalshiPublicClient,
    min_ev: float,
    contracts: int,
    pal_guard: PALGuard,
    driver_fixtures: Path,
    strategy_name: str,
    maker_only: bool,
    allow_tails: bool,
    risk_manager: PortfolioRiskManager | None,
    max_var: float | None,
    offline: bool,
    sizing_mode: str,
    kelly_cap: float,
) -> list[Proposal]:
    series_obj = _find_series(client, series)
    events = client.get_events(series_obj.id)
    proposals: list[Proposal] = []

    for event in events:
        markets = client.get_markets(event.id)
        for market in markets:
            if not market.ladder_strikes or not market.ladder_yes_prices:
                continue

            rungs = [
                LadderRung(strike=float(strike), yes_price=float(price))
                for strike, price in zip(
                    market.ladder_strikes,
                    market.ladder_yes_prices,
                    strict=True,
                )
            ]
            market_pmf = pmf_from_quotes(rungs)
            market_survival = _market_survival_from_pmf(market_pmf, rungs)

            strategy_pmf = _strategy_pmf_for_series(
                series=series_obj.ticker,
                strikes=[rung.strike for rung in rungs],
                fixtures_dir=driver_fixtures,
                override=strategy_name,
                offline=offline,
            )
            strategy_survival = pmf_to_survival(strategy_pmf, [rung.strike for rung in rungs])
            allowed_indices = None
            if not allow_tails:
                allowed_indices = _adjacent_indices(strategy_pmf, len(rungs))

            rung_proposals = _evaluate_market(
                market_id=market.id,
                market_ticker=market.ticker,
                rungs=rungs,
                market_survival=market_survival,
                strategy_survival=strategy_survival,
                min_ev=min_ev,
                contracts=contracts,
                pal_guard=pal_guard,
                allowed_indices=allowed_indices,
                maker_only=maker_only,
                risk_manager=risk_manager,
                max_var=max_var,
                strategy_name=series_obj.ticker.upper(),
                sizing_mode=sizing_mode,
                kelly_cap=kelly_cap,
            )
            proposals.extend(rung_proposals)
    return proposals


def _strategy_pmf_for_series(
    *,
    series: str,
    strikes: list[float],
    fixtures_dir: Path,
    override: str,
    offline: bool,
) -> list[LadderBinProbability]:
    pick = override.lower()
    ticker = series.upper()
    if pick == "auto":
        pick = ticker.lower()

    if pick == "cpi" and ticker == "CPI":
        return cpi.strategy_pmf(strikes, fixtures_dir=fixtures_dir, offline=offline)
    raise NotImplementedError(f"No strategy PMF implemented for series {series}")


def _market_survival_from_pmf(
    pmf: Sequence[LadderBinProbability],
    rungs: Sequence[LadderRung],
) -> list[float]:
    return pmf_to_survival(pmf, [rung.strike for rung in rungs])


def _adjacent_indices(
    pmf: Sequence[LadderBinProbability],
    rung_count: int,
) -> set[int]:
    if rung_count == 0:
        return set()
    max_prob = max(bin_prob.probability for bin_prob in pmf)
    tolerance = 1e-9
    mode_indices = [
        idx
        for idx, bin_prob in enumerate(pmf[:rung_count])
        if bin_prob.probability >= max_prob - tolerance
    ]
    if not mode_indices:
        mode_indices = [min(len(pmf) - 1, rung_count - 1)]
    allowed: set[int] = set()
    for idx in mode_indices:
        start = max(idx - 1, 0)
        end = min(idx + 1, rung_count - 1)
        allowed.update(range(start, end + 1))
    return allowed


def _evaluate_market(  # noqa: PLR0913
    *,
    market_id: str,
    market_ticker: str,
    rungs: Sequence[LadderRung],
    market_survival: Sequence[float],
    strategy_survival: Sequence[float],
    min_ev: float,
    contracts: int,
    pal_guard: PALGuard,
    allowed_indices: set[int] | None,
    maker_only: bool,
    risk_manager: PortfolioRiskManager | None,
    max_var: float | None,
    strategy_name: str,
    sizing_mode: str,
    kelly_cap: float,
) -> list[Proposal]:
    proposals: list[Proposal] = []

    for index, rung in enumerate(rungs):
        if allowed_indices is not None and index not in allowed_indices:
            continue
        yes_price = rung.yes_price
        event_probability = strategy_survival[index]
        survival_market = market_survival[index]

        per_contract = expected_value_summary(
            contracts=1,
            yes_price=yes_price,
            event_probability=event_probability,
            schedule=DEFAULT_FEE_SCHEDULE,
            market_name=market_ticker,
        )
        best_side, best_ev = _choose_side(per_contract, maker_only=maker_only)
        if best_ev < min_ev:
            continue

        order_id = f"{market_ticker}:{rung.strike}"
        max_loss_single = max_loss_for_order(
            OrderProposal(
                strike_id=order_id,
                yes_price=yes_price,
                contracts=1,
                side=best_side,
                liquidity=Liquidity.MAKER,
                market_name=market_ticker,
            )
        )
        remaining_limit = pal_guard.policy.limit_for_strike(order_id) - pal_guard.exposure_for(
            order_id
        )
        if max_loss_single <= 0 or remaining_limit <= 0:
            continue
        max_contracts = min(int(remaining_limit // max_loss_single), contracts)
        if max_contracts <= 0:
            continue

        contract_count = max_contracts
        if sizing_mode == "kelly":
            if best_side is OrderSide.YES:
                raw_fraction = kelly_yes_no(event_probability, yes_price)
            else:
                raw_fraction = kelly_yes_no(1.0 - event_probability, 1.0 - yes_price)
            sized_fraction = truncate_kelly(raw_fraction, kelly_cap)
            if sized_fraction <= 0.0:
                continue
            capital_base = pal_guard.policy.default_max_loss
            raw_risk = capital_base * sized_fraction
            var_remaining = None
            if risk_manager and max_var is not None:
                var_remaining = max(max_var - risk_manager.current_var(), 0.0)
            capped_risk = apply_caps(
                raw_risk,
                pal=remaining_limit,
                max_loss_per_strike=pal_guard.policy.limit_for_strike(order_id),
                max_var=var_remaining,
            )
            if capped_risk <= 0.0:
                continue
            contract_count = min(max_contracts, int(capped_risk // max_loss_single))
            if contract_count <= 0:
                continue

        total_ev = expected_value_summary(
            contracts=contract_count,
            yes_price=yes_price,
            event_probability=event_probability,
            schedule=DEFAULT_FEE_SCHEDULE,
            market_name=market_ticker,
        )

        maker_key = "maker_yes" if best_side is OrderSide.YES else "maker_no"
        taker_key = "taker_yes" if best_side is OrderSide.YES else "taker_no"
        if maker_only:
            total_ev[taker_key] = 0.0
            per_contract[taker_key] = 0.0

        total_max_loss = max_loss_single * contract_count
        if risk_manager and not risk_manager.can_accept(
            strategy=strategy_name,
            max_loss=total_max_loss,
            max_var=max_var,
        ):
            continue

        proposal = Proposal(
            market_id=market_id,
            market_ticker=market_ticker,
            strike=rung.strike,
            side=best_side.name,
            contracts=contract_count,
            maker_ev=total_ev[maker_key],
            taker_ev=total_ev[taker_key],
            maker_ev_per_contract=per_contract[maker_key],
            taker_ev_per_contract=per_contract[taker_key],
            strategy_probability=event_probability,
            market_yes_price=yes_price,
            survival_market=survival_market,
            survival_strategy=event_probability,
            max_loss=total_max_loss,
            strategy=strategy_name,
            metadata=None,
        )
        pal_guard.register(
            OrderProposal(
                strike_id=order_id,
                yes_price=yes_price,
                contracts=contract_count,
                side=best_side,
                liquidity=Liquidity.MAKER,
                market_name=market_ticker,
            )
        )
        proposals.append(proposal)
    return proposals


def _choose_side(
    per_contract_evs: dict[str, float],
    *,
    maker_only: bool,
) -> tuple[OrderSide, float]:
    maker_yes = per_contract_evs["maker_yes"]
    maker_no = per_contract_evs["maker_no"]
    if maker_only:
        return (OrderSide.YES, maker_yes) if maker_yes >= maker_no else (OrderSide.NO, maker_no)

    taker_yes = per_contract_evs.get("taker_yes", float("-inf"))
    taker_no = per_contract_evs.get("taker_no", float("-inf"))
    best_yes = max(maker_yes, taker_yes)
    best_no = max(maker_no, taker_no)
    if best_yes >= best_no:
        return OrderSide.YES, best_yes
    return OrderSide.NO, best_no


def write_proposals(*, series: str, proposals: Sequence[Proposal], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    dated_dir = output_dir / series.upper()
    dated_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(tz=UTC).date()
    filename = dated_dir / f"{today.isoformat()}.json"
    counter = 1
    while filename.exists():
        filename = dated_dir / f"{today.isoformat()}_{counter}.json"
        counter += 1
    payload = {
        "series": series.upper(),
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "proposals": [asdict(proposal) for proposal in proposals],
    }
    filename.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return filename


def _find_series(client: KalshiPublicClient, ticker: str) -> Series:
    series_list = client.get_series()
    for series in series_list:
        if series.ticker.upper() == ticker.upper():
            return series
    raise ValueError(f"Series {ticker} not found in fixtures")


if __name__ == "__main__":
    main()
