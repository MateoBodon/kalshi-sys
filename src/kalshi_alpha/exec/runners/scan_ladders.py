"""CLI scanner that produces dry-run order proposals for Kalshi ladders."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from kalshi_alpha.core.fees import DEFAULT_FEE_SCHEDULE
from kalshi_alpha.core.kalshi_api import KalshiPublicClient, Series
from kalshi_alpha.core.pricing import (
    LadderBinProbability,
    LadderRung,
    Liquidity,
    OrderSide,
    pmf_from_quotes,
)
from kalshi_alpha.core.risk import OrderProposal, PALGuard, PALPolicy, max_loss_for_order
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


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    fixtures_root = Path(args.fixtures_root).resolve()
    api_fixtures = fixtures_root / "kalshi"
    driver_fixtures = fixtures_root / "drivers"

    client = KalshiPublicClient(
        offline_dir=api_fixtures,
        use_offline=True,
    )

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
    pal_guard = PALGuard(policy)

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
    )

    output_path = write_proposals(
        series=args.series,
        proposals=proposals,
        output_dir=Path(args.output_dir),
    )
    if not args.quiet:
        print(f"Wrote {len(proposals)} proposals to {output_path}")


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
            )
            proposals.extend(rung_proposals)
    return proposals


def _strategy_pmf_for_series(
    *,
    series: str,
    strikes: list[float],
    fixtures_dir: Path,
    override: str,
) -> list[LadderBinProbability]:
    pick = override.lower()
    ticker = series.upper()
    if pick == "auto":
        pick = ticker.lower()

    if pick == "cpi" and ticker == "CPI":
        return cpi.strategy_pmf(strikes, fixtures_dir=fixtures_dir)
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

        total_ev = expected_value_summary(
            contracts=max_contracts,
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

        proposal = Proposal(
            market_id=market_id,
            market_ticker=market_ticker,
            strike=rung.strike,
            side=best_side.name,
            contracts=max_contracts,
            maker_ev=total_ev[maker_key],
            taker_ev=total_ev[taker_key],
            maker_ev_per_contract=per_contract[maker_key],
            taker_ev_per_contract=per_contract[taker_key],
            strategy_probability=event_probability,
            market_yes_price=yes_price,
            survival_market=survival_market,
            survival_strategy=event_probability,
            max_loss=max_loss_single * max_contracts,
        )
        pal_guard.register(
            OrderProposal(
                strike_id=order_id,
                yes_price=yes_price,
                contracts=max_contracts,
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
