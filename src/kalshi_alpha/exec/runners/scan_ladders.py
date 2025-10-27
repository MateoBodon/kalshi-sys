"""CLI scanner that produces dry-run order proposals for Kalshi ladders."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from kalshi_alpha.core.archive import archive_scan, replay_manifest
from kalshi_alpha.core.execution.fillratio import FillRatioEstimator, tune_alpha
from kalshi_alpha.core.execution.slippage import SlippageModel
from kalshi_alpha.core.fees import DEFAULT_FEE_SCHEDULE
from kalshi_alpha.core.kalshi_api import Event, KalshiPublicClient, Market, Orderbook, Series
from kalshi_alpha.core.pricing import (
    LadderBinProbability,
    LadderRung,
    Liquidity,
    OrderSide,
    pmf_from_quotes,
)
from kalshi_alpha.core.risk import (
    drawdown,
    OrderProposal,
    PALGuard,
    PALPolicy,
    PortfolioConfig,
    PortfolioRiskManager,
    max_loss_for_order,
)
from kalshi_alpha.core.sizing import apply_caps, kelly_yes_no, scale_kelly, truncate_kelly
from kalshi_alpha.drivers.aaa_gas import fetch as aaa_fetch
from kalshi_alpha.drivers.aaa_gas import ingest as aaa_ingest
from kalshi_alpha.datastore.paths import PROC_ROOT, RAW_ROOT
from kalshi_alpha.exec.ledger import PaperLedger, simulate_fills
from kalshi_alpha.exec.reports import write_markdown_report
from kalshi_alpha.exec.scanners import cpi
from kalshi_alpha.exec.scanners.utils import expected_value_summary, pmf_to_survival
from kalshi_alpha.strategies import claims as claims_strategy
from kalshi_alpha.strategies import cpi as cpi_strategy
from kalshi_alpha.strategies import teny as teny_strategy
from kalshi_alpha.strategies import weather as weather_strategy

DEFAULT_MIN_EV = 0.05  # USD per contract after maker fees
DEFAULT_CONTRACTS = 10
DEFAULT_FILL_ALPHA = 0.6


def _resolve_fill_alpha_arg(fill_alpha_arg: object, series: str) -> tuple[float, bool]:
    if fill_alpha_arg is None:
        return DEFAULT_FILL_ALPHA, False
    if isinstance(fill_alpha_arg, str):
        value = fill_alpha_arg.strip().lower()
        if value == "auto":
            tuned = tune_alpha(series, RAW_ROOT / "kalshi")
            if tuned is not None:
                return float(tuned), True
            return DEFAULT_FILL_ALPHA, True
        try:
            return float(value), False
        except ValueError:
            return DEFAULT_FILL_ALPHA, False
    try:
        return float(fill_alpha_arg), False
    except (TypeError, ValueError):
        return DEFAULT_FILL_ALPHA, False


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


@dataclass
class ScanOutcome:
    proposals: list[Proposal]
    monitors: dict[str, object] = field(default_factory=dict)
    cdf_diffs: list[dict[str, object]] = field(default_factory=list)
    series: Series | None = None
    events: list[Event] = field(default_factory=list)
    markets: list[Market] = field(default_factory=list)


class _LossBudget:
    def __init__(self, cap: float | None) -> None:
        self.cap = cap if cap is not None and cap > 0.0 else None
        self.remaining: float | None = float(self.cap) if self.cap is not None else None

    def max_contracts(self, unit_loss: float, requested: int) -> int:
        if self.cap is None or unit_loss <= 0 or requested <= 0:
            return requested
        if self.remaining is None or self.remaining <= 0:
            return 0
        allowed = int(self.remaining // unit_loss)
        return min(requested, allowed)

    def consume(self, loss: float) -> None:
        if self.cap is None or loss <= 0:
            return
        if self.remaining is None:
            self.remaining = float(self.cap)
        self.remaining = max(0.0, self.remaining - loss)


_PORTFOLIO_CONFIG_CACHE: PortfolioConfig | None = None

def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    fixtures_root = Path(args.fixtures_root).resolve()
    driver_fixtures = fixtures_root / "drivers"

    if args.online and args.offline:
        raise ValueError("Cannot specify both --online and --offline.")

    client = _build_client(fixtures_root, use_online=args.online)
    fill_alpha_value, fill_alpha_auto = _resolve_fill_alpha_arg(args.fill_alpha, args.series)
    pal_guard = _build_pal_guard(args)
    risk_manager = _build_risk_manager(args)
    offline_mode = args.offline or not args.online

    drawdown_status = drawdown.check_limits(
        args.daily_loss_cap,
        args.weekly_loss_cap,
    )
    if not drawdown_status.ok:
        if not args.quiet:
            reasons = ", ".join(drawdown_status.reasons) or "drawdown cap breached"
            print(f"[drawdown] Skipping scan due to {reasons}")
        return

    outcome = scan_series(
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
        offline=offline_mode,
        sizing_mode=args.sizing,
        kelly_cap=args.kelly_cap,
        uncertainty_penalty=args.uncertainty_penalty,
        ob_imbalance_penalty=args.ob_imbalance_penalty,
        daily_loss_cap=args.daily_loss_cap,
    )

    proposals = outcome.proposals
    if fill_alpha_auto:
        outcome.monitors["fill_alpha_auto"] = fill_alpha_value
    else:
        outcome.monitors.setdefault("fill_alpha", fill_alpha_value)
    should_archive = args.report or args.paper_ledger
    orderbook_ids: set[str] = set()
    if args.report or args.paper_ledger:
        orderbook_ids.update({proposal.market_id for proposal in proposals})
    if should_archive:
        orderbook_ids.update({market.id for market in outcome.markets})

    orderbooks: dict[str, Orderbook] = {}
    for market_id in sorted(orderbook_ids):
        try:
            orderbooks[market_id] = client.get_orderbook(market_id)
        except Exception:  # pragma: no cover - tolerate missing orderbooks
            continue

    ledger = _maybe_simulate_ledger(
        args,
        proposals,
        client,
        orderbooks=orderbooks,
        fill_alpha=fill_alpha_value,
        series=outcome.series,
        events=outcome.events,
        markets=outcome.markets,
    )
    if ledger:
        drawdown.record_pnl(ledger.total_expected_pnl())
    exposure_summary = _compute_exposure_summary(proposals)
    cdf_path = _write_cdf_diffs(outcome.cdf_diffs)
    if proposals:
        _attach_series_metadata(
            proposals=proposals,
            series=args.series,
            driver_fixtures=driver_fixtures,
            offline=offline_mode,
        )
    output_path = write_proposals(
        series=args.series,
        proposals=proposals,
        output_dir=Path(args.output_dir),
    )
    if not args.quiet:
        print(f"Wrote {len(proposals)} proposals to {output_path}")

    manifest_path: Path | None = None
    if should_archive:
        manifest_path = _archive_and_replay(
            client=client,
            series=outcome.series,
            events=outcome.events,
            markets=outcome.markets,
            orderbooks=orderbooks,
            proposals_path=output_path,
            driver_fixtures=driver_fixtures,
            scanner_fixtures=fixtures_root,
        )
        if manifest_path and not args.quiet:
            print(f"Archived snapshot manifest at {manifest_path}")

    if ledger and (args.paper_ledger or args.report):
        artifacts_dir = Path("reports/_artifacts")
        ledger.write_artifacts(artifacts_dir, manifest_path=manifest_path)

    _maybe_write_report(
        args,
        proposals,
        ledger,
        outcome.monitors,
        exposure_summary,
        manifest_path,
        go_status=True,
        fill_alpha=fill_alpha_value,
    )

    if not proposals:
        return


def _build_client(fixtures_root: Path, *, use_online: bool) -> KalshiPublicClient:
    api_fixtures = fixtures_root / "kalshi"
    return KalshiPublicClient(
        offline_dir=api_fixtures,
        use_offline=not use_online,
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
    *,
    orderbooks: dict[str, Orderbook] | None = None,
    fill_alpha: float | None = None,
    series: Series | None = None,
    events: Sequence[Event] | None = None,
    markets: Sequence[Market] | None = None,
) -> PaperLedger | None:
    if not proposals or not (args.paper_ledger or args.report):
        return None
    cache = orderbooks if orderbooks is not None else {}
    for proposal in proposals:
        if proposal.market_id not in cache:
            try:
                cache[proposal.market_id] = client.get_orderbook(proposal.market_id)
            except Exception:  # pragma: no cover - tolerate missing books
                continue
    estimator = FillRatioEstimator(fill_alpha) if fill_alpha is not None else None
    event_lookup: dict[str, str] = {}
    if events is not None and markets is not None:
        event_tickers = {event.id: event.ticker for event in events}
        for market in markets:
            label = event_tickers.get(market.event_id) or market.ticker
            event_lookup[market.id] = label
    series_label = series.ticker if series is not None else args.series
    slippage_mode = (getattr(args, "slippage_mode", "top") or "top").lower()
    impact_cap_arg = getattr(args, "impact_cap", None)
    slippage_model = None
    if slippage_mode in {"top", "depth"}:
        if impact_cap_arg is not None:
            slippage_model = SlippageModel(mode=slippage_mode, impact_cap=float(impact_cap_arg))
        else:
            slippage_model = SlippageModel(mode=slippage_mode)
    elif slippage_mode != "mid":
        slippage_mode = "top"
    ledger = simulate_fills(
        proposals,
        cache,
        fill_estimator=estimator,
        ledger_series=series_label,
        market_event_lookup=event_lookup,
        mode=slippage_mode,
        slippage_model=slippage_model,
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
    monitors: dict[str, object],
    exposure_summary: dict[str, object],
    manifest_path: Path | None,
    go_status: bool | None,
    fill_alpha: float | None,
) -> None:
    if not args.report:
        return
    report_path = write_markdown_report(
        series=args.series,
        proposals=proposals,
        ledger=ledger,
        output_dir=Path("reports") / args.series.upper(),
        monitors=monitors,
        exposure_summary=exposure_summary,
        manifest_path=manifest_path,
        go_status=go_status,
        fill_alpha=fill_alpha,
    )
    if not args.quiet:
        print(f"Wrote report to {report_path}")


def _archive_and_replay(
    *,
    client: KalshiPublicClient | None,
    series: Series | None,
    events: Sequence[Event],
    markets: Sequence[Market],
    orderbooks: dict[str, Orderbook],
    proposals_path: Path,
    driver_fixtures: Path,
    scanner_fixtures: Path,
) -> Path | None:
    if series is None:
        return None
    manifest_path = archive_scan(
        series=series,
        client=client,
        events=events,
        markets=markets,
        orderbooks=orderbooks,
        out_dir=RAW_ROOT / "kalshi",
    )
    _enrich_manifest(
        manifest_path,
        proposals_path=proposals_path,
        driver_fixtures=driver_fixtures,
        scanner_fixtures=scanner_fixtures,
    )
    replay_manifest(manifest_path)
    return manifest_path


def _enrich_manifest(
    manifest_path: Path,
    *,
    proposals_path: Path,
    driver_fixtures: Path,
    scanner_fixtures: Path,
) -> None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    manifest["proposals_path"] = str(proposals_path)
    manifest["driver_fixtures"] = str(driver_fixtures)
    manifest["scanner_fixtures"] = str(scanner_fixtures)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


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
        "--online",
        action="store_true",
        help="Use live Kalshi API data (requires network).",
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
        "--fill-alpha",
        default="0.6",
        help="Fraction of visible depth expected to fill (0-1) or 'auto'.",
    )
    parser.add_argument(
        "--slippage-mode",
        default="top",
        choices=["top", "depth", "mid"],
        help="Slippage model to use for paper ledger fills.",
    )
    parser.add_argument(
        "--impact-cap",
        type=float,
        default=0.02,
        help="Maximum absolute price impact for depth slippage.",
    )
    parser.add_argument(
        "--uncertainty-penalty",
        type=float,
        default=0.0,
        help="Penalty multiplier (0-1) applied when model confidence is low.",
    )
    parser.add_argument(
        "--ob-imbalance-penalty",
        type=float,
        default=0.0,
        help="Penalty multiplier (0-1) for orderbook imbalance.",
    )
    parser.add_argument(
        "--daily-loss-cap",
        type=float,
        help="Maximum aggregate loss budget across all proposals (USD).",
    )
    parser.add_argument(
        "--weekly-loss-cap",
        type=float,
        help="Maximum aggregate weekly loss budget (USD).",
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
    uncertainty_penalty: float = 0.0,
    ob_imbalance_penalty: float = 0.0,
    daily_loss_cap: float | None = None,
) -> ScanOutcome:
    series_obj = _find_series(client, series)
    events = client.get_events(series_obj.id)
    proposals: list[Proposal] = []
    non_monotone = 0
    daily_budget = _LossBudget(daily_loss_cap)
    cdf_diffs: list[dict[str, object]] = []
    all_markets: list[Market] = []

    for event in events:
        markets = client.get_markets(event.id)
        all_markets.extend(markets)
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
            if not _is_monotone(strategy_survival):
                non_monotone += 1
            cdf_diffs.extend(
                _collect_cdf_diffs(
                    market_id=market.id,
                    market_ticker=market.ticker,
                    rungs=rungs,
                    market_survival=market_survival,
                    strategy_survival=strategy_survival,
                )
            )

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
                uncertainty_penalty=uncertainty_penalty,
                ob_imbalance_penalty=ob_imbalance_penalty,
                daily_budget=daily_budget,
            )
            proposals.extend(rung_proposals)

    monitors = {
        "non_monotone_ladders": non_monotone,
        "model_drift": _model_drift_flag(series_obj.ticker),
        "tz_not_et": _tz_not_et(),
    }
    return ScanOutcome(
        proposals=proposals,
        monitors=monitors,
        cdf_diffs=cdf_diffs,
        series=series_obj,
        events=events,
        markets=all_markets,
    )


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
    if pick in {"claims", "jobless"} and ticker == "CLAIMS":
        history = _load_history(fixtures_dir, "claims")
        claims_history = [int(item["claims"]) for item in history[-6:]] if history else None
        latest_claims = claims_history[-1] if claims_history else None
        holiday_flag = bool(history[-1].get("holiday")) if history else False
        inputs = claims_strategy.ClaimsInputs(
            history=claims_history,
            holiday_next=holiday_flag,
            freeze_active=claims_strategy.freeze_window(),
            latest_initial_claims=latest_claims,
            four_week_avg=None,
        )
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
    raise NotImplementedError(f"No strategy PMF implemented for series {series}")


def _load_history(fixtures_dir: Path, namespace: str) -> list[dict[str, object]]:
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


def _is_monotone(sequence: Sequence[float]) -> bool:
    return all(a >= b - 1e-9 for a, b in zip(sequence, sequence[1:], strict=False))


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
    uncertainty_penalty: float,
    ob_imbalance_penalty: float,
    daily_budget: _LossBudget,
) -> list[Proposal]:
    proposals: list[Proposal] = []
    uncertainty_penalty = max(0.0, uncertainty_penalty)
    ob_imbalance_penalty = max(0.0, ob_imbalance_penalty)

    for index, rung in enumerate(rungs):
        if allowed_indices is not None and index not in allowed_indices:
            continue
        yes_price = rung.yes_price
        event_probability = strategy_survival[index]
        survival_market = market_survival[index]
        raw_fraction: float | None = None
        truncated_fraction: float | None = None
        scaled_fraction: float | None = None
        uncertainty_metric = max(0.0, min(1.0, 1.0 - abs(event_probability - 0.5) * 2.0))
        imbalance_metric = max(0.0, min(1.0, abs(survival_market - 0.5) * 2.0))

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
        strike_cap = pal_guard.policy.limit_for_strike(order_id)
        remaining_limit = strike_cap - pal_guard.exposure_for(order_id)
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
            truncated_fraction = truncate_kelly(raw_fraction, kelly_cap)
            scaled_fraction = scale_kelly(
                truncated_fraction,
                uncertainty_metric * uncertainty_penalty,
                imbalance_metric * ob_imbalance_penalty,
                kelly_cap,
            )
            if scaled_fraction <= 0.0:
                continue
            capital_base = pal_guard.policy.default_max_loss
            if capital_base is None or capital_base <= 0:
                capital_base = remaining_limit
            if capital_base is None or capital_base <= 0:
                continue
            raw_risk = capital_base * scaled_fraction
            var_remaining = None
            if risk_manager and max_var is not None:
                var_remaining = max(max_var - risk_manager.current_var(), 0.0)
            capped_risk = apply_caps(
                raw_risk,
                pal=remaining_limit,
                max_loss_per_strike=strike_cap,
                max_var=var_remaining,
            )
            if capped_risk <= 0.0:
                continue
            desired_contracts = int(capped_risk // max_loss_single)
            contract_count = min(max_contracts, desired_contracts)
            if contract_count <= 0:
                continue

        budget_before = daily_budget.remaining
        total_max_loss = max_loss_single * contract_count
        if total_max_loss <= 0:
            continue
        allowed_contracts = daily_budget.max_contracts(max_loss_single, contract_count)
        if allowed_contracts <= 0:
            continue
        if allowed_contracts < contract_count:
            contract_count = allowed_contracts
            total_max_loss = max_loss_single * contract_count

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

        daily_budget.consume(total_max_loss)
        budget_after = daily_budget.remaining

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
            metadata={
                "sizing": {
                    "kelly_raw": raw_fraction if sizing_mode == "kelly" else None,
                    "kelly_truncated": truncated_fraction if sizing_mode == "kelly" else None,
                    "kelly_scaled": scaled_fraction if sizing_mode == "kelly" else None,
                    "uncertainty_metric": uncertainty_metric,
                    "uncertainty_penalty": uncertainty_penalty,
                    "ob_imbalance_metric": imbalance_metric,
                    "ob_imbalance_penalty": ob_imbalance_penalty,
                    "daily_loss_before": budget_before,
                    "daily_loss_after": budget_after,
                }
            },
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


def _collect_cdf_diffs(
    *,
    market_id: str,
    market_ticker: str,
    rungs: Sequence[LadderRung],
    market_survival: Sequence[float],
    strategy_survival: Sequence[float],
) -> list[dict[str, object]]:
    diffs: list[dict[str, object]] = []
    for idx, rung in enumerate(rungs):
        p_model = float(strategy_survival[idx])
        p_market = float(market_survival[idx])
        diffs.append(
            {
                "market_id": market_id,
                "market_ticker": market_ticker,
                "bin_index": idx,
                "strike": float(rung.strike),
                "p_model": p_model,
                "p_market": p_market,
                "delta": p_model - p_market,
            }
        )
    return diffs


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


def _write_cdf_diffs(diffs: Sequence[dict[str, object]]) -> Path | None:
    if not diffs:
        return None
    frame = pl.DataFrame(diffs)
    artifacts_dir = Path("reports/_artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / "cdf_diffs.parquet"
    frame.write_parquet(path)
    return path


def _load_portfolio_config() -> PortfolioConfig | None:
    global _PORTFOLIO_CONFIG_CACHE
    if _PORTFOLIO_CONFIG_CACHE is not None:
        return _PORTFOLIO_CONFIG_CACHE
    path = Path("configs/portfolio.yaml")
    if not path.exists():
        return None
    try:
        _PORTFOLIO_CONFIG_CACHE = PortfolioConfig.from_yaml(path)
    except Exception:  # pragma: no cover - config parse errors fall back to None
        _PORTFOLIO_CONFIG_CACHE = None
    return _PORTFOLIO_CONFIG_CACHE


def _compute_exposure_summary(proposals: Sequence[Proposal]) -> dict[str, object]:
    summary: dict[str, object] = {
        "total_max_loss": 0.0,
        "per_series": {},
        "net_contracts": {},
        "factors": {},
        "var": 0.0,
        "series_factors": {},
        "series_net": {},
    }
    if not proposals:
        return summary

    total_max_loss = sum(float(proposal.max_loss) for proposal in proposals)
    per_series: dict[str, float] = defaultdict(float)
    net_contracts: dict[str, int] = defaultdict(int)
    market_losses: dict[str, float] = defaultdict(float)
    market_series: dict[str, str] = {}
    for proposal in proposals:
        per_series[proposal.strategy] += float(proposal.max_loss)
        sign = 1 if proposal.side.upper() == "YES" else -1
        net_contracts[proposal.market_ticker] += sign * proposal.contracts
        market_losses[proposal.market_ticker] += float(proposal.max_loss)
        market_series.setdefault(proposal.market_ticker, proposal.strategy)

    summary["total_max_loss"] = total_max_loss
    summary["per_series"] = dict(sorted(per_series.items()))
    summary["net_contracts"] = dict(sorted(net_contracts.items()))
    summary["market_losses"] = dict(sorted(market_losses.items()))
    summary["market_series"] = market_series

    config = _load_portfolio_config()
    if config is not None:
        factor_exposures: dict[str, float] = defaultdict(float)
        series_factor_exposures: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for proposal in proposals:
            betas = config.strategy_betas.get(proposal.strategy.upper(), {"TOTAL": 1.0})
            for factor, beta in betas.items():
                exposure = beta * float(proposal.max_loss)
                factor_exposures[factor] += exposure
                series_factor_exposures[proposal.strategy][factor] += exposure
        summary["factors"] = dict(sorted(factor_exposures.items()))
        var_sum = 0.0
        for factor, exposure in factor_exposures.items():
            vol = config.factor_vols.get(factor, 1.0)
            var_sum += (exposure * vol) ** 2
        summary["var"] = math.sqrt(var_sum)
        summary["series_factors"] = {
            series: dict(sorted(factors.items())) for series, factors in series_factor_exposures.items()
        }
    else:
        summary["factors"] = {"TOTAL": total_max_loss}
        summary["var"] = total_max_loss
        summary["series_factors"] = {series: {"TOTAL": value} for series, value in per_series.items()}

    series_net: dict[str, dict[str, int]] = defaultdict(lambda: {"long": 0, "short": 0})
    for market, value in net_contracts.items():
        series_name = market_series.get(market)
        if series_name is None:
            continue
        if value >= 0:
            series_net[series_name]["long"] += int(value)
        else:
            series_net[series_name]["short"] += int(-value)
    summary["series_net"] = {series: data for series, data in series_net.items()}

    return summary


def _find_series(client: KalshiPublicClient, ticker: str) -> Series:
    series_list = client.get_series()
    for series in series_list:
        if series.ticker.upper() == ticker.upper():
            return series
    raise ValueError(f"Series {ticker} not found in fixtures")


def _model_drift_flag(series_ticker: str) -> bool:
    path_map = {
        "CPI": PROC_ROOT / "cpi_calib.parquet",
        "CLAIMS": PROC_ROOT / "claims_calib.parquet",
        "TNEY": PROC_ROOT / "teny_calib.parquet",
        "WEATHER": PROC_ROOT / "weather_calib.parquet",
    }
    path = path_map.get(series_ticker.upper())
    if path is None or not path.exists():
        return False
    frame = pl.read_parquet(path)
    summary = frame.filter(pl.col("record_type") == "params")
    if summary.is_empty():
        return False
    row = summary.row(0, named=True)
    crps = row.get("crps")
    baseline_crps = row.get("baseline_crps")
    if crps is not None and baseline_crps is not None:
        if float(crps) > float(baseline_crps) * 1.1:
            return True
    brier = row.get("brier")
    baseline_brier = row.get("baseline_brier")
    if brier is not None and baseline_brier is not None:
        if float(brier) > float(baseline_brier) * 1.1:
            return True
    return False


def _tz_not_et() -> bool:
    now_et = datetime.now(tz=ZoneInfo("America/New_York"))
    return getattr(now_et.tzinfo, "key", "") != "America/New_York"


if __name__ == "__main__":
    main()
