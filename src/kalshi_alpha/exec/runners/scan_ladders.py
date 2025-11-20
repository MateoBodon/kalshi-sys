"""CLI scanner that produces dry-run order proposals for Kalshi ladders."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

import kalshi_alpha.core.fees as fee_utils
from kalshi_alpha.brokers import create_broker
from kalshi_alpha.brokers.kalshi.base import BrokerOrder
from kalshi_alpha.config import IndexOpsConfig, IndexOpsWindow, IndexRule, load_index_ops_config, lookup_index_rule
from kalshi_alpha.core import kalshi_ws
from kalshi_alpha.core.archive import archive_scan, replay_manifest
from kalshi_alpha.core.execution import fillprob
from kalshi_alpha.core.execution.fillratio import FillRatioEstimator, load_alpha, tune_alpha
from kalshi_alpha.core.execution.index_models import load_alpha_curve, load_slippage_curve
from kalshi_alpha.core.execution.slippage import SlippageModel, load_slippage_model
from kalshi_alpha.core.fees import DEFAULT_FEE_SCHEDULE
from kalshi_alpha.core.fees import index_series as index_fee_utils
from kalshi_alpha.core.gates import QualityGateResult, load_quality_gate_config, run_quality_gates
from kalshi_alpha.core.kalshi_api import Event, KalshiPublicClient, Market, Orderbook, Series
from kalshi_alpha.core.pricing import (
    LadderBinProbability,
    LadderRung,
    Liquidity,
    OrderSide,
    implied_cdf_kinks,
    kink_spreads,
    pmf_from_quotes,
    prob_sum_gap,
)
from kalshi_alpha.core.pricing.align import align_pmf_to_strikes
from kalshi_alpha.core.risk import (
    OrderProposal,
    PALGuard,
    PALPolicy,
    PortfolioConfig,
    PortfolioRiskManager,
    drawdown,
    max_loss_for_order,
)
from kalshi_alpha.core.sizing import apply_caps, kelly_yes_no, scale_kelly, truncate_kelly
from kalshi_alpha.datastore.paths import PROC_ROOT, RAW_ROOT
from kalshi_alpha.drivers import macro_calendar
from kalshi_alpha.drivers.aaa_gas import fetch as aaa_fetch
from kalshi_alpha.drivers.aaa_gas import ingest as aaa_ingest
from kalshi_alpha.drivers.calendar.loader import calendar_tags_for
from kalshi_alpha.drivers.polygon_index.client import IndexSnapshot, PolygonAPIError, PolygonIndicesClient
from kalshi_alpha.drivers.polygon_index.symbols import resolve_series as resolve_index_series
from kalshi_alpha.exec import fees as exec_fees
from kalshi_alpha.exec import quote_microprice
from kalshi_alpha.data import WSFreshnessSentry
from kalshi_alpha.exec.limits import LossBudget, ProposalLimitChecker, LimitViolation
from kalshi_alpha.exec.gate_utils import resolve_quality_gate_config_path, write_go_no_go
from kalshi_alpha.exec.heartbeat import (
    heartbeat_stale,
    kill_switch_engaged,
    resolve_kill_switch_path,
    write_heartbeat,
)
from kalshi_alpha.exec.ledger import ExecutionMetrics, PaperLedger, simulate_fills
from kalshi_alpha.exec.monitors import fee_rules, sigma_drift
from kalshi_alpha.exec.monitors.freshness import FRESHNESS_ARTIFACT_PATH
from kalshi_alpha.exec.monitors.freshness import load_artifact as load_freshness_artifact
from kalshi_alpha.exec.monitors.freshness import summarize_artifact as summarize_freshness_artifact
from kalshi_alpha.exec.monitors.summary import (
    DEFAULT_MONITOR_MAX_AGE_MINUTES,
    DEFAULT_PANIC_ALERT_THRESHOLD,
    DEFAULT_PANIC_ALERT_WINDOW_MINUTES,
    MONITOR_ARTIFACTS_DIR,
    summarize_monitor_artifacts,
)
from kalshi_alpha.exec.quote_optim import QuoteContext, QuoteOptimizer
from kalshi_alpha.exec.pilot import (
    PilotConfig,
    PilotSession,
    apply_pilot_mode,
    write_pilot_session_artifact,
)
from kalshi_alpha.exec.reports import write_markdown_report
from kalshi_alpha.exec.scanners import cpi
from kalshi_alpha.exec.scanners.utils import expected_value_summary, pmf_to_survival
from kalshi_alpha.exec.state.orders import OutstandingOrdersState
from kalshi_alpha.markets.discovery import discover_markets_for_day
from kalshi_alpha.structures import build_range_structures
from kalshi_alpha.sched import TradingWindow, current_window as scheduler_current_window, regimes
from kalshi_alpha.strategies import claims as claims_strategy
from kalshi_alpha.strategies import cpi as cpi_strategy
from kalshi_alpha.strategies import index as index_strategy
from kalshi_alpha.strategies import teny as teny_strategy
from kalshi_alpha.strategies import weather as weather_strategy
from kalshi_alpha.risk import CORRELATION_CONFIG_PATH, CorrelationAwareLimiter
from kalshi_alpha.risk import var_index
from kalshi_alpha.utils.env import load_env

INDEX_OPS_CONFIG: IndexOpsConfig = load_index_ops_config()

DEFAULT_MIN_EV = float(INDEX_OPS_CONFIG.min_ev_usd)
DEFAULT_CONTRACTS = 10
DEFAULT_FILL_ALPHA = 0.6

_POLYGON_FIXTURE_DIR = "polygon_index"
_WINDOW_HOURLY = INDEX_OPS_CONFIG.window_hourly
_WINDOW_CLOSE = INDEX_OPS_CONFIG.window_close
_TARGET_CLOSE = _WINDOW_CLOSE.end
_ET_ZONE = ZoneInfo("America/New_York")
_U_SERIES = {"INXU", "NASDAQ100U"}
_INDEX_WS_SERIES = frozenset({"INX", "INXU", "NASDAQ100", "NASDAQ100U"})
_HOUR_PATTERN = re.compile(r"H(?P<hour>\d{2})(?P<minute>\d{2})")
CLOCK_SKEW_THRESHOLD_SECONDS = 1.5
HONESTY_CLAMP_PATH = Path("reports/_artifacts/honesty/honesty_clamp.json")


def _load_data_freshness_summary() -> dict[str, object]:
    freshness_path = MONITOR_ARTIFACTS_DIR / FRESHNESS_ARTIFACT_PATH.name
    payload = load_freshness_artifact(freshness_path)
    return summarize_freshness_artifact(payload, artifact_path=freshness_path)


def _safe_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _resolve_fee_paths(series: str | None) -> dict[str, str | None]:
    override_path = fee_utils.FEE_OVERRIDE_PATH
    config_path = fee_utils.FEE_CONFIG_PATH
    taker_source = override_path if override_path.exists() else config_path

    maker_source: Path | None = None
    series_upper = series.upper() if isinstance(series, str) else None
    if series_upper and series_upper.startswith(tuple(fee_utils.INDEX_PREFIXES)):
        candidate = exec_fees.DEFAULT_FEES_PATH
        if not candidate.exists():
            candidate = index_fee_utils.DEFAULT_INDEX_FEE_PATH.resolve()
        if not candidate.exists() and index_fee_utils.LEGACY_INDEX_FEE_PATH.exists():
            candidate = index_fee_utils.LEGACY_INDEX_FEE_PATH.resolve()
        if candidate.exists():
            maker_source = candidate
    if maker_source is None:
        maker_source = taker_source

    return {
        "maker": maker_source.resolve().as_posix(),
        "taker": taker_source.resolve().as_posix(),
    }


def _polygon_ws_freshness_detail(
    *,
    gate_details: Mapping[str, object] | None,
    data_summary: Mapping[str, object] | None,
    fatal_reason: str | None,
) -> dict[str, object]:
    age_hours = None
    issues: set[str] = set()
    if isinstance(gate_details, Mapping):
        raw_age = gate_details.get("data_freshness.polygon_ws.age_hours")
        age_hours = _safe_float(raw_age)
        for key, value in gate_details.items():
            if not isinstance(key, str):
                continue
            if not key.startswith("data_freshness.polygon_ws."):
                continue
            suffix = key.split("data_freshness.polygon_ws.", 1)[-1]
            if suffix == "age_hours":
                continue
            issues.add(f"{suffix}={value}")

    ok = fatal_reason not in {"polygon_ws_stale", "freshness_missing", "data_freshness_no_go"}
    if issues:
        ok = False

    age_seconds = float(age_hours) * 3600.0 if age_hours is not None else None
    artifact_status = None
    artifact_generated = None
    artifact_path = None
    if isinstance(data_summary, Mapping):
        artifact_status = data_summary.get("status")
        artifact_generated = data_summary.get("generated_at")
        artifact_path = data_summary.get("artifact_path")

    detail: dict[str, object] = {
        "ok": ok,
        "fatal_reason": fatal_reason,
        "age_hours": age_hours,
        "age_seconds": age_seconds,
        "artifact_status": artifact_status,
        "artifact_generated_at": artifact_generated,
        "artifact_path": artifact_path,
    }
    if issues:
        detail["issues"] = tuple(sorted(issues))
    return detail


def _polygon_ws_age_ms(summary: Mapping[str, object] | None) -> float | None:
    if not isinstance(summary, Mapping):
        return None
    feeds = summary.get("feeds")
    if not isinstance(feeds, list):
        return None
    for feed in feeds:
        if not isinstance(feed, Mapping):
            continue
        feed_id = str(feed.get("id") or "").strip().lower()
        if feed_id != "polygon_index.websocket":
            continue
        details = feed.get("details") if isinstance(feed.get("details"), Mapping) else {}
        if isinstance(details, Mapping):
            age_seconds = details.get("age_seconds")
            if isinstance(age_seconds, (int, float)):
                return float(age_seconds) * 1000.0
        age_minutes = feed.get("age_minutes")
        if isinstance(age_minutes, (int, float)):
            return float(age_minutes) * 60_000.0
    return None


def _freshness_fatal_reason(
    summary: Mapping[str, object] | None,
    *,
    require_polygon_ws: bool,
) -> str | None:
    if not require_polygon_ws:
        return None
    if summary is None:
        return "freshness_missing"

    status = str(summary.get("status") or "").upper()
    required_ok = bool(summary.get("required_feeds_ok", True))
    if status == "MISSING" and not required_ok:
        return "freshness_missing"
    if required_ok:
        return None

    stale_feeds = summary.get("stale_feeds") or []
    stale_normalized = {str(feed).strip().lower() for feed in stale_feeds if isinstance(feed, str)}
    if "polygon_index.websocket" in stale_normalized:
        return "polygon_ws_stale"
    return "data_freshness_no_go"


def _resolve_fill_alpha_arg(fill_alpha_arg: object, series: str) -> tuple[float, bool]:  # noqa: PLR0912
    stored = load_alpha(series)
    auto = False
    candidate: float | None = None

    if fill_alpha_arg is None:
        if stored is not None:
            candidate = stored
            auto = True
        else:
            candidate = DEFAULT_FILL_ALPHA
    elif isinstance(fill_alpha_arg, str):
        value = fill_alpha_arg.strip().lower()
        if value == "auto":
            tuned = tune_alpha(series, RAW_ROOT / "kalshi")
            if tuned is not None:
                candidate = float(tuned)
            elif stored is not None:
                candidate = stored
            else:
                candidate = DEFAULT_FILL_ALPHA
            auto = True
        else:
            try:
                candidate = float(value)
            except ValueError:
                if stored is not None:
                    candidate = stored
                    auto = True
                else:
                    candidate = DEFAULT_FILL_ALPHA
    else:
        try:
            candidate = float(fill_alpha_arg)
        except (TypeError, ValueError):
            if stored is not None:
                candidate = stored
                auto = True
            else:
                candidate = DEFAULT_FILL_ALPHA

    if candidate is None:
        candidate = DEFAULT_FILL_ALPHA
    adjusted = fillprob.adjust_alpha(series, float(candidate))
    return adjusted, auto


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
    series: str
    metadata: dict[str, object] | None = None


@dataclass
class ScanOutcome:
    proposals: list[Proposal]
    monitors: dict[str, object] = field(default_factory=dict)
    cdf_diffs: list[dict[str, object]] = field(default_factory=list)
    mispricings: list[dict[str, object]] = field(default_factory=list)
    series: Series | None = None
    events: list[Event] = field(default_factory=list)
    markets: list[Market] = field(default_factory=list)
    model_metadata: dict[str, object] = field(default_factory=dict)
    books_at_scan: dict[str, Orderbook] = field(default_factory=dict)
    book_snapshot_started_at: datetime | None = None
    book_snapshot_completed_at: datetime | None = None
    roll_info: dict[str, object] | None = None
    execution_metrics: ExecutionMetrics | None = None


@dataclass(frozen=True)
class BinConstraintEntry:
    """Represents a per-bin EV honesty constraint sourced from readiness outputs."""

    series: str
    market_id: str | None
    market_ticker: str | None
    strike: float | None
    side: str
    weight: float | None
    cap: int | None
    sources: tuple[str, ...] = ()


class BinConstraintResolver:
    """Lookup helper that applies per-bin contract caps/weights."""

    def __init__(self, entries: Sequence[BinConstraintEntry], *, source_path: Path | None = None) -> None:
        self._entries: tuple[BinConstraintEntry, ...] = tuple(entries)
        self._by_market_id: dict[tuple[str, str], BinConstraintEntry] = {}
        self._by_ticker: dict[tuple[str, float, str], BinConstraintEntry] = {}
        for entry in self._entries:
            normalized_side = entry.side.upper()
            if entry.market_id:
                self._by_market_id[(entry.market_id, normalized_side)] = entry
            if entry.market_ticker and entry.strike is not None:
                key = (entry.market_ticker.upper(), round(float(entry.strike), 4), normalized_side)
                self._by_ticker[key] = entry
        self._source_hits: Counter[str] = Counter()
        self.applied: int = 0
        self.dropped: int = 0
        self.source_path = source_path

    @property
    def has_rules(self) -> bool:
        return bool(self._entries)

    def _record_sources(self, entry: BinConstraintEntry) -> None:
        if entry.sources:
            self._source_hits.update(entry.sources)
        else:
            self._source_hits.update(["unspecified"])

    def _resolve(
        self,
        *,
        market_id: str | None,
        market_ticker: str | None,
        strike: float | None,
        side: str,
    ) -> BinConstraintEntry | None:
        normalized_side = side.upper()
        if market_id:
            entry = self._by_market_id.get((market_id, normalized_side))
            if entry is not None:
                return entry
        if market_ticker is not None and strike is not None:
            key = (market_ticker.upper(), round(float(strike), 4), normalized_side)
            entry = self._by_ticker.get(key)
            if entry is not None:
                return entry
        return None

    def apply(
        self,
        *,
        market_id: str,
        market_ticker: str,
        strike: float,
        side: str,
        contracts: int,
    ) -> tuple[int, dict[str, object] | None]:
        if contracts <= 0:
            return contracts, None
        entry = self._resolve(
            market_id=market_id,
            market_ticker=market_ticker,
            strike=strike,
            side=side,
        )
        if entry is None:
            return contracts, None

        original_contracts = contracts
        new_contracts = contracts
        details: dict[str, object] = {
            "strike": entry.strike,
            "side": entry.side,
        }
        if entry.market_id:
            details["market_id"] = entry.market_id
        if entry.market_ticker:
            details["market_ticker"] = entry.market_ticker
        if entry.sources:
            details["sources"] = list(entry.sources)

        if entry.weight is not None:
            scaled = max(0, int(math.floor(original_contracts * entry.weight + 1e-9)))
            details["recommended_weight"] = entry.weight
            details["weight_contracts"] = scaled
            new_contracts = min(new_contracts, scaled)
        if entry.cap is not None:
            cap_value = max(0, int(entry.cap))
            details["recommended_cap"] = cap_value
            new_contracts = min(new_contracts, cap_value)

        changed = new_contracts != original_contracts
        if changed:
            details["original_contracts"] = original_contracts
            details["adjusted_contracts"] = new_contracts
            self.applied += 1
            self._record_sources(entry)
        if new_contracts <= 0:
            self.dropped += 1
            details["adjusted_contracts"] = 0
            return 0, details
        return new_contracts, (details if changed else None)

    def summary(self) -> dict[str, object]:
        data: dict[str, object] = {
            "rules": len(self._entries),
            "applied": self.applied,
            "dropped": self.dropped,
        }
        if self._source_hits:
            data["source_hits"] = dict(self._source_hits)
        if self.source_path is not None:
            data["source_path"] = self.source_path.as_posix()
        return data


def _load_ev_honesty_constraints(series: str, readiness_path: Path | None = None) -> BinConstraintResolver | None:
    path = Path(readiness_path) if readiness_path is not None else Path("reports/pilot_ready.json")
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive
        return None
    series_entries = payload.get("series")
    if not isinstance(series_entries, list):
        return None
    normalized_series = series.strip().upper()
    entries: list[BinConstraintEntry] = []
    for record in series_entries:
        if not isinstance(record, dict):
            continue
        record_series = str(record.get("series") or "").upper()
        if record_series != normalized_series:
            continue
        bins = record.get("ev_honesty_bins")
        if not isinstance(bins, list):
            continue
        for bin_entry in bins:
            if not isinstance(bin_entry, dict):
                continue
            weight_raw = bin_entry.get("recommended_weight")
            weight_val: float | None
            if isinstance(weight_raw, (int, float)) and math.isfinite(float(weight_raw)):
                weight_val = max(0.0, min(1.0, float(weight_raw)))
            else:
                weight_val = None
            cap_raw = bin_entry.get("recommended_cap")
            cap_val: int | None
            if isinstance(cap_raw, (int, float)) and math.isfinite(float(cap_raw)):
                cap_val = max(0, int(math.floor(float(cap_raw))))
            else:
                cap_val = None
            if weight_val is None and cap_val is None:
                continue
            strike_val: float | None
            try:
                strike_candidate = bin_entry.get("strike")
                strike_val = float(strike_candidate) if strike_candidate is not None else None
            except (TypeError, ValueError):  # pragma: no cover - defensive
                strike_val = None
            side_raw = bin_entry.get("side")
            side = str(side_raw).upper() if isinstance(side_raw, str) else "YES"
            market_id_raw = bin_entry.get("market_id")
            market_id = str(market_id_raw) if market_id_raw is not None else None
            market_ticker_raw = bin_entry.get("market_ticker")
            market_ticker = str(market_ticker_raw) if market_ticker_raw is not None else None
            sources_value = bin_entry.get("sources")
            sources: tuple[str, ...]
            if isinstance(sources_value, list):
                sources = tuple(str(item) for item in sources_value if isinstance(item, str) and item)
            else:
                sources = tuple()
            entries.append(
                BinConstraintEntry(
                    series=normalized_series,
                    market_id=market_id,
                    market_ticker=market_ticker,
                    strike=strike_val,
                    side=side,
                    weight=weight_val,
                    cap=cap_val,
                    sources=sources,
                )
            )
    if not entries:
        return None
    return BinConstraintResolver(entries, source_path=path)


def _load_honesty_clamp(series: str) -> float | None:
    if not HONESTY_CLAMP_PATH.exists():
        return None
    try:
        payload = json.loads(HONESTY_CLAMP_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    series_payload = payload.get("series")
    if not isinstance(series_payload, dict):
        return None
    entry = series_payload.get(series.upper())
    if not isinstance(entry, dict):
        return None
    clamp = entry.get("clamp")
    try:
        value = float(clamp)
    except (TypeError, ValueError):
        return None
    if value <= 0.0 or value > 1.0:
        return None
    return value


_PORTFOLIO_CONFIG_CACHE: PortfolioConfig | None = None


def _clear_dry_orders_start(
    *,
    enabled: bool,
    broker_mode: str,
    quiet: bool,
    state: OutstandingOrdersState | None = None,
) -> dict[str, int]:
    """Optionally clear outstanding DRY orders before generating proposals."""

    state_obj = state if state is not None else OutstandingOrdersState.load()
    normalized = (broker_mode or "dry").strip().lower()
    removed = 0
    if enabled and normalized == "dry":
        dry_orders = state_obj.outstanding_for("dry")
        dry_keys = list(dry_orders.keys())
        if dry_keys:
            removed_keys = state_obj.remove("dry", dry_keys)
            removed = len(removed_keys)
            if removed:
                state_obj.clear_cancel_all()
        if not quiet:
            print(f"[orders] Cleared dry orders at start: {removed}")

    summary = state_obj.summary()
    total = sum(summary.values())
    if not quiet:
        breakdown = ", ".join(f"{mode}={count}" for mode, count in sorted(summary.items()))
        print(f"Outstanding orders: {total} ({breakdown})")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    load_env()
    args = parse_args(argv)
    if args.online and args.offline:
        raise ValueError("Cannot specify both --online and --offline.")
    if args.today and not args.discover:
        raise ValueError("--today is only supported with --discover.")
    if args.discover:
        _run_discovery(args)
        return
    if not args.series:
        raise ValueError("--series is required unless --discover is specified.")
    try:
        exec_fees.load_fee_config()
    except FileNotFoundError as exc:  # pragma: no cover - fails fast in live mode
        raise RuntimeError(
            "Index fee configuration missing (expected configs/fees.json); refusing to run."
        ) from exc
    except ValueError as exc:  # pragma: no cover - config validation
        raise RuntimeError(f"Invalid fee configuration: {exc}") from exc
    pilot_session: PilotSession | None = apply_pilot_mode(args)
    pilot_config: PilotConfig | None = pilot_session.config if pilot_session else None
    bin_constraints = _load_ev_honesty_constraints(args.series) if pilot_session else None
    fixtures_root = Path(args.fixtures_root).resolve()
    driver_fixtures = fixtures_root / "drivers"

    client = _build_client(fixtures_root, use_online=args.online)
    fill_alpha_value, fill_alpha_auto = _resolve_fill_alpha_arg(args.fill_alpha, args.series)
    pal_guard = _build_pal_guard(args)
    risk_manager = _build_risk_manager(args)
    var_limiter = var_index.FamilyVarLimiter(var_index.load_family_limits())
    correlation_config_path = Path(args.correlation_config) if args.correlation_config else CORRELATION_CONFIG_PATH
    correlation_guard = CorrelationAwareLimiter.from_yaml(correlation_config_path)
    quote_optimizer = QuoteOptimizer()
    offline_mode = args.offline or not args.online
    series_upper = args.series.upper()
    honesty_clamp = _load_honesty_clamp(series_upper)
    if honesty_clamp is not None:
        args.ev_honesty_shrink = min(float(args.ev_honesty_shrink), honesty_clamp)

    outstanding_start = _clear_dry_orders_start(
        enabled=getattr(args, "clear_dry_orders_start", False),
        broker_mode=args.broker,
        quiet=args.quiet,
    )

    drawdown_status = drawdown.check_limits(
        args.daily_loss_cap,
        args.weekly_loss_cap,
    )
    if not drawdown_status.ok:
        if not args.quiet:
            reasons = ", ".join(drawdown_status.reasons) or "drawdown cap breached"
            print(f"[drawdown] Skipping scan due to {reasons}")
        return

    skip_fee_rules = offline_mode or str(args.broker or "").lower() != "live"
    fee_ready, fee_reason = (True, None) if skip_fee_rules else fee_rules.is_ready()
    if not fee_ready:
        raise RuntimeError(f"Fee/rule watcher pending: {fee_reason or 'unknown'}")

    now_override = datetime.now(tz=UTC)
    clock_skew_seconds = _clock_skew_seconds(now_override)
    cancel_requested = False
    if args.series.upper() in _U_SERIES:
        roll_decision = _u_series_roll_decision(now_override)
        if roll_decision.get("cancel_required"):
            state = OutstandingOrdersState.load()
            state.mark_cancel_all("u_series_hourly_roll", modes=[args.broker])
            cancel_requested = True
            if not args.quiet:
                target_label = _format_hour_label(int(roll_decision.get("target_hour", 0)))
                print(f"[u-roll] Cancel-all requested for {args.series.upper()} ahead of {target_label}")
    else:
        roll_decision = None
    scheduler_window = scheduler_current_window(series_upper, now_override)
    if scheduler_window and scheduler_window.seconds_to_freeze(now_override) <= 0 and not cancel_requested:
        state = OutstandingOrdersState.load()
        state.mark_cancel_all("scheduler_t_minus_2s", modes=[args.broker])
        cancel_requested = True
        if not args.quiet:
            print(
                f"[scheduler] Cancel-all requested for {series_upper} ahead of {scheduler_window.label}"
            )

    data_freshness_summary = _load_data_freshness_summary()
    fatal_freshness_reason = _freshness_fatal_reason(
        data_freshness_summary,
        require_polygon_ws=args.series.upper() in _INDEX_WS_SERIES,
    )
    ws_age_ms = _polygon_ws_age_ms(data_freshness_summary)
    ws_sentry = WSFreshnessSentry()
    now_for_freshness = datetime.now(tz=UTC)
    if ws_age_ms is not None:
        ws_sentry.record_latency(ws_age_ms, now=now_for_freshness)
    active_window = scheduler_current_window(series_upper, now_for_freshness)
    strict_final_minute = bool(active_window and active_window.in_final_minute(now_for_freshness))
    final_minute_reason: str | None = None
    if strict_final_minute:
        freshness_ok = ws_sentry.is_fresh(strict=True, now=now_for_freshness) if ws_age_ms is not None else False
        if not freshness_ok:
            final_minute_reason = "polygon_ws_final_minute_stale"
            fatal_freshness_reason = fatal_freshness_reason or final_minute_reason

    target_time = None
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
        ev_honesty_shrink=args.ev_honesty_shrink,
        daily_loss_cap=args.daily_loss_cap,
        weekly_loss_cap=args.weekly_loss_cap,
        mispricing_only=args.mispricing_only,
        max_legs=args.max_legs,
        prob_sum_gap_threshold=args.prob_sum_gap_threshold,
        pilot_config=pilot_config,
        bin_constraints=bin_constraints,
        now_override=now_override,
        target_time_override=target_time,
        var_limiter=var_limiter,
        correlation_guard=correlation_guard,
        quote_optimizer=quote_optimizer,
        freshness_ms=ws_age_ms,
        sniper=getattr(args, "sniper", False),
        sniper_threshold=float(getattr(args, "sniper_threshold", 0.05)),
    )

    if pilot_session:
        outcome.monitors.setdefault("pilot_session_id", pilot_session.session_id)
        outcome.monitors.setdefault("pilot_session_started_at", pilot_session.started_at.isoformat())
    if scheduler_window:
        payload = _scheduler_window_payload(scheduler_window)
        if payload:
            outcome.monitors.setdefault("scheduler_window", payload)

    outcome.monitors.setdefault("clock_skew_seconds", round(float(clock_skew_seconds), 6))
    if clock_skew_seconds > CLOCK_SKEW_THRESHOLD_SECONDS:
        outcome.monitors.setdefault("clock_skew_exceeded", True)

    if (
        outcome.roll_info
        and outcome.series
        and outcome.series.ticker.upper() in _U_SERIES
        and bool(outcome.roll_info.get("rolled"))
    ):
        if not args.quiet:
            current_label = str(outcome.roll_info.get("current_hour_label") or "")
            target_label = str(outcome.roll_info.get("target_hour_label") or "")
            print(f"[u-roll] {current_label}→{target_label}")
    elif cancel_requested and not args.quiet and outcome.series and outcome.series.ticker.upper() in _U_SERIES:
        # Ensure we at least log current targeting if cancel triggered but no roll occurred
        current_label = str(outcome.roll_info.get("current_hour_label") if outcome.roll_info else "")
        target_label = str(outcome.roll_info.get("target_hour_label") if outcome.roll_info else "")
        if current_label or target_label:
            print(f"[u-roll] {current_label or '?'}→{target_label or '?'}")

    outcome.monitors.setdefault("data_freshness", data_freshness_summary)
    if ws_age_ms is not None:
        outcome.monitors.setdefault("ws_freshness_age_ms", ws_age_ms)
    outcome.monitors.setdefault(
        "ws_final_minute_guard",
        {
            "strict": strict_final_minute,
            "window_label": active_window.label if active_window else None,
            "age_ms": ws_age_ms,
        },
    )

    proposals = list(outcome.proposals)
    cancel_reason: str | None = None
    if fatal_freshness_reason:
        outcome.monitors["fatal_data_freshness"] = {
            "reason": fatal_freshness_reason,
            "status": data_freshness_summary.get("status"),
            "stale_feeds": tuple(data_freshness_summary.get("stale_feeds", [])),
        }
        if final_minute_reason and fatal_freshness_reason == final_minute_reason:
            outcome.monitors["final_minute_freeze"] = {
                "reason": final_minute_reason,
                "age_ms": ws_age_ms,
            }
        if not args.quiet:
            print(
                "[freshness] fatal data freshness "
                f"({fatal_freshness_reason}); cancelling outstanding quotes"
            )
        proposals = []
        cancel_reason = fatal_freshness_reason
    outcome.proposals = proposals

    books_at_scan = dict(getattr(outcome, "books_at_scan", {}))
    book_snapshot_started_at = getattr(outcome, "book_snapshot_started_at", None)
    book_snapshot_completed_at = getattr(outcome, "book_snapshot_completed_at", None)
    outcome.monitors.setdefault("orderbook_snapshots", len(books_at_scan))
    if book_snapshot_started_at is not None:
        outcome.monitors.setdefault(
            "book_snapshot_started_at",
            book_snapshot_started_at.isoformat(),
        )
    if book_snapshot_completed_at is not None:
        outcome.monitors.setdefault(
            "book_snapshot_completed_at",
            book_snapshot_completed_at.isoformat(),
        )
        outcome.monitors.setdefault(
            "data_timestamp_used",
            book_snapshot_completed_at.astimezone(UTC).isoformat(),
        )
    outcome.monitors.setdefault(
        "outstanding_orders_start_total",
        sum(outstanding_start.values()),
    )
    outcome.monitors.setdefault(
        "outstanding_orders_start_breakdown",
        dict(sorted(outstanding_start.items())),
    )
    if fill_alpha_auto:
        outcome.monitors["fill_alpha_auto"] = fill_alpha_value
    else:
        outcome.monitors.setdefault("fill_alpha", fill_alpha_value)
    fee_series = (
        outcome.series.ticker.upper()
        if outcome.series and getattr(outcome.series, "ticker", None)
        else series_upper
    )
    outcome.monitors.setdefault("fee_path", _resolve_fee_paths(fee_series))
    should_archive = args.report or args.paper_ledger
    if should_archive and outcome.markets:
        expected_market_ids = {market.id for market in outcome.markets}
        missing_market_ids = expected_market_ids.difference(books_at_scan.keys())
        if missing_market_ids:
            outcome.monitors["orderbook_snapshot_missing"] = len(missing_market_ids)

    ledger = None
    if proposals:
        ledger = _maybe_simulate_ledger(
            args,
            proposals,
            client,
            orderbooks=books_at_scan,
            fill_alpha=fill_alpha_value,
            series=outcome.series,
            events=outcome.events,
            markets=outcome.markets,
        )
    execution_metrics: dict[str, object] | None = None
    if ledger:
        drawdown.record_pnl(ledger.total_expected_pnl())
        execution_metrics = ledger.execution_metrics()
        if execution_metrics:
            outcome.monitors.setdefault(
                "fill_ratio_avg",
                execution_metrics.get("fill_ratio_avg"),
            )
            outcome.monitors.setdefault(
                "fill_alpha_target_avg",
                execution_metrics.get("alpha_target_avg"),
            )
            outcome.monitors.setdefault(
                "slippage_ticks_avg",
                execution_metrics.get("slippage_ticks_avg"),
            )
            outcome.monitors.setdefault(
                "fill_minus_alpha",
                execution_metrics.get("fill_ratio_minus_alpha"),
            )
            outcome.monitors.setdefault(
                "slippage_delta_ticks",
                execution_metrics.get("slippage_ticks_avg"),
            )
            outcome.monitors.setdefault(
                "slippage_usd_avg",
                execution_metrics.get("slippage_usd_avg"),
            )
            outcome.monitors.setdefault(
                "ev_realized_bps_avg",
                execution_metrics.get("ev_realized_bps_avg"),
            )
    outcome.execution_metrics = execution_metrics
    exposure_summary = _compute_exposure_summary(proposals)
    _write_cdf_diffs(outcome.cdf_diffs)
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
    replay_path: Path | None = None
    if should_archive:
        archive_result = _archive_and_replay(
            client=client,
            series=outcome.series,
            events=outcome.events,
            markets=outcome.markets,
            orderbooks=books_at_scan,
            proposals_path=output_path,
            driver_fixtures=driver_fixtures,
            scanner_fixtures=fixtures_root,
            model_metadata=outcome.model_metadata,
        )
        if isinstance(archive_result, tuple):
            manifest_path, replay_path = archive_result
        else:
            manifest_path = archive_result
            replay_path = None
        if manifest_path and not args.quiet:
            print(f"Archived snapshot manifest at {manifest_path}")

    if manifest_path and book_snapshot_completed_at is not None:
        try:
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover - best effort metrics
            manifest_payload = None
        if isinstance(manifest_payload, dict):
            generated_at_raw = manifest_payload.get("generated_at")
            if isinstance(generated_at_raw, str):
                try:
                    archiver_ts = datetime.fromisoformat(generated_at_raw)
                except ValueError:  # pragma: no cover - tolerate malformed timestamp
                    archiver_ts = None
                if archiver_ts is not None:
                    latency = (archiver_ts - book_snapshot_completed_at).total_seconds() * 1000.0
                    outcome.monitors["book_latency_ms"] = round(max(0.0, latency), 3)

    ev_honesty_rows: list[dict[str, object]] = []
    ev_honesty_max_delta: float | None = None
    if replay_path:
        replay_records = _load_replay_for_ev_honesty(replay_path)
        ev_honesty_rows, ev_honesty_max_delta = _compute_ev_honesty_rows(proposals, replay_records)
        if ev_honesty_rows:
            outcome.monitors["ev_honesty_table"] = ev_honesty_rows
            outcome.monitors["ev_honesty_max_delta"] = ev_honesty_max_delta
            outcome.monitors["ev_honesty_count"] = len(ev_honesty_rows)
            if ev_honesty_max_delta is not None:
                outcome.monitors.setdefault("ev_per_contract_diff_max", ev_honesty_max_delta)
    _apply_ev_honesty_gate(outcome.monitors, threshold=0.10)

    if ledger and (args.paper_ledger or args.report):
        artifacts_dir = Path("reports/_artifacts")
        ledger.write_artifacts(artifacts_dir, manifest_path=manifest_path)

    gate_result: QualityGateResult | None = None
    go_status: bool | None = None
    if args.report or proposals or fatal_freshness_reason:
        gate_result = _quality_gate_for_broker(
            args,
            outcome.monitors,
            data_freshness_summary=data_freshness_summary,
        )
        go_status = gate_result.go
        outcome.monitors.setdefault("quality_gate_go", gate_result.go)
        if gate_result.reasons:
            outcome.monitors.setdefault("quality_gate_reasons", tuple(gate_result.reasons))
        polygon_ws_detail = _polygon_ws_freshness_detail(
            gate_details=gate_result.details,
            data_summary=data_freshness_summary,
            fatal_reason=fatal_freshness_reason,
        )
        if "polygon_ws_stale" in (gate_result.reasons or []):
            polygon_ws_detail["ok"] = False
            polygon_ws_detail["fatal_reason"] = fatal_freshness_reason or "polygon_ws_stale"
            existing_issues = set(polygon_ws_detail.get("issues", ()))
            existing_issues.add("quality_gate_reason=polygon_ws_stale")
            polygon_ws_detail["issues"] = tuple(sorted(existing_issues))
        gate_result.details["polygon_ws_freshness"] = polygon_ws_detail
        outcome.monitors["polygon_ws_freshness"] = polygon_ws_detail
    else:
        outcome.monitors.setdefault(
            "polygon_ws_freshness",
            _polygon_ws_freshness_detail(
                gate_details=None,
                data_summary=data_freshness_summary,
                fatal_reason=fatal_freshness_reason,
            ),
        )

    if cancel_reason:
        state = OutstandingOrdersState.load()
        state.mark_cancel_all(cancel_reason, modes=[args.broker])
        outcome.monitors.setdefault("cancel_all_reason", cancel_reason)

    broker_status = None
    if proposals:
        try:
            broker_status = execute_broker(
                broker_mode=args.broker,
                proposals=proposals,
                args=args,
                monitors=outcome.monitors,
                quiet=args.quiet,
                go_status=go_status,
            )
        except RuntimeError as exc:
            broker_status = {"mode": args.broker, "orders_recorded": 0, "error": str(exc)}
            if not args.quiet:
                print(f"[broker] {exc}")
        else:
            if broker_status and not args.quiet:
                print(
                    f"[broker] mode={broker_status.get('mode')} "
                    f"orders={broker_status.get('orders_recorded')}"
                )

    outstanding_summary = OutstandingOrdersState.load().summary()
    pilot_metadata = {
        "mode": args.broker,
        "kelly_cap": getattr(args, "kelly_cap", None),
        "max_var": getattr(args, "max_var", None),
        "fill_alpha": fill_alpha_value,
        "outstanding_total": sum(outstanding_summary.values()),
    }
    if pilot_session:
        pilot_metadata.update(pilot_session.metadata())
    _maybe_write_report(
        args,
        proposals,
        ledger,
        outcome.monitors,
        exposure_summary,
        manifest_path,
        go_status=go_status,
        fill_alpha=fill_alpha_value,
        mispricings=outcome.mispricings,
        model_metadata=outcome.model_metadata,
        outstanding_summary=outstanding_summary,
        pilot_metadata=pilot_metadata,
        execution_metrics=execution_metrics,
    )
    heartbeat_extra = {
        "outstanding": outstanding_summary,
        "broker": broker_status,
        "series": args.series.upper(),
    }
    if pilot_session:
        heartbeat_extra.update(
            {
                "pilot_session_id": pilot_session.session_id,
                "pilot_session_started_at": pilot_session.started_at.isoformat(),
                "pilot_mode": True,
            }
        )
    write_heartbeat(
        mode=f"scan_ladders:{args.series.upper()}",
        monitors=outcome.monitors,
        extra=heartbeat_extra,
    )

    monitor_snapshot = summarize_monitor_artifacts(
        MONITOR_ARTIFACTS_DIR,
        now=datetime.now(tz=UTC),
        window=timedelta(minutes=DEFAULT_PANIC_ALERT_WINDOW_MINUTES),
    )
    if pilot_session:
        write_pilot_session_artifact(
            session=pilot_session,
            ledger=ledger,
            monitors=outcome.monitors,
            monitor_summary=monitor_snapshot,
            broker_status=broker_status,
        )

    if not proposals:
        return


def _build_client(fixtures_root: Path, *, use_online: bool) -> KalshiPublicClient:
    api_fixtures = fixtures_root / "kalshi"
    return KalshiPublicClient(
        offline_dir=api_fixtures,
        use_offline=not use_online,
    )


def _run_discovery(args: argparse.Namespace) -> None:
    trading_day = _resolve_discover_day(args)
    fixtures_root = Path(args.fixtures_root).resolve()
    client = _build_client(fixtures_root, use_online=args.online)
    status = "open"
    results = discover_markets_for_day(client, trading_day=trading_day, status=status)
    mode = "online" if args.online else "offline"
    print(
        f"[discover] trading day {trading_day.isoformat()} (mode={mode}, status={status})"
    )
    if not results:
        print("[discover] no eligible INX/NDX markets found")
        return
    for window in results:
        expected = ", ".join(window.expected_series) if window.expected_series else "-"
        print(
            f"[discover] {window.label} ({window.target_type}) "
            f"target={window.target_et.isoformat()} expected={expected}"
        )
        if window.markets:
            for market in window.markets:
                event_label = market.event_ticker or market.event_id
                close_et = market.close_time_et.isoformat()
                print(
                    "  - {series} event={event} close={close} bins={bins}".format(
                        series=market.series,
                        event=event_label,
                        close=close_et,
                        bins=market.market_count,
                    )
                )
        else:
            print("  - (no matching markets)")
        if window.missing_series:
            print(f"    missing: {', '.join(window.missing_series)}")


def _resolve_discover_day(args: argparse.Namespace) -> date:
    if args.discover_date and args.today:
        raise ValueError("--discover-date cannot be combined with --today.")
    if args.discover_date:
        return args.discover_date
    reference = datetime.now(tz=_ET_ZONE)
    return reference.date()


def _build_pal_guard(args: argparse.Namespace) -> PALGuard:
    pal_policy_arg = getattr(args, "pal_policy", None)
    policy_path = Path(pal_policy_arg) if pal_policy_arg else Path("configs/pal_policy.yaml")
    if not policy_path.exists():
        policy_path = Path("configs/pal_policy.example.yaml")
    series_hint = getattr(args, "series", None)
    normalized_series = series_hint.upper() if isinstance(series_hint, str) else None

    def _load_policy(path: Path) -> PALPolicy:
        try:
            return PALPolicy.from_yaml(path, series=normalized_series)
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            return PALPolicy.from_yaml(path)

    try:
        policy = _load_policy(policy_path)
    except KeyError:
        fallback = Path("configs/pal_policy.example.yaml")
        policy = _load_policy(fallback)
    max_loss_per_strike = getattr(args, "max_loss_per_strike", None)
    if max_loss_per_strike is not None:
        policy = PALPolicy(
            series=policy.series,
            default_max_loss=max_loss_per_strike,
            per_strike=dict(policy.per_strike),
        )
    return PALGuard(policy)


def _build_risk_manager(args: argparse.Namespace) -> PortfolioRiskManager | None:
    portfolio_config = getattr(args, "portfolio_config", None)
    if portfolio_config:
        config = PortfolioConfig.from_yaml(Path(portfolio_config))
        return PortfolioRiskManager(config)
    if getattr(args, "max_var", None) is not None:
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
    cache = dict(orderbooks) if orderbooks is not None else {}
    if orderbooks is None:
        for proposal in proposals:
            if proposal.market_id in cache:
                continue
            try:
                cache[proposal.market_id] = client.get_orderbook(proposal.market_id)
            except Exception:  # pragma: no cover - tolerate missing books  # noqa: S112
                continue
    estimator = FillRatioEstimator(fill_alpha) if fill_alpha is not None else None
    event_lookup: dict[str, str] = {}
    if events is not None and markets is not None:
        event_tickers = {event.id: event.ticker for event in events}
        for market in markets:
            label = event_tickers.get(market.event_id) or market.ticker
            event_lookup[market.id] = label
    series_label = series.ticker if series is not None else args.series
    alpha_curve = load_alpha_curve(series_label)
    if alpha_curve is not None:
        estimator = None
    slippage_mode = (getattr(args, "slippage_mode", "top") or "top").lower()
    impact_cap_arg = getattr(args, "impact_cap", None)
    slippage_model = None
    slippage_curve = load_slippage_curve(series_label)
    if slippage_mode in {"top", "depth"} and slippage_curve is None:
        if impact_cap_arg is not None:
            slippage_model = SlippageModel(mode=slippage_mode, impact_cap=float(impact_cap_arg))
        elif slippage_mode == "depth":
            calibrated = load_slippage_model(series_label, mode=slippage_mode)
            if calibrated is not None:
                slippage_model = calibrated
        if slippage_model is None:
            slippage_model = SlippageModel(mode=slippage_mode)
    elif slippage_mode != "mid":
        slippage_mode = "top"
    ledger = simulate_fills(
        proposals,
        cache,
        fill_estimator=estimator,
        alpha_curve=alpha_curve,
        slippage_curve=slippage_curve,
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
    mispricings: Sequence[dict[str, object]] | None = None,
    model_metadata: dict[str, object] | None = None,
    outstanding_summary: dict[str, int] | None = None,
    pilot_metadata: dict[str, object] | None = None,
    execution_metrics: ExecutionMetrics | None = None,
) -> None:
    if not args.report:
        return
    effective_go = go_status
    if effective_go is None:
        try:
            gate_result = _quality_gate_for_broker(args, monitors or {})
        except Exception:  # pragma: no cover - defensive fallback
            gate_result = None
        else:
            effective_go = gate_result.go
    report_path = write_markdown_report(
        series=args.series,
        proposals=proposals,
        ledger=ledger,
        output_dir=Path("reports") / args.series.upper(),
        monitors=monitors,
        exposure_summary=exposure_summary,
        manifest_path=manifest_path,
        go_status=effective_go,
        fill_alpha=fill_alpha,
        mispricings=mispricings,
        model_metadata=model_metadata,
        outstanding_summary=outstanding_summary,
        pilot_metadata=pilot_metadata,
        execution_metrics=execution_metrics,
    )
    if not args.quiet:
        print(f"Wrote report to {report_path}")


def execute_broker(
    *,
    broker_mode: str,
    proposals: Sequence[Proposal],
    args: argparse.Namespace,
    monitors: dict[str, object],
    quiet: bool,
    go_status: bool | None = None,
) -> dict[str, object] | None:
    normalized = (broker_mode or "dry").strip().lower()
    if not proposals:
        return {"mode": normalized, "orders_recorded": 0}

    state = OutstandingOrdersState.load()
    kill_switch_path = resolve_kill_switch_path(getattr(args, "kill_switch_file", None))
    if kill_switch_path.exists():
        state.mark_cancel_all("kill_switch_engaged", modes=[normalized])
        raise RuntimeError(
            f"Kill switch engaged at {kill_switch_path.as_posix()}; refusing broker submission"
        )

    if go_status is None:
        gate_result = _quality_gate_for_broker(args, monitors)
        if not gate_result.go:
            state.mark_cancel_all("quality_gate_no_go", modes=[normalized])
            raise RuntimeError("Quality gates returned NO-GO; refusing broker submission")
    elif not go_status:
        state.mark_cancel_all("quality_gate_no_go", modes=[normalized])
        raise RuntimeError("Quality gates returned NO-GO; refusing broker submission")

    _enforce_broker_guards(proposals, args)
    orders = [_proposal_to_broker_order(p) for p in proposals]
    broker = create_broker(
        normalized,
        artifacts_dir=Path("reports/_artifacts"),
        audit_dir=Path("data/proc/audit"),
        acknowledge_risks=getattr(args, "i_understand_the_risks", False),
    )
    broker.place(orders)
    state.record_submission(normalized, orders)
    status = broker.status()
    status.setdefault("orders_recorded", len(orders))
    if not quiet:
        status_msg = status.get("message") or ""
        if status_msg:
            print(f"[broker] {status_msg}")
    return status


def _proposal_to_broker_order(proposal: Proposal) -> BrokerOrder:
    key_source = f"{proposal.market_id}|{proposal.side}|{proposal.strike:.3f}|{proposal.contracts}"
    key = hashlib.sha256(key_source.encode("utf-8")).hexdigest()
    metadata = dict(proposal.metadata or {})
    metadata.setdefault("market_ticker", proposal.market_ticker)
    metadata.setdefault("series", proposal.series)
    metadata.setdefault("liquidity", metadata.get("liquidity", "maker"))
    metadata.setdefault("strike", proposal.strike)
    metadata.setdefault("market_id", proposal.market_id)
    return BrokerOrder(
        idempotency_key=key,
        market_id=proposal.market_id,
        strike=proposal.strike,
        side=proposal.side,
        price=proposal.market_yes_price,
        contracts=proposal.contracts,
        probability=proposal.strategy_probability,
        metadata=metadata,
    )


def _enforce_broker_guards(proposals: Sequence[Proposal], args: argparse.Namespace) -> None:
    pal_guard = _build_pal_guard(args)
    risk_manager = _build_risk_manager(args)
    max_var = getattr(args, "max_var", None)
    daily_loss_cap = getattr(args, "daily_loss_cap", None)
    weekly_loss_cap = getattr(args, "weekly_loss_cap", None)
    daily_budget = LossBudget(daily_loss_cap)
    weekly_budget = LossBudget(weekly_loss_cap)
    limit_checker = ProposalLimitChecker(pal_guard, daily_budget=daily_budget, weekly_budget=weekly_budget)

    for proposal in proposals:
        order_id = f"{proposal.market_ticker}:{proposal.strike}"
        liquidity_meta = str((proposal.metadata or {}).get("liquidity") or "maker").lower()
        liquidity_flag = Liquidity.TAKER if liquidity_meta == "taker" else Liquidity.MAKER
        order = OrderProposal(
            strike_id=order_id,
            yes_price=proposal.market_yes_price,
            contracts=proposal.contracts,
            side=OrderSide[proposal.side.upper()],
            liquidity=liquidity_flag,
            market_name=proposal.market_ticker,
            series=proposal.series,
        )
        total_max_loss = proposal.max_loss
        if risk_manager and not risk_manager.can_accept(
            strategy=proposal.strategy,
            max_loss=total_max_loss,
            max_var=max_var,
        ):
            raise RuntimeError("Portfolio VaR limit reached; aborting broker submission")
        try:
            limit_checker.try_accept(order, max_loss=total_max_loss)
        except LimitViolation as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Limit violation for {order_id}: {exc}") from exc


def _quality_gate_for_broker(
    args: argparse.Namespace,
    monitors: dict[str, object],
    *,
    data_freshness_summary: dict[str, object] | None = None,
) -> QualityGateResult:
    now_utc = datetime.now(tz=UTC)
    monitor_summary = summarize_monitor_artifacts(
        MONITOR_ARTIFACTS_DIR,
        now=now_utc,
        window=timedelta(minutes=DEFAULT_PANIC_ALERT_WINDOW_MINUTES),
    )
    config_override = getattr(args, "quality_gates_config", None)
    if config_override:
        config_path = Path(config_override)
    else:
        config_path = resolve_quality_gate_config_path()
    config = load_quality_gate_config(config_path)
    result = run_quality_gates(
        config=config,
        now=now_utc,
        proc_root=PROC_ROOT,
        raw_root=RAW_ROOT,
        monitors=monitors,
    )
    drawdown_status = drawdown.check_limits(
        getattr(args, "daily_loss_cap", None),
        getattr(args, "weekly_loss_cap", None),
        now=now_utc,
    )
    freshness_path = MONITOR_ARTIFACTS_DIR / FRESHNESS_ARTIFACT_PATH.name
    if data_freshness_summary is None:
        data_freshness_payload = load_freshness_artifact(freshness_path)
        data_freshness_summary = summarize_freshness_artifact(
            data_freshness_payload,
            artifact_path=freshness_path,
        )
    else:
        # Create a shallow copy to avoid mutating caller state.
        data_freshness_summary = dict(data_freshness_summary)

    reasons = list(result.reasons)
    for reason in result.reasons:
        if isinstance(reason, str) and reason.startswith("data_freshness.polygon_ws.stale"):
            if "polygon_ws_stale" not in reasons:
                reasons.append("polygon_ws_stale")
    details = dict(result.details)
    details.setdefault("data_freshness", data_freshness_summary)
    clock_skew = None
    if monitors is not None:
        value = monitors.get("clock_skew_seconds")
        if isinstance(value, (int, float)):
            clock_skew = float(value)
            details.setdefault("clock_skew_seconds", clock_skew)

    def _append_reason(reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    runtime_monitor_details: dict[str, object] = {
        "statuses": monitor_summary.statuses,
        "alerts_recent": sorted(monitor_summary.alerts_recent),
        "max_age_minutes": monitor_summary.max_age_minutes,
        "file_count": monitor_summary.file_count,
    }
    if monitor_summary.metrics:
        runtime_monitor_details["metrics"] = monitor_summary.metrics
    details.setdefault("runtime_monitors", runtime_monitor_details)
    if drawdown_status.metrics:
        details.setdefault("drawdown", drawdown_status.metrics)
    go_flag = result.go and drawdown_status.ok and data_freshness_summary.get("required_feeds_ok", True)
    if clock_skew is not None and clock_skew > CLOCK_SKEW_THRESHOLD_SECONDS:
        go_flag = False
        _append_reason("clock_skew_exceeded")
        details.setdefault("clock_skew_threshold", CLOCK_SKEW_THRESHOLD_SECONDS)
    if monitors is not None and not bool(monitors.get("index_rules_ok", True)):
        go_flag = False
        _append_reason("index_rules_mismatch")
        reasons_payload = monitors.get("index_rules_reasons")
        if reasons_payload:
            details.setdefault("index_rules_reasons", list(reasons_payload))
    if not drawdown_status.ok:
        for reason in drawdown_status.reasons:
            _append_reason(reason)
    if not data_freshness_summary.get("required_feeds_ok", True):
        _append_reason("STALE_FEEDS")
        stale_feeds = data_freshness_summary.get("stale_feeds") or []
        stale_normalized = {str(feed).strip().lower() for feed in stale_feeds if isinstance(feed, str)}
        if "polygon_index.websocket" in stale_normalized:
            _append_reason("polygon_ws_stale")
        if data_freshness_summary.get("status") == "MISSING":
            _append_reason("data_freshness_missing")
    monitor_reasons: list[str] = []
    if monitor_summary.file_count == 0:
        monitor_reasons.append("monitors_missing")
    elif (
        monitor_summary.max_age_minutes is not None
        and monitor_summary.max_age_minutes > DEFAULT_MONITOR_MAX_AGE_MINUTES
    ):
        monitor_reasons.append("monitors_stale")
    if (
        DEFAULT_PANIC_ALERT_THRESHOLD > 0
        and len(monitor_summary.alerts_recent) >= DEFAULT_PANIC_ALERT_THRESHOLD
    ):
        monitor_reasons.append("panic_backoff")
    if monitor_summary.statuses.get("kill_switch") == "ALERT":
        monitor_reasons.append("kill_switch_engaged")
    if monitor_summary.statuses.get("drawdown") == "ALERT":
        monitor_reasons.append("drawdown")
    if monitor_summary.statuses.get("ev_seq_guard") == "ALERT":
        monitor_reasons.append("sequential_alert")
    if monitor_summary.statuses.get("freeze_window") == "ALERT":
        monitor_reasons.append("freeze_window")

    blocking_monitor_reasons = {
        "monitors_missing",
        "monitors_stale",
        "panic_backoff",
        "kill_switch_engaged",
        "sequential_alert",
        "freeze_window",
    }
    for reason in monitor_reasons:
        _append_reason(reason)
    if any(reason in blocking_monitor_reasons for reason in monitor_reasons):
        go_flag = False

    kill_switch_path = resolve_kill_switch_path(getattr(args, "kill_switch_file", None))
    if kill_switch_engaged(kill_switch_path):
        go_flag = False
        _append_reason("kill_switch_engaged")
        details["kill_switch_path"] = kill_switch_path.as_posix()
        OutstandingOrdersState.load().mark_cancel_all(
            "kill_switch_engaged",
            modes=[(getattr(args, "broker", "dry") or "dry")],
        )

    stale, heartbeat_payload = heartbeat_stale(threshold=timedelta(minutes=5))
    if stale:
        go_flag = False
        _append_reason("heartbeat_stale")
        if heartbeat_payload:
            details.setdefault("heartbeat", heartbeat_payload)

    ev_no_go = bool(monitors.get("ev_honesty_no_go")) if monitors is not None else False
    if ev_no_go:
        go_flag = False
        _append_reason("ev_honesty_stale")
        ev_payload: dict[str, object] = {}
        if monitors is not None:
            max_delta = monitors.get("ev_honesty_max_delta")
            threshold = monitors.get("ev_honesty_threshold")
            latency_ms = monitors.get("book_latency_ms")
            if max_delta is not None:
                ev_payload["max_delta"] = max_delta
            if threshold is not None:
                ev_payload["threshold"] = threshold
            if latency_ms is not None:
                ev_payload["book_latency_ms"] = latency_ms
        if ev_payload:
            details["ev_honesty"] = ev_payload

    combined = QualityGateResult(go=go_flag, reasons=reasons, details=details)
    write_go_no_go(combined)
    return combined


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
    model_metadata: dict[str, object] | None = None,
) -> tuple[Path | None, Path | None]:
    if series is None:
        return None, None
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
    metadata = model_metadata or {}
    replay_path = replay_manifest(
        manifest_path,
        model_version=str(metadata.get("model_version", "v15")),
        orderbooks_override=orderbooks,
    )
    return manifest_path, replay_path


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


def _load_replay_for_ev_honesty(path: Path | None) -> list[dict[str, object]]:
    if path is None:
        return []
    if not path.exists():
        return []
    try:
        frame = pl.read_parquet(path)
    except Exception:  # pragma: no cover - tolerate parquet failures
        return []
    if frame.is_empty():
        return []
    return frame.to_dicts()


def _compute_ev_honesty_rows(
    proposals: Sequence[Proposal],
    replay_rows: Sequence[dict[str, object]],
) -> tuple[list[dict[str, object]], float | None]:
    if not proposals or not replay_rows:
        return [], None

    def _to_float(value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    lookup: dict[tuple[str, float], dict[str, object]] = {}
    for row in replay_rows:
        market_id = row.get("market_id")
        strike_value = row.get("strike")
        if market_id is None or strike_value is None:
            continue
        try:
            key = (str(market_id), float(strike_value))
        except (TypeError, ValueError):  # pragma: no cover - malformed payloads
            continue
        lookup[key] = row

    results: list[dict[str, object]] = []
    max_delta = 0.0
    for proposal in proposals:
        key = (proposal.market_id, float(proposal.strike))
        replay_row = lookup.get(key)
        if replay_row is None:
            continue
        fill_price = _to_float(replay_row.get("fill_price"), default=proposal.market_yes_price)
        summary_per = expected_value_summary(
            contracts=1,
            yes_price=fill_price,
            event_probability=float(proposal.strategy_probability),
            schedule=DEFAULT_FEE_SCHEDULE,
            series=proposal.series,
            market_name=proposal.market_ticker,
        )
        shrink_factor = float((proposal.metadata or {}).get("ev_shrink", 1.0)) if proposal.metadata else 1.0
        if shrink_factor not in (0.0, 1.0):
            summary_per["maker_yes"] *= shrink_factor
            summary_per["maker_no"] *= shrink_factor
        maker_key = "maker_yes" if proposal.side.upper() == "YES" else "maker_no"
        original_per = _to_float(summary_per.get(maker_key))
        replay_per = _to_float(
            replay_row.get("maker_ev_per_contract_replay", replay_row.get("maker_ev_replay")),
            default=_to_float(replay_row.get("maker_ev_replay")),
        )
        delta = abs(original_per - replay_per)
        max_delta = max(max_delta, delta)
        proposal_per = _to_float(proposal.maker_ev_per_contract)
        results.append(
            {
                "market_id": proposal.market_id,
                "market_ticker": proposal.market_ticker,
                "strike": float(proposal.strike),
                "side": proposal.side,
                "maker_ev_per_contract_original": original_per,
                "maker_ev_per_contract_replay": replay_per,
                "maker_ev_per_contract_proposal": proposal_per,
                "maker_ev_original": _to_float(proposal.maker_ev),
                "maker_ev_replay": _to_float(replay_row.get("maker_ev_replay")),
                "fill_price": fill_price,
                "delta": delta,
            }
        )

    results.sort(key=lambda row: (row["market_ticker"], row["strike"], row["side"]))
    if not results:
        return [], None
    return results, max_delta


def _apply_ev_honesty_gate(monitors: dict[str, object], *, threshold: float) -> None:
    monitors.setdefault("ev_honesty_threshold", threshold)
    max_delta_raw = monitors.get("ev_honesty_max_delta")
    try:
        max_delta = float(max_delta_raw) if max_delta_raw is not None else None
    except (TypeError, ValueError):  # pragma: no cover - contaminated monitor value
        max_delta = None
    if max_delta is None:
        monitors["ev_honesty_no_go"] = False
        return
    monitors["ev_honesty_no_go"] = bool(max_delta > threshold)
    if max_delta > threshold:
        monitors["ev_honesty_delta_excess"] = round(max_delta - threshold, 6)


def _parse_date_arg(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Trading day must be YYYY-MM-DD") from exc


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan Kalshi ladders and output dry-run proposals."
    )
    parser.add_argument(
        "--series",
        help="Kalshi series ticker, e.g. CPI",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="List INX/NDX markets for the requested trading day and exit without scanning.",
    )
    parser.add_argument(
        "--discover-date",
        type=_parse_date_arg,
        help="Trading day (YYYY-MM-DD) for --discover outputs (defaults to today in ET).",
    )
    parser.add_argument(
        "--today",
        action="store_true",
        help="Shortcut for --discover-date=<today in ET> (only valid with --discover).",
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
        "--quality-gates-config",
        type=Path,
        default=None,
        help="Override quality gates configuration file.",
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
        "--ev-honesty-shrink",
        type=float,
        default=1.0,
        help="Maker EV shrink factor used for EV honesty (0-1).",
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
        dest="maker_only",
        action="store_true",
        help="Only consider maker-side executions (default).",
    )
    parser.add_argument(
        "--allow-taker",
        dest="maker_only",
        action="store_false",
        help="Permit taker-side proposals (unsafe; disables maker-only guard).",
    )
    parser.add_argument(
        "--sniper",
        action="store_true",
        help="Take top-of-book liquidity when market prob deviates >5% from model.",
    )
    parser.add_argument(
        "--sniper-threshold",
        type=float,
        default=0.05,
        help="Absolute probability gap required to trigger sniper taker (default: 0.05).",
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
        "--correlation-config",
        type=Path,
        default=CORRELATION_CONFIG_PATH,
        help="Path to correlation-aware VaR configuration file.",
    )
    parser.add_argument(
        "--portfolio-config",
        type=Path,
        help="Path to portfolio factor configuration YAML file.",
    )
    parser.add_argument(
        "--mispricing-only",
        action="store_true",
        help="Restrict proposals to bins participating in detected kink spreads.",
    )
    parser.add_argument(
        "--max-legs",
        type=int,
        default=4,
        help="Maximum number of adjacent bins to consider when forming mispricing spreads.",
    )
    parser.add_argument(
        "--prob-sum-gap-threshold",
        type=float,
        default=0.0,
        help="Minimum probability mass gap required to log mispricing diagnostics.",
    )
    parser.add_argument(
        "--broker",
        default="dry",
        choices=["dry", "live"],
        help="Broker adapter to use when executing orders.",
    )
    parser.add_argument(
        "--clear-dry-orders-start",
        action="store_true",
        help="Clear outstanding DRY orders before scanning.",
    )
    parser.add_argument(
        "--i-understand-the-risks",
        action="store_true",
        help="Required acknowledgement flag when arming the live broker adapter.",
    )
    parser.add_argument(
        "--kill-switch-file",
        help="Path to kill-switch sentinel file (default: data/proc/state/kill_switch).",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Enable live pilot mode with restricted sizing and tagging.",
    )
    parser.add_argument(
        "--pilot-config",
        type=Path,
        help="Override pilot configuration YAML (defaults to configs/pilot.yaml).",
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
    parser.set_defaults(maker_only=True)
    return parser.parse_args(argv)


def _parse_hour_label(ticker: str | None) -> tuple[int, int] | None:
    if not ticker:
        return None
    match = _HOUR_PATTERN.search(ticker.upper())
    if match is None:
        return None
    try:
        hour = int(match.group("hour"))
        minute = int(match.group("minute"))
    except (KeyError, ValueError):
        return None
    return hour % 24, minute % 60


def _format_hour_label(hour: int) -> str:
    return f"H{hour % 24:02d}00"


def _default_hourly_target(now_utc: datetime) -> time:
    now_et = now_utc.astimezone(_ET_ZONE)
    target_hour = now_et.hour
    if now_et.minute >= 40:
        target_hour = (target_hour + 1) % 24
    return time(target_hour, 0)


def _ops_window_for_series(series: str) -> IndexOpsWindow | None:
    try:
        return INDEX_OPS_CONFIG.window_for_series(series)
    except KeyError:
        return None


def _cancel_buffer_seconds(series: str | None = None) -> float:
    if series is not None:
        window = _ops_window_for_series(series)
        if window is not None:
            return float(window.cancel_buffer_seconds)
    return float(_WINDOW_CLOSE.cancel_buffer_seconds)


def _ops_window_metadata(series: str, now_utc: datetime, *, target_time: time | None = None) -> dict[str, object]:
    window = _ops_window_for_series(series)
    if window is None:
        return {}
    tz = INDEX_OPS_CONFIG.timezone
    reference = now_utc.astimezone(tz)
    resolved_target = target_time
    if resolved_target is None and window.start_offset_minutes is not None:
        resolved_target = _default_hourly_target(now_utc)
    fallback_reason: str | None = None
    try:
        start_local, end_local = window.bounds_for(
            reference=reference,
            target_time=resolved_target,
            timezone=tz,
        )
    except ValueError:
        # Fall back to default target when offsets are defined but target missing
        if resolved_target is None:
            fallback_reason = "on_before"
            start_local, end_local = window.bounds_for(
                reference=reference,
                target_time=_default_hourly_target(now_utc),
                timezone=tz,
            )
        else:
            raise
    cancel_buffer = float(window.cancel_buffer_seconds)
    cancel_by = end_local - timedelta(seconds=cancel_buffer)
    seconds_to_cancel = max((end_local - reference).total_seconds() - cancel_buffer, 0.0)
    target_dt_et = end_local
    target_dt_utc = target_dt_et.astimezone(UTC)
    return {
        "ops_window_name": window.name,
        "ops_window_start_et": start_local.isoformat(),
        "ops_window_end_et": end_local.isoformat(),
        "ops_cancel_buffer_seconds": cancel_buffer,
        "ops_seconds_to_cancel": round(seconds_to_cancel, 3),
        "ops_cancel_at": cancel_by.isoformat(),
        "ops_timezone": tz.key if hasattr(tz, "key") else str(tz),
        "ops_target_et": target_dt_et.isoformat(),
        "ops_target_unix": int(target_dt_utc.timestamp()),
        "ops_target_fallback": fallback_reason,
    }


def _scheduler_window_payload(window: TradingWindow | None) -> dict[str, object] | None:
    if window is None:
        return None
    return {
        "label": window.label,
        "target_type": window.target_type,
        "series": window.series,
        "start_et": window.start_et.isoformat(),
        "target_et": window.target_et.isoformat(),
        "freeze_et": window.freeze_et.isoformat(),
        "freshness_strict_et": window.freshness_strict_et.isoformat(),
    }


def _u_series_roll_decision(now_utc: datetime) -> dict[str, object]:
    now_et = now_utc.astimezone(_ET_ZONE)
    current_hour = now_et.hour
    minute = now_et.minute
    target_hour = current_hour
    if minute >= 40:
        target_hour = (current_hour + 1) % 24
    rolled = target_hour != current_hour
    boundary = now_et.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    seconds_to_boundary = max((boundary - now_et).total_seconds(), 0.0)
    cancel_buffer = float(_WINDOW_HOURLY.cancel_buffer_seconds)
    cancel_required = seconds_to_boundary <= cancel_buffer
    return {
        "current_hour": current_hour,
        "target_hour": target_hour,
        "rolled": rolled,
        "now_et": now_et,
        "seconds_to_boundary": seconds_to_boundary,
        "cancel_required": cancel_required,
        "cancel_buffer_seconds": cancel_buffer,
    }


def _filter_events_by_hour(events: Sequence[Event], target_hour: int) -> list[Event]:
    if not events:
        return []
    normalized_hour = target_hour % 24
    matching: list[tuple[int, Event]] = []
    fallback: list[Event] = []
    for event in events:
        parsed = _parse_hour_label(event.ticker)
        if parsed is None:
            fallback.append(event)
            continue
        if parsed[0] == normalized_hour:
            matching.append((parsed[1], event))
    if matching:
        matching.sort(key=lambda item: item[0])
        return [item[1] for item in matching]
    return fallback if fallback else list(events)


def _filter_u_series_events(
    events: Sequence[Event],
    *,
    decision: dict[str, object],
) -> list[Event]:
    if not events:
        return []
    target_hour = int(decision.get("target_hour", 0)) % 24
    parsed: list[tuple[int, int, Event]] = []
    unmatched: list[Event] = []
    for event in events:
        parsed_hour = _parse_hour_label(event.ticker)
        if parsed_hour is None:
            unmatched.append(event)
            continue
        parsed.append((parsed_hour[0], parsed_hour[1], event))
    if parsed:
        matching = [entry for entry in parsed if entry[0] == target_hour]
        if matching:
            matching.sort(key=lambda item: (item[1], item[0]))
            return [entry[2] for entry in matching]
        parsed.sort(key=lambda item: ((item[0] - target_hour) % 24, item[1]))
        return [parsed[0][2]]
    if unmatched:
        return list(unmatched)
    return list(events)


def _expected_rule_hour(series_code: str) -> int | None:
    code = series_code.upper()
    if code in _U_SERIES:
        return 12
    if code in {"INX", "NASDAQ100"}:
        return 16
    return None


def _validate_index_rules(
    series: Series | None,
    events: Sequence[Event],
    rule: IndexRule | None,
) -> dict[str, object]:
    if series is None or rule is None:
        return {"ok": True, "reasons": []}
    expected_hour = _expected_rule_hour(series.ticker)
    reasons: list[str] = []
    ok = True
    if expected_hour is not None:
        if f"{expected_hour:02d}:00" not in rule.evaluation_time_et:
            ok = False
            reasons.append("evaluation_time_mismatch")
    fallback_text = str(rule.fallback_clause or "").strip()
    if not fallback_text:
        ok = False
        reasons.append("fallback_missing")
    event_hours = {
        (_parse_hour_label(event.ticker) or (None, None))[0]
        for event in events
    }
    event_hours.discard(None)
    if expected_hour is not None and event_hours and expected_hour not in event_hours:
        ok = False
        reasons.append("event_hour_mismatch")
    return {"ok": ok, "reasons": reasons}


def _clock_skew_seconds(reference_utc: datetime | None = None) -> float:
    now_utc = reference_utc or datetime.now(tz=UTC)
    et_from_reference = now_utc.astimezone(_ET_ZONE)
    local_et = datetime.now(tz=_ET_ZONE)
    return abs((local_et - et_from_reference).total_seconds())


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
    ev_honesty_shrink: float = 0.9,
    daily_loss_cap: float | None = None,
    weekly_loss_cap: float | None = None,
    mispricing_only: bool = False,
    max_legs: int = 4,
    prob_sum_gap_threshold: float = 0.0,
    model_version: str = "v15",
    pilot_config: PilotConfig | None = None,
    bin_constraints: BinConstraintResolver | None = None,
    now_override: datetime | None = None,
    target_time_override: time | None = None,
    var_limiter: var_index.FamilyVarLimiter | None = None,
    correlation_guard: CorrelationAwareLimiter | None = None,
    quote_optimizer: QuoteOptimizer | None = None,
    freshness_ms: float | None = None,
    sniper: bool = False,
    sniper_threshold: float = 0.05,
) -> ScanOutcome:
    now_utc = now_override if now_override is not None else datetime.now(tz=UTC)
    regime_flags = regimes.regime_for(now_utc)
    base_contracts = max(int(contracts), 1)
    contracts_per_quote = max(1, int(round(base_contracts * regime_flags.size_multiplier)))
    kelly_cap_regime = max(float(kelly_cap) * regime_flags.slo_multiplier, 0.05)
    sigma_artifact = sigma_drift.load_artifact()
    sniper_threshold = max(0.0, float(sniper_threshold))
    sigma_shrink: float | None = None
    series_obj = _find_series(client, series)
    daily_budget = LossBudget(daily_loss_cap)
    weekly_budget = LossBudget(weekly_loss_cap)
    limit_checker = ProposalLimitChecker(pal_guard, daily_budget=daily_budget, weekly_budget=weekly_budget)
    if series_obj is not None:
        sigma_candidate = sigma_drift.shrink_for_series(series_obj.ticker, artifact=sigma_artifact)
        if sigma_candidate is not None and 0.0 < sigma_candidate < ev_honesty_shrink:
            sigma_shrink = sigma_candidate
            ev_honesty_shrink = sigma_candidate
    events = client.get_events(series_obj.id)
    try:
        rule_config = lookup_index_rule(series_obj.ticker)
    except KeyError:
        rule_config = None
    events_to_scan = list(events)
    roll_decision: dict[str, object] | None = None
    guardrail_reasons: list[str] = []
    target_time: time | None = target_time_override
    if series_obj.ticker.upper() in _U_SERIES:
        if target_time_override is None:
            roll_decision = _u_series_roll_decision(now_utc)
            filtered = _filter_u_series_events(events_to_scan, decision=roll_decision)
            if filtered:
                events_to_scan = filtered
            seconds_to_boundary = float(roll_decision.get("seconds_to_boundary", 0.0)) if roll_decision else 0.0
            default_hourly_target = _default_hourly_target(now_utc)
            target_hour = (
                int(roll_decision.get("target_hour", default_hourly_target.hour))
                if roll_decision
                else default_hourly_target.hour
            )
            target_time = time(target_hour % 24, 0)
            current_hour = int(roll_decision.get("current_hour", 0)) if roll_decision else 0
            event_hours = {
                hour
                for hour, _minute in (
                    _parse_hour_label(event.ticker) or (None, None)
                    for event in events
                )
                if hour is not None
            }
            filtered_hours = {
                hour
                for hour, _minute in (
                    _parse_hour_label(event.ticker) or (None, None)
                    for event in events_to_scan
                )
                if hour is not None
            }
            has_fallback_event = any(((hour - target_hour) % 24) == 1 for hour in filtered_hours)
            has_target_event = target_hour in event_hours
            if seconds_to_boundary <= 180.0 and not has_target_event and not has_fallback_event:
                guardrail_reasons.append("next_hour_missing")
            if seconds_to_boundary <= 10.0 and seconds_to_boundary > 0.0:
                guardrail_reasons.append("pre_boundary_cooldown")
            if seconds_to_boundary <= 0.0:
                guardrail_reasons.append("past_boundary")
            cancel_buffer = _cancel_buffer_seconds(series_obj.ticker)
            if seconds_to_boundary <= cancel_buffer:
                state = OutstandingOrdersState.load()
                if state.total() > 0:
                    guardrail_reasons.append("cancel_ack_pending")
            now_et_value = roll_decision.get("now_et") if roll_decision else None
            if isinstance(now_et_value, datetime):
                seconds_since_boundary = float(
                    now_et_value.minute * 60
                    + now_et_value.second
                    + now_et_value.microsecond / 1_000_000
                )
                if current_hour == target_hour and seconds_since_boundary < 1.0:
                    guardrail_reasons.append("post_roll_delay")
            guardrail_reasons = sorted(set(guardrail_reasons))
            if guardrail_reasons:
                events_to_scan = []
        else:
            target_hour = target_time_override.hour
            events_to_scan = _filter_events_by_hour(events_to_scan, target_hour)
    ops_metadata: dict[str, object] = {}
    if series_obj is not None:
        ops_metadata = _ops_window_metadata(series_obj.ticker, now_utc, target_time=target_time)
    rule_validation = _validate_index_rules(series_obj, events, rule_config)
    max_legs = max(2, int(max_legs))
    prob_sum_gap_threshold = float(max(prob_sum_gap_threshold, 0.0))

    proposals: list[Proposal] = []
    non_monotone = 0
    daily_budget = LossBudget(daily_loss_cap)
    cdf_diffs: list[dict[str, object]] = []
    mispricing_records: list[dict[str, object]] = []
    all_markets: list[Market] = []
    books_at_scan: dict[str, Orderbook] = {}
    book_snapshot_failures = 0
    book_snapshot_started_at = now_utc
    pilot_trimmed_bins = 0
    range_structure_summary: list[dict[str, object]] = []
    range_structure_sigma = 0.0
    replacement_throttle = quote_microprice.ReplacementThrottle()

    model_metadata: dict[str, object] = {}
    imbalance_cache: dict[str, float | None] = {}
    for event in events_to_scan:
        markets = client.get_markets(event.id)
        all_markets.extend(markets)
        for market in markets:
            if market.id not in books_at_scan:
                try:
                    books_at_scan[market.id] = client.get_orderbook(market.id)
                except Exception:  # pragma: no cover - tolerate missing books
                    book_snapshot_failures += 1
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

            orderbook_imbalance: float | None = None
            event_timestamp = now_utc
            if series_obj is not None and series_obj.ticker.upper() == "TNEY":
                ticker_key = market.ticker
                if ticker_key in imbalance_cache:
                    orderbook_imbalance = imbalance_cache[ticker_key]
                else:
                    imbalance_entry = kalshi_ws.load_latest_imbalance(ticker_key)
                    orderbook_imbalance = float(imbalance_entry[0]) if imbalance_entry is not None else None
                    imbalance_cache[ticker_key] = orderbook_imbalance
            strategy_pmf, metadata = _strategy_pmf_for_series(
                series=series_obj.ticker,
                strikes=[rung.strike for rung in rungs],
                fixtures_dir=driver_fixtures,
                override=strategy_name,
                offline=offline,
                model_version=model_version,
                orderbook_imbalance=orderbook_imbalance,
                event_timestamp=event_timestamp,
                target_time=target_time,
            )
            if metadata and not model_metadata:
                model_metadata = dict(metadata)
            strategy_survival = pmf_to_survival(strategy_pmf, [rung.strike for rung in rungs])
            if correlation_guard is not None:
                correlation_guard.update_surface(series_obj.ticker, [rung.strike for rung in rungs], strategy_survival)
            structures = build_range_structures(
                series=series_obj.ticker if series_obj is not None else series,
                market_id=market.id,
                market_ticker=market.ticker,
                rungs=rungs,
                strategy_survival=strategy_survival,
                contracts=contracts_per_quote,
            )
            if structures:
                range_structure_summary.append(
                    {
                        "market_id": market.id,
                        "market_ticker": market.ticker,
                        "structures": [entry.to_summary() for entry in structures],
                    }
                )
                range_structure_sigma += sum(entry.sigma for entry in structures)
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

            kink_metrics = implied_cdf_kinks(strategy_survival)
            prob_gap = prob_sum_gap(strategy_pmf)
            spreads = kink_spreads(
                strategy_pmf[: len(rungs)],
                market_pmf[: len(rungs)],
                max_legs=max_legs,
            )
            mispricing_indices: set[int] = set()
            for spread in spreads:
                mispricing_indices.update(range(int(spread["start_index"]), int(spread["end_index"]) + 1))
            if prob_gap >= prob_sum_gap_threshold and not mispricing_indices:
                mispricing_indices.update(range(len(rungs)))
            if mispricing_only:
                if mispricing_indices:
                    if allowed_indices is None:
                        allowed_indices = set(mispricing_indices)
                    else:
                        allowed_indices = set(allowed_indices) & mispricing_indices
                else:
                    allowed_indices = set()

            if prob_gap >= prob_sum_gap_threshold or spreads:
                mispricing_records.append(
                    {
                        "market_id": market.id,
                        "market_ticker": market.ticker,
                        "prob_sum_gap": prob_gap,
                        "max_kink": kink_metrics.max_kink,
                        "mean_abs_kink": kink_metrics.mean_abs_kink,
                        "monotonicity_penalty": kink_metrics.monotonicity_penalty,
                        "kink_count": kink_metrics.kink_count,
                        "spreads": spreads[:3],
                    }
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
                kelly_cap=kelly_cap_regime,
                uncertainty_penalty=uncertainty_penalty,
                ob_imbalance_penalty=ob_imbalance_penalty,
                daily_budget=daily_budget,
                weekly_budget=weekly_budget,
                limit_checker=limit_checker,
                series_ticker=series_obj.ticker,
                ev_shrink=ev_honesty_shrink,
                bin_constraints=bin_constraints,
                var_limiter=var_limiter,
                correlation_guard=correlation_guard,
                quote_optimizer=quote_optimizer,
                orderbook=books_at_scan.get(market.id),
                freshness_ms=freshness_ms,
                replacement_throttle=replacement_throttle,
                sniper=sniper,
                sniper_threshold=sniper_threshold,
                now_ts=now_utc,
            )
            if pilot_config is not None:
                rung_proposals, trimmed = _limit_proposals_for_pilot(
                    rung_proposals,
                    max_unique_bins=pilot_config.max_unique_bins,
                )
                pilot_trimmed_bins += trimmed
            proposals.extend(rung_proposals)

    book_snapshot_completed_at = datetime.now(tz=UTC)

    monitors = {
        "non_monotone_ladders": non_monotone,
        "model_drift": _model_drift_flag(series_obj.ticker),
        "tz_not_et": _tz_not_et(),
        "orderbook_snapshots": len(books_at_scan),
        "ev_honesty_shrink": ev_honesty_shrink,
        "contracts_per_quote": contracts_per_quote,
        "kelly_cap_effective": round(kelly_cap_regime, 4),
        "regime": regime_flags.label,
        "regime_size_multiplier": regime_flags.size_multiplier,
        "regime_slo_multiplier": regime_flags.slo_multiplier,
    }
    if sigma_shrink is not None:
        monitors["sigma_drift_shrink"] = sigma_shrink
    if ops_metadata:
        monitors.update(ops_metadata)
    if daily_budget.cap is not None:
        monitors.setdefault("daily_loss_remaining", daily_budget.remaining)
    if weekly_budget.cap is not None:
        monitors.setdefault("weekly_loss_remaining", weekly_budget.remaining)
    if rule_validation is not None:
        monitors.setdefault("index_rules_ok", bool(rule_validation.get("ok", True)))
        if not rule_validation.get("ok", True) and rule_validation.get("reasons"):
            monitors["index_rules_reasons"] = tuple(rule_validation.get("reasons", ()))
    else:
        monitors.setdefault("index_rules_ok", True)
    if roll_decision is not None:
        monitors["u_series_roll_from"] = _format_hour_label(int(roll_decision.get("current_hour", 0)))
        monitors["u_series_roll_to"] = _format_hour_label(int(roll_decision.get("target_hour", 0)))
        monitors["u_series_rolled"] = bool(roll_decision.get("rolled", False))
        monitors["u_series_seconds_to_boundary"] = round(
            float(roll_decision.get("seconds_to_boundary", 0.0)),
            3,
        )
        monitors["u_series_cancel_required"] = bool(roll_decision.get("cancel_required", False))
    if guardrail_reasons:
        monitors["u_series_guardrails"] = tuple(guardrail_reasons)
        monitors["u_series_guardrail_blocked"] = True
    elif series_obj is not None and series_obj.ticker.upper() in _U_SERIES:
        monitors.setdefault("u_series_guardrail_blocked", False)
    if series_obj is not None:
        monitors.setdefault("series", series_obj.ticker.upper())
    monitors.setdefault("model_version", model_version)
    if range_structure_summary:
        monitors["range_ab_structures"] = tuple(range_structure_summary)
        monitors["range_ab_sigma"] = round(range_structure_sigma, 6)
    throttle_snapshot = replacement_throttle.snapshot() if replacement_throttle is not None else {}
    if throttle_snapshot:
        monitors["replacement_throttle"] = throttle_snapshot
    if mispricing_records:
        monitors["max_prob_sum_gap"] = max(record["prob_sum_gap"] for record in mispricing_records)
    if book_snapshot_failures:
        monitors["orderbook_snapshot_failures"] = book_snapshot_failures
    if pilot_config is not None:
        monitors["pilot_mode"] = True
        monitors.setdefault("pilot_max_unique_bins", pilot_config.max_unique_bins)
        monitors.setdefault("pilot_max_contracts", pilot_config.max_contracts_per_order)
        if pilot_trimmed_bins:
            monitors["pilot_bins_trimmed"] = pilot_trimmed_bins
    if bin_constraints is not None and bin_constraints.has_rules:
        summary = bin_constraints.summary()
        monitors["ev_honesty_constraints"] = summary
        applied_count = summary.get("applied")
        dropped_count = summary.get("dropped")
        if applied_count:
            monitors["ev_honesty_bins_adjusted"] = applied_count
        if dropped_count:
            monitors["ev_honesty_bins_blocked"] = dropped_count
    if 0.0 < ev_honesty_shrink < 1.0:
        monitors.setdefault("ev_honesty_shrink_factor", ev_honesty_shrink)
    if var_limiter is not None:
        monitors["var_family_exposure"] = var_limiter.snapshot()
    if correlation_guard is not None:
        monitors["correlation_var"] = correlation_guard.snapshot()
    if sniper:
        taker_count = sum(
            1
            for proposal in proposals
            if str((proposal.metadata or {}).get("liquidity") or "").lower() == "taker"
        )
        monitors["sniper_enabled"] = True
        monitors["sniper_threshold"] = round(float(sniper_threshold), 4)
        monitors["sniper_proposals"] = taker_count
    else:
        monitors.setdefault("sniper_enabled", False)
    roll_payload: dict[str, object] | None = None
    if roll_decision is not None:
        now_et_value = roll_decision.get("now_et")
        roll_payload = {
            "current_hour": int(roll_decision.get("current_hour", 0)),
            "target_hour": int(roll_decision.get("target_hour", 0)),
            "current_hour_label": _format_hour_label(int(roll_decision.get("current_hour", 0))),
            "target_hour_label": _format_hour_label(int(roll_decision.get("target_hour", 0))),
            "rolled": bool(roll_decision.get("rolled", False)),
            "seconds_to_boundary": round(float(roll_decision.get("seconds_to_boundary", 0.0)), 3),
            "cancel_required": bool(roll_decision.get("cancel_required", False)),
            "now_et": now_et_value.isoformat() if isinstance(now_et_value, datetime) else None,
        }
        if guardrail_reasons:
            roll_payload["guardrails"] = tuple(guardrail_reasons)
    return ScanOutcome(
        proposals=proposals,
        monitors=monitors,
        cdf_diffs=cdf_diffs,
        mispricings=mispricing_records,
        series=series_obj,
        events=list(events_to_scan),
        markets=all_markets,
        model_metadata=model_metadata,
        books_at_scan=books_at_scan,
        book_snapshot_started_at=book_snapshot_started_at,
        book_snapshot_completed_at=book_snapshot_completed_at,
        roll_info=roll_payload,
    )


def _strategy_pmf_for_series(
    *,
    series: str,
    strikes: list[float],
    fixtures_dir: Path,
    override: str,
    offline: bool,
    model_version: str = "v15",
    orderbook_imbalance: float | None = None,
    event_timestamp: datetime | None = None,
    target_time: time | None = None,
) -> tuple[list[LadderBinProbability], dict[str, object]]:
    pick = override.lower()
    ticker = series.upper()
    if pick == "auto":
        pick = ticker.lower()

    version = (model_version or "v15").lower()
    if version not in {"v0", "v15"}:
        version = "v15"
    metadata: dict[str, object] = {"model_version": version}
    pmf_values: list[LadderBinProbability] | None = None

    if pick == "cpi" and ticker == "CPI":
        pmf_values = cpi.strategy_pmf(
            strikes,
            fixtures_dir=fixtures_dir,
            offline=offline,
            model_version=version,
        )
        if version == "v15":
            config = cpi_strategy.load_v15_config()
            metadata["component_weights"] = {
                "gas": config.component_weights.gas,
                "shelter": config.component_weights.shelter,
                "autos": config.component_weights.autos,
            }
    elif pick in {"claims", "jobless"} and ticker == "CLAIMS":
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
            pmf_values = claims_strategy.pmf_v15(strikes, inputs=inputs)
        else:
            pmf_values = claims_strategy.pmf(strikes, inputs=inputs)
    elif pick in {"tney", "rates"} and ticker == "TNEY":
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
            latest = {}
        macro_lookup = _macro_calendar_lookup(
            history,
            offline=offline,
            fixtures_dir=fixtures_dir,
        )
        date_key = _normalize_macro_date(latest.get("date")) if history else None
        macro_dummies = macro_lookup.get(date_key, {}) if date_key is not None else {}
        imbalance_value = orderbook_imbalance
        if imbalance_value is None:
            imbalance_entry = kalshi_ws.load_latest_imbalance(ticker)
            if imbalance_entry is not None:
                imbalance_value = float(imbalance_entry[0])
        timestamp = event_timestamp if event_timestamp is not None else datetime.now(tz=UTC)
        inputs = teny_strategy.TenYInputs(
            prior_close=prior_close,
            macro_shock=macro_shock,
            trailing_history=trailing,
            macro_shock_dummies=macro_dummies,
            orderbook_imbalance=imbalance_value,
            event_timestamp=timestamp,
        )
        if imbalance_value is not None:
            metadata.setdefault("teny_orderbook_imbalance", float(imbalance_value))
        if version == "v15":
            pmf_values = teny_strategy.pmf_v15(strikes, inputs=inputs)
        else:
            pmf_values = teny_strategy.pmf(strikes, inputs=inputs)
    elif ticker in {"INXU", "NASDAQ100U"}:
        meta = resolve_index_series(ticker)
        snapshot = _load_index_snapshot(meta.polygon_ticker, offline=offline, fixtures_dir=fixtures_dir)
        now = event_timestamp if event_timestamp is not None else datetime.now(tz=UTC)
        hourly_target_time = target_time or _default_hourly_target(now)
        minutes_to_target = _minutes_to_target(now, hourly_target_time)
        event_tags = calendar_tags_for(now)
        inputs = index_strategy.HourlyInputs(
            series=ticker,
            current_price=_resolve_index_price(snapshot),
            minutes_to_target=minutes_to_target,
            prev_close=snapshot.previous_close,
            event_tags=event_tags,
            target_hour=hourly_target_time.hour,
        )
        pmf_values = index_strategy.hourly_pmf(strikes, inputs=inputs)
        metadata.update(
            {
                "polygon_ticker": meta.polygon_ticker,
                "minutes_to_target": minutes_to_target,
                "snapshot_last_price": snapshot.last_price,
                "snapshot_timestamp": snapshot.timestamp.isoformat() if snapshot.timestamp else None,
                "target_time_et": hourly_target_time.isoformat(),
                "event_tags": event_tags,
            }
        )
    elif ticker in {"INX", "NASDAQ100"}:
        meta = resolve_index_series(ticker)
        snapshot = _load_index_snapshot(meta.polygon_ticker, offline=offline, fixtures_dir=fixtures_dir)
        now = event_timestamp if event_timestamp is not None else datetime.now(tz=UTC)
        minutes_to_target = _minutes_to_target(now, _TARGET_CLOSE)
        event_tags = calendar_tags_for(now)
        inputs = index_strategy.CloseInputs(
            series=ticker,
            current_price=_resolve_index_price(snapshot),
            minutes_to_close=minutes_to_target,
            event_tags=event_tags,
        )
        pmf_values = index_strategy.close_pmf(strikes, inputs=inputs)
        metadata.update(
            {
                "polygon_ticker": meta.polygon_ticker,
                "minutes_to_target": minutes_to_target,
                "snapshot_last_price": snapshot.last_price,
                "snapshot_timestamp": snapshot.timestamp.isoformat() if snapshot.timestamp else None,
                "event_tags": event_tags,
            }
        )
    elif pick in {"weather"} and ticker == "WEATHER":
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
        pmf_values = weather_strategy.pmf(strikes, inputs=inputs)
    else:
        pmf_values = weather_strategy.pmf(strikes, weather_strategy.WeatherInputs(forecast_high=70.0))

    if pmf_values is None:
        raise ValueError(f"No strategy PMF implemented for series {series}")

    aligned = align_pmf_to_strikes(pmf_values, strikes)
    return aligned, metadata


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


@lru_cache(maxsize=1)
def _polygon_client_cached() -> PolygonIndicesClient:
    return PolygonIndicesClient()


def _load_index_snapshot(
    symbol: str,
    *,
    offline: bool,
    fixtures_dir: Path,
) -> IndexSnapshot:
    fixture_path = _polygon_fixture_path(symbol, fixtures_dir)
    if offline and fixture_path.exists():
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        return _snapshot_from_payload(symbol, payload)
    try:
        client = _polygon_client_cached()
        return client.fetch_snapshot(symbol)
    except PolygonAPIError:
        if fixture_path.exists():
            payload = json.loads(fixture_path.read_text(encoding="utf-8"))
            return _snapshot_from_payload(symbol, payload)
        raise


def _polygon_fixture_path(symbol: str, fixtures_dir: Path) -> Path:
    safe_symbol = symbol.replace(":", "_")
    return fixtures_dir / _POLYGON_FIXTURE_DIR / f"{safe_symbol}_snapshot.json"


def _snapshot_from_payload(symbol: str, payload: dict[str, object]) -> IndexSnapshot:
    return IndexSnapshot(
        ticker=str(payload.get("ticker") or symbol),
        last_price=_maybe_float(payload.get("last_price")),
        change=_maybe_float(payload.get("change")),
        change_percent=_maybe_float(payload.get("change_percent")),
        previous_close=_maybe_float(payload.get("previous_close")),
        timestamp=_parse_timestamp(payload.get("timestamp")),
    )


def _maybe_float(value: object | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_timestamp(value: object | None) -> datetime | None:  # noqa: PLR0911
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=UTC)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed
    return None


def _minutes_to_target(now_utc: datetime, target: time) -> int:
    now_et = now_utc.astimezone(_ET_ZONE)
    target_dt = datetime.combine(now_et.date(), target, tzinfo=_ET_ZONE)
    if target_dt <= now_et:
        target_dt = target_dt + timedelta(days=1)
    delta_minutes = int((target_dt - now_et).total_seconds() // 60)
    return max(delta_minutes, 0)


def _resolve_index_price(snapshot: IndexSnapshot) -> float:
    if snapshot.last_price is not None and snapshot.last_price > 0:
        return float(snapshot.last_price)
    if snapshot.previous_close is not None and snapshot.previous_close > 0:
        return float(snapshot.previous_close)
    return 0.0


def _macro_calendar_lookup(
    history: list[dict[str, object]] | None,
    *,
    offline: bool,
    fixtures_dir: Path,
) -> dict[str, dict[str, float]]:
    lookup: dict[str, dict[str, float]] = {}
    if not history:
        return lookup
    dates: list[date] = []
    for row in history:
        raw_date = row.get("date")
        if isinstance(raw_date, str):
            try:
                dates.append(date.fromisoformat(raw_date))
            except ValueError:
                continue
    if dates:
        start_date, end_date = min(dates), max(dates)
        try:
            macro_calendar.emit_day_dummies(
                start_date,
                end_date,
                offline=offline,
                fixtures_dir=fixtures_dir if offline else None,
            )
        except Exception:  # pragma: no cover - macro calendar is optional
            return lookup
    path = macro_calendar.DEFAULT_OUTPUT
    if path.exists():
        try:
            frame = pl.read_parquet(path)
        except Exception:  # pragma: no cover - corrupt parquet
            frame = None
        if frame is not None and not frame.is_empty():
            columns = [name for name in frame.columns if name != "date"]
            for row in frame.iter_rows(named=True):
                key = _normalize_macro_date(row.get("date"))
                if key is None:
                    continue
                lookup[key] = {
                    _strip_dummy_prefix(column): float(row.get(column) or 0.0)
                    for column in columns
                }
    return lookup


def _normalize_macro_date(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return date.fromisoformat(value).isoformat()
        except ValueError:
            return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return None


def _strip_dummy_prefix(column: str) -> str:
    return column[3:] if column.startswith("is_") else column


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


def _limit_proposals_for_pilot(
    proposals: list[Proposal],
    *,
    max_unique_bins: int,
) -> tuple[list[Proposal], int]:
    if max_unique_bins <= 0 or len(proposals) <= max_unique_bins:
        return proposals, 0
    ranked = sorted(
        proposals,
        key=lambda item: item.maker_ev_per_contract,
        reverse=True,
    )
    allowed_keys = {
        (proposal.market_id, proposal.side, proposal.strike) for proposal in ranked[:max_unique_bins]
    }
    filtered: list[Proposal] = [
        proposal for proposal in proposals if (proposal.market_id, proposal.side, proposal.strike) in allowed_keys
    ]
    trimmed = len(proposals) - len(filtered)
    return filtered, trimmed


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
    daily_budget: LossBudget,
    weekly_budget: LossBudget,
    limit_checker: ProposalLimitChecker,
    series_ticker: str,
    ev_shrink: float,
    bin_constraints: BinConstraintResolver | None = None,
    var_limiter: var_index.FamilyVarLimiter | None = None,
    correlation_guard: CorrelationAwareLimiter | None = None,
    quote_optimizer: QuoteOptimizer | None = None,
    orderbook: Orderbook | None = None,
    freshness_ms: float | None = None,
    replacement_throttle: quote_microprice.ReplacementThrottle | None = None,
    sniper: bool = False,
    sniper_threshold: float = 0.05,
    now_ts: datetime | None = None,
) -> list[Proposal]:
    proposals: list[Proposal] = []
    uncertainty_penalty = max(0.0, uncertainty_penalty)
    ob_imbalance_penalty = max(0.0, ob_imbalance_penalty)
    series_upper = series_ticker.upper()
    apply_shrink = 0.0 < ev_shrink < 1.0
    microprice_signal = quote_microprice.compute_signal(orderbook)
    microprice_value = microprice_signal.microprice
    best_bid_price = microprice_signal.best_bid
    best_ask_price = microprice_signal.best_ask
    best_bid_size = None
    best_ask_size = None
    if orderbook is not None:
        try:
            if orderbook.bids:
                best_bid_size = float(orderbook.bids[0].get("size", 0.0))
            if orderbook.asks:
                best_ask_size = float(orderbook.asks[0].get("size", 0.0))
        except Exception:
            best_bid_size = None
            best_ask_size = None

    for index, rung in enumerate(rungs):
        if allowed_indices is not None and index not in allowed_indices:
            continue
        yes_price = rung.yes_price
        liquidity = Liquidity.MAKER
        event_probability = strategy_survival[index]
        survival_market = market_survival[index]
        raw_fraction: float | None = None
        truncated_fraction: float | None = None
        scaled_fraction: float | None = None
        uncertainty_metric = max(0.0, min(1.0, 1.0 - abs(event_probability - 0.5) * 2.0))
        imbalance_metric = max(0.0, min(1.0, abs(survival_market - 0.5) * 2.0))
        sniper_used = False
        sniper_gap: float | None = None
        sniper_source: str | None = None
        top_size: float | None = None
        if sniper and orderbook is not None:
            ask_gap = None
            bid_gap = None
            if best_ask_price is not None:
                ask_gap = event_probability - best_ask_price
            if best_bid_price is not None:
                bid_gap = best_bid_price - event_probability
            if ask_gap is not None and ask_gap >= sniper_threshold and best_ask_price is not None:
                yes_price = float(best_ask_price)
                liquidity = Liquidity.TAKER
                sniper_used = True
                sniper_gap = float(ask_gap)
                sniper_source = "ask"
                top_size = best_ask_size
            elif bid_gap is not None and bid_gap >= sniper_threshold and best_bid_price is not None:
                yes_price = float(best_bid_price)
                liquidity = Liquidity.TAKER
                sniper_used = True
                sniper_gap = float(bid_gap)
                sniper_source = "bid"
                top_size = best_bid_size

        per_contract_raw = expected_value_summary(
            contracts=1,
            yes_price=yes_price,
            event_probability=event_probability,
            schedule=DEFAULT_FEE_SCHEDULE,
            series=series_ticker,
            market_name=market_ticker,
        )
        per_contract_eval = dict(per_contract_raw)
        if apply_shrink:
            per_contract_eval["maker_yes"] *= ev_shrink
            per_contract_eval["maker_no"] *= ev_shrink
        throttle_key: str | None = None
        penalty = 0.0
        if sniper_used:
            best_side = OrderSide.YES if sniper_source == "ask" else OrderSide.NO
            maker_key = "maker_yes" if best_side is OrderSide.YES else "maker_no"
            taker_key = "taker_yes" if best_side is OrderSide.YES else "taker_no"
            best_ev = per_contract_eval[taker_key]
            effective_min_ev = min_ev
        else:
            best_side, best_ev = _choose_side(per_contract_eval, maker_only=maker_only)
            maker_key = "maker_yes" if best_side is OrderSide.YES else "maker_no"
            taker_key = "taker_yes" if best_side is OrderSide.YES else "taker_no"
            if quote_optimizer is not None:
                quote_context = QuoteContext(
                    market_id=market_id,
                    strike=rung.strike,
                    side=best_side,
                    pmf_probability=event_probability,
                    market_probability=survival_market,
                    microprice=microprice_value,
                    best_bid=best_bid_price,
                    best_ask=best_ask_price,
                    freshness_ms=freshness_ms,
                    maker_ev_per_contract=per_contract_eval[maker_key],
                )
                penalty = quote_optimizer.penalty(quote_context)
                throttle_key = quote_optimizer.key_for_order(market_id, rung.strike, best_side)
            effective_min_ev = min_ev + penalty
        if best_ev < effective_min_ev:
            continue
        if (
            throttle_key
            and replacement_throttle is not None
            and liquidity is Liquidity.MAKER
            and replacement_throttle.should_block(throttle_key, now=now_ts)
        ):
            continue

        order_id = f"{market_ticker}:{rung.strike}"
        max_loss_single = max_loss_for_order(
            OrderProposal(
                strike_id=order_id,
                yes_price=yes_price,
                contracts=1,
                side=best_side,
                liquidity=liquidity,
                market_name=market_ticker,
                series=series_ticker,
            )
        )
        strike_cap = pal_guard.policy.limit_for_strike(order_id)
        remaining_limit = strike_cap - pal_guard.exposure_for(order_id)
        if max_loss_single <= 0 or remaining_limit <= 0:
            continue
        max_contracts = min(int(remaining_limit // max_loss_single), contracts)
        if top_size is not None and top_size > 0:
            try:
                max_contracts = min(max_contracts, int(top_size))
            except (TypeError, ValueError):
                pass
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

        daily_before = daily_budget.remaining
        weekly_before = weekly_budget.remaining
        total_max_loss = max_loss_single * contract_count
        if total_max_loss <= 0:
            continue
        allowed_contracts = daily_budget.max_contracts(max_loss_single, contract_count)
        if allowed_contracts <= 0:
            continue
        if allowed_contracts < contract_count:
            contract_count = allowed_contracts
            total_max_loss = max_loss_single * contract_count

        weekly_allowed = weekly_budget.max_contracts(max_loss_single, contract_count)
        if weekly_allowed <= 0:
            continue
        if weekly_allowed < contract_count:
            contract_count = weekly_allowed
            total_max_loss = max_loss_single * contract_count

        constraint_details = None
        if bin_constraints is not None:
            adjusted_contracts, constraint_details = bin_constraints.apply(
                market_id=market_id,
                market_ticker=market_ticker,
                strike=rung.strike,
                side=best_side.name,
                contracts=contract_count,
            )
            if adjusted_contracts <= 0:
                continue
            if adjusted_contracts != contract_count:
                contract_count = adjusted_contracts
                total_max_loss = max_loss_single * contract_count

        correlation_metadata: dict[str, float] | None = None
        correlation_exposure = None
        if correlation_guard is not None:
            capped_contracts, exposure_record, metadata = correlation_guard.cap_contracts(
                series=series_ticker,
                strike=rung.strike,
                side=best_side,
                contracts=contract_count,
                probability=event_probability,
            )
            if capped_contracts <= 0:
                continue
            if capped_contracts != contract_count:
                contract_count = capped_contracts
                total_max_loss = max_loss_single * contract_count
            correlation_exposure = exposure_record
            if metadata:
                correlation_metadata = metadata

        total_ev_raw = expected_value_summary(
            contracts=contract_count,
            yes_price=yes_price,
            event_probability=event_probability,
            schedule=DEFAULT_FEE_SCHEDULE,
            series=series_ticker,
            market_name=market_ticker,
        )
        total_ev_eval = dict(total_ev_raw)
        if apply_shrink:
            total_ev_eval["maker_yes"] *= ev_shrink
            total_ev_eval["maker_no"] *= ev_shrink

        if maker_only and not sniper_used:
            total_ev_eval[taker_key] = 0.0
            per_contract_eval[taker_key] = 0.0
            total_ev_raw[taker_key] = 0.0
            per_contract_raw[taker_key] = 0.0

        total_max_loss = max_loss_single * contract_count
        if risk_manager and not risk_manager.can_accept(
            strategy=strategy_name,
            max_loss=total_max_loss,
            max_var=max_var,
        ):
            continue

        if var_limiter and not var_limiter.can_accept(series_ticker, total_max_loss):
            continue
        order = OrderProposal(
            strike_id=order_id,
            yes_price=yes_price,
            contracts=contract_count,
            side=best_side,
            liquidity=liquidity,
            market_name=market_ticker,
            series=series_ticker,
        )
        try:
            limit_checker.try_accept(order, max_loss=total_max_loss)
        except LimitViolation:
            continue
        if var_limiter:
            var_limiter.register(series_ticker, total_max_loss)
        daily_after = daily_budget.remaining
        weekly_after = weekly_budget.remaining

        proposal_metadata: dict[str, object] = {
            "sizing": {
                "kelly_raw": raw_fraction if sizing_mode == "kelly" else None,
                "kelly_truncated": truncated_fraction if sizing_mode == "kelly" else None,
                "kelly_scaled": scaled_fraction if sizing_mode == "kelly" else None,
                "uncertainty_metric": uncertainty_metric,
                "uncertainty_penalty": uncertainty_penalty,
                "ob_imbalance_metric": float(imbalance_metric),
                "ob_imbalance_penalty": float(ob_imbalance_penalty) if ob_imbalance_penalty else 0.0,
                "daily_loss_before": daily_before,
                "daily_loss_after": daily_after,
                "weekly_loss_before": weekly_before,
                "weekly_loss_after": weekly_after,
            }
        }
        if apply_shrink:
            proposal_metadata["ev_shrink"] = ev_shrink
            proposal_metadata.setdefault(
                "ev_shrunk",
                {
                    "maker_per_contract": per_contract_eval[maker_key],
                    "maker_total": total_ev_eval[maker_key],
                },
            )
        if constraint_details is not None:
            proposal_metadata["bin_constraint"] = constraint_details

        if correlation_metadata is not None:
            risk_block = proposal_metadata.setdefault("risk", {})
            risk_block["correlation"] = correlation_metadata
        if penalty > 0.0:
            risk_block = proposal_metadata.setdefault("risk", {})
            optim_meta = risk_block.setdefault("quote_optim", {})
            optim_meta["ev_penalty"] = penalty
            optim_meta["min_ev"] = effective_min_ev
        if sniper_used:
            proposal_metadata["liquidity"] = "taker"
            proposal_metadata["sniper"] = {
                "gap": round(sniper_gap, 6) if sniper_gap is not None else None,
                "source": sniper_source,
                "top_size": top_size,
            }
            book_snapshot: dict[str, object] = {}
            if best_bid_price is not None:
                book_snapshot["bid"] = {"price": best_bid_price, "size": best_bid_size}
            if best_ask_price is not None:
                book_snapshot["ask"] = {"price": best_ask_price, "size": best_ask_size}
            if book_snapshot:
                proposal_metadata["book_snapshot"] = book_snapshot
        else:
            proposal_metadata["liquidity"] = "maker"

        if best_side is OrderSide.YES:
            fee_price = yes_price
        else:
            fee_price = 1.0 - yes_price
        fee_liquidity = "taker" if liquidity is Liquidity.TAKER else "maker"
        if series_ticker.upper().startswith(tuple(fee_utils.INDEX_PREFIXES)):
            fee_details = exec_fees.fee_breakdown(
                series=series_ticker,
                price=fee_price,
                contracts=contract_count,
                liquidity=fee_liquidity,
            )
            per_contract_fee = float(fee_details["per_contract_effective"])
            total_fee = float(fee_details["per_order"])
        else:
            fee_fn = (
                DEFAULT_FEE_SCHEDULE.taker_fee
                if liquidity is Liquidity.TAKER
                else DEFAULT_FEE_SCHEDULE.maker_fee
            )
            per_contract_fee = float(
                fee_fn(
                    1,
                    fee_price,
                    series=series_ticker,
                    market_name=market_ticker,
                )
            )
            total_fee = float(
                fee_fn(
                    contract_count,
                    fee_price,
                    series=series_ticker,
                    market_name=market_ticker,
                )
            )
        ev_key = taker_key if liquidity is Liquidity.TAKER else maker_key
        proposal_metadata["ev_components"] = {
            "per_contract": {
                "gross": per_contract_raw[ev_key] + per_contract_fee,
                "net": per_contract_raw[ev_key],
                "fee": per_contract_fee,
            },
            "total": {
                "gross": total_ev_raw[ev_key] + total_fee,
                "net": total_ev_raw[ev_key],
                "fee": total_fee,
            },
            "fee_price": fee_price,
            "liquidity": fee_liquidity,
            "side": best_side.name.lower(),
        }

        if correlation_guard is not None and correlation_exposure is not None:
            correlation_guard.register(correlation_exposure)
        if (
            replacement_throttle is not None
            and throttle_key is not None
            and liquidity is Liquidity.MAKER
        ):
            replacement_throttle.record(throttle_key, now=now_ts)

        proposal = Proposal(
            market_id=market_id,
            market_ticker=market_ticker,
            strike=rung.strike,
            side=best_side.name,
            contracts=contract_count,
            maker_ev=total_ev_raw[maker_key],
            taker_ev=total_ev_raw[taker_key],
            maker_ev_per_contract=per_contract_raw[maker_key],
            taker_ev_per_contract=per_contract_raw[taker_key],
            strategy_probability=event_probability,
            market_yes_price=yes_price,
            survival_market=survival_market,
            survival_strategy=event_probability,
            max_loss=total_max_loss,
            strategy=strategy_name,
            series=series_upper,
            metadata=proposal_metadata,
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
