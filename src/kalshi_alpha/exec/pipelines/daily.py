"""Daily orchestration pipeline for ladder strategies."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from kalshi_alpha.core.execution.fillratio import FillRatioEstimator, tune_alpha
from kalshi_alpha.core.execution.slippage import SlippageModel
from kalshi_alpha.core.gates import QualityGateResult, load_quality_gate_config, run_quality_gates
from kalshi_alpha.core.kalshi_api import KalshiPublicClient, Orderbook
from kalshi_alpha.core.risk import PALGuard, PALPolicy, PortfolioRiskManager, drawdown
from kalshi_alpha.datastore import ingest as datastore_ingest
from kalshi_alpha.datastore.paths import PROC_ROOT, RAW_ROOT
from kalshi_alpha.exec.ledger import PaperLedger, simulate_fills
from kalshi_alpha.exec.pipelines.calendar import resolve_run_window
from kalshi_alpha.core.archive.scorecards import build_replay_scorecard
from kalshi_alpha.exec.reports import write_markdown_report
from kalshi_alpha.exec.gate_utils import resolve_quality_gate_config_path, write_go_no_go
from kalshi_alpha.exec.runners.scan_ladders import (
    _archive_and_replay,
    _attach_series_metadata,
    _compute_exposure_summary,
    _write_cdf_diffs,
    execute_broker,
    scan_series,
    write_proposals,
)
from kalshi_alpha.exec.state.orders import OutstandingOrdersState
from kalshi_alpha.exec.heartbeat import (
    heartbeat_stale,
    kill_switch_engaged,
    resolve_kill_switch_path,
    write_heartbeat,
)
from kalshi_alpha.strategies.claims import calibrate as calibrate_claims
from kalshi_alpha.strategies.cpi import calibrate as calibrate_cpi
from kalshi_alpha.strategies.teny import calibrate as calibrate_teny
from kalshi_alpha.strategies.weather import calibrate as calibrate_weather

ET = ZoneInfo("America/New_York")

MODE_SEQUENCE = ["pre_cpi", "pre_claims", "teny_close", "weather_cycle"]
DEFAULT_FILL_ALPHA = 0.6


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily pipeline orchestrator.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["pre_cpi", "pre_claims", "teny_close", "weather_cycle", "full"],
        help="Pipeline mode to execute.",
    )
    env_group = parser.add_mutually_exclusive_group(required=True)
    env_group.add_argument("--offline", action="store_true", help="Run with offline fixtures.")
    env_group.add_argument("--online", action="store_true", help="Run with live fetchers.")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh during ingestion.",
    )
    parser.add_argument(
        "--force-run",
        action="store_true",
        help="Bypass calendar scan windows (DRY broker only).",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Render markdown report after scanning.",
    )
    parser.add_argument(
        "--paper-ledger",
        action="store_true",
        help="Simulate paper ledger fills.",
    )
    parser.add_argument(
        "--driver-fixtures",
        default="tests/fixtures",
        help="Driver fixture root for offline calibrations.",
    )
    parser.add_argument(
        "--scanner-fixtures",
        default="tests/data_fixtures",
        help="Kalshi public API fixture root when offline.",
    )
    parser.add_argument(
        "--kelly-cap",
        type=float,
        default=0.25,
        help="Kelly truncation cap for scanner sizing.",
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
        "--broker",
        choices=["dry", "live"],
        default="dry",
        help="Broker adapter to use when generating orders.",
    )
    parser.add_argument(
        "--allow-no-go",
        action="store_true",
        help="Continue scanning even if quality gates block execution.",
    )
    parser.add_argument(
        "--mispricing-only",
        action="store_true",
        help="Restrict generated proposals to detected mispricing bins.",
    )
    parser.add_argument(
        "--max-legs",
        type=int,
        default=4,
        help="Maximum legs for mispricing spread detection.",
    )
    parser.add_argument(
        "--prob-sum-gap-threshold",
        type=float,
        default=0.0,
        help="Probability mass gap threshold for mispricing logging.",
    )
    parser.add_argument(
        "--model-version",
        choices=["v0", "v15"],
        default="v15",
        help="Strategy model version to run for CPI/Claims/TENY.",
    )
    parser.add_argument(
        "--kill-switch-file",
        help="Path to kill-switch sentinel file (default: data/proc/state/kill_switch).",
    )
    parser.add_argument(
        "--when",
        type=_parse_date,
        help="Override target date (YYYY-MM-DD) for calendar scheduling.",
    )
    parser.add_argument(
        "--window-et",
        help="Human-readable scan window (ET) forwarded from week/today orchestrators.",
    )
    parser.add_argument(
        "--daily-loss-cap",
        type=float,
        help="Daily expected-loss drawdown cap (USD).",
    )
    parser.add_argument(
        "--weekly-loss-cap",
        type=float,
        help="Weekly expected-loss drawdown cap (USD).",
    )
    return parser.parse_args(argv)


def _resolve_fill_alpha_value(fill_alpha_arg: object, series: str) -> tuple[float, bool]:
    if fill_alpha_arg is None:
        return DEFAULT_FILL_ALPHA, False
    if isinstance(fill_alpha_arg, str):
        raw = fill_alpha_arg.strip().lower()
        if raw == "auto":
            tuned = tune_alpha(series, RAW_ROOT / "kalshi")
            if tuned is not None:
                return float(tuned), True
            return DEFAULT_FILL_ALPHA, True
        try:
            return float(raw), False
        except ValueError:
            return DEFAULT_FILL_ALPHA, False
    try:
        return float(fill_alpha_arg), False
    except (TypeError, ValueError):
        return DEFAULT_FILL_ALPHA, False


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    modes = MODE_SEQUENCE if args.mode == "full" else [args.mode]
    for mode in modes:
        run_mode(mode, args)


def run_mode(mode: str, args: argparse.Namespace) -> None:
    now_utc = datetime.now(tz=UTC)
    log: dict[str, object] = {
        "mode": mode,
        "timestamp": now_utc.isoformat(),
        "offline": args.offline,
        "steps": [],
        "model_version": args.model_version,
    }
    force_run_requested = bool(getattr(args, "force_run", False))
    force_run_enabled = bool(force_run_requested and getattr(args, "broker", "dry") == "dry")
    if force_run_requested and not force_run_enabled:
        log.setdefault("warnings", []).append("force_run_requires_dry_broker")

    def _refresh_heartbeat(stage: str, extra: dict[str, object] | None = None) -> None:
        payload: dict[str, object] = {"stage": stage, "mode": mode, "force_run": force_run_enabled}
        if extra:
            payload.update(extra)
        window_et = getattr(args, "window_et", None)
        if window_et:
            payload.setdefault("window_et", window_et)
        try:
            outstanding_summary = OutstandingOrdersState.load().summary()
        except Exception:  # pragma: no cover - defensive
            outstanding_summary = None
        if outstanding_summary is not None:
            payload.setdefault("outstanding", outstanding_summary)
        write_heartbeat(mode=f"daily:{mode}", extra=payload)

    _refresh_heartbeat("start")
    try:
        run_ingest(args, log)
        _refresh_heartbeat("post_ingest")
        run_calibrations(args, log, heartbeat_cb=_refresh_heartbeat)
        _refresh_heartbeat("post_calibrations")
        series = resolve_series(mode)
        fill_alpha_value = DEFAULT_FILL_ALPHA
        fill_alpha_auto = False
        extra_monitors: dict[str, object] = {}
        if series is not None:
            fill_alpha_value, fill_alpha_auto = _resolve_fill_alpha_value(args.fill_alpha, series)
            log.setdefault("fill_alpha", {})[series] = fill_alpha_value
            key = "fill_alpha_auto" if fill_alpha_auto else "fill_alpha"
            extra_monitors[key] = fill_alpha_value
        if force_run_enabled:
            extra_monitors["force_run"] = True
        gate_result = run_quality_gate_step(args, now_utc, log, monitors=extra_monitors)
        if not gate_result.go and not getattr(args, "allow_no_go", False):
            raise SystemExit(1)
        target_date: date = args.when if args.when else now_utc.astimezone(ET).date()
        run_window = resolve_run_window(mode=mode, target_date=target_date, now=now_utc, proc_root=PROC_ROOT)
        log["window"] = run_window.to_dict(now_utc)

        scan_allowed = run_window.scan_allowed(now_utc)
        if not scan_allowed and not force_run_enabled:
            reason = "outside window"
            if run_window.notes:
                reason = ",".join(run_window.notes)
            log.setdefault("scan_notes", {})[mode] = reason
        else:
            if series is None:
                log.setdefault("scan_notes", {})[mode] = "series_not_supported"
            else:
                if not scan_allowed and force_run_enabled:
                    log.setdefault("scan_notes", {})[mode] = "force_run_dry"
                _refresh_heartbeat(
                    "pre_scan",
                    {
                        "series": series,
                        "run_window": run_window.to_dict(now_utc),
                        "force_run": force_run_enabled,
                        "outside_window": not scan_allowed,
                        "when": (args.when.isoformat() if getattr(args, "when", None) else None),
                    },
                )
                run_scan(
                    mode,
                    args,
                    log,
                    series=series,
                    fill_alpha_value=fill_alpha_value,
                    fill_alpha_auto=fill_alpha_auto,
                )
    except Exception as exc:  # pragma: no cover - orchestration guard
        log.setdefault("errors", []).append(str(exc))
        raise
    finally:
        write_log(mode, log, now_utc)


def run_ingest(args: argparse.Namespace, log: dict[str, object]) -> None:
    ingest_args: list[str] = ["--all"]
    if args.offline:
        ingest_args.append("--offline")
        ingest_args.extend(["--fixtures", args.driver_fixtures])
    else:
        ingest_args.append("--online")
    if args.force_refresh:
        ingest_args.append("--force-refresh")
    run_step(
        name="ingest",
        log=log,
        func=lambda: datastore_ingest.main(ingest_args),
        metadata={"args": ingest_args},
    )


def run_calibrations(
    args: argparse.Namespace,
    log: dict[str, object],
    heartbeat_cb: Callable[[str, dict[str, object] | None], None] | None = None,
) -> None:
    fixtures_root = Path(args.driver_fixtures)

    def load_history(name: str) -> list[dict[str, object]] | None:
        path = fixtures_root / name / "history.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))["history"]
        except Exception:  # pragma: no cover - malformed fixture
            return None

    histories = {
        "cpi": load_history("cpi"),
        "claims": load_history("claims"),
        "teny": load_history("teny"),
        "weather": load_history("weather"),
    }

    def calibrate_wrapper(name: str, func, history) -> None:
        if not history:
            log.setdefault("calibration_skipped", {})[name] = "missing_history"
            if heartbeat_cb:
                heartbeat_cb("post_calibrate", {"calibration": name, "status": "skipped"})
            return
        run_step(name=f"calibrate_{name}", log=log, func=lambda: func(history))
        if heartbeat_cb:
            heartbeat_cb("post_calibrate", {"calibration": name})

    calibrate_wrapper("cpi", calibrate_cpi, histories["cpi"])
    calibrate_wrapper("claims", calibrate_claims, histories["claims"])
    calibrate_wrapper("teny", calibrate_teny, histories["teny"])
    calibrate_wrapper("weather", calibrate_weather, histories["weather"])


def run_quality_gate_step(
    args: argparse.Namespace,
    now_utc: datetime,
    log: dict[str, object],
    *,
    monitors: dict[str, float] | None = None,
) -> QualityGateResult:
    config = load_quality_gate_config(resolve_quality_gate_config_path())
    result = run_quality_gates(
        config=config,
        now=now_utc,
        proc_root=PROC_ROOT,
        raw_root=RAW_ROOT,
        monitors=monitors or {},
    )
    drawdown_status = drawdown.check_limits(
        getattr(args, "daily_loss_cap", None),
        getattr(args, "weekly_loss_cap", None),
        now=now_utc,
    )

    combined_reasons = list(result.reasons)
    details = dict(result.details)
    if drawdown_status.metrics:
        details.setdefault("drawdown", drawdown_status.metrics)
    go_flag = result.go and drawdown_status.ok
    if not drawdown_status.ok:
        combined_reasons.extend(drawdown_status.reasons)

    kill_switch_path = resolve_kill_switch_path(getattr(args, "kill_switch_file", None))
    if kill_switch_engaged(kill_switch_path):
        go_flag = False
        combined_reasons.append("kill_switch_engaged")
        details["kill_switch_path"] = kill_switch_path.as_posix()
        OutstandingOrdersState.load().mark_cancel_all(
            "kill_switch_engaged",
            modes=[(getattr(args, "broker", "dry") or "dry")],
        )

    stale, heartbeat_payload = heartbeat_stale(threshold=timedelta(minutes=5))
    if stale:
        go_flag = False
        combined_reasons.append("heartbeat_stale")
        if heartbeat_payload:
            details.setdefault("heartbeat", heartbeat_payload)

    combined = QualityGateResult(go=go_flag, reasons=combined_reasons, details=details)

    log_entry = {
        "go": combined.go,
        "reasons": combined.reasons,
        "details": combined.details,
    }
    if monitors:
        log_entry["monitors"] = monitors
    log["quality_gates"] = log_entry
    write_go_no_go(combined)
    return combined


def _write_latest_manifest(manifest_path: Path | None) -> None:
    artifacts_dir = Path("reports/_artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    marker = artifacts_dir / "latest_manifest.txt"
    if manifest_path is None:
        if marker.exists():
            marker.unlink()
        return
    marker.write_text(manifest_path.as_posix(), encoding="utf-8")


def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:  # pragma: no cover - surfaced via argparse
        raise argparse.ArgumentTypeError(f"Invalid date for --when: {value}") from exc


def run_scan(
    mode: str,
    args: argparse.Namespace,
    log: dict[str, object],
    *,
    series: str,
    fill_alpha_value: float,
    fill_alpha_auto: bool,
) -> None:
    fixtures_root = Path(args.scanner_fixtures)
    driver_fixtures = Path(args.driver_fixtures)
    offline_mode = args.offline or not args.online
    force_run_enabled = bool(getattr(args, "force_run", False) and getattr(args, "broker", "dry") == "dry")
    client = KalshiPublicClient(offline_dir=fixtures_root / "kalshi", use_offline=offline_mode)

    pal_policy_path = Path("configs/pal_policy.yaml")
    if not pal_policy_path.exists():
        pal_policy_path = Path("configs/pal_policy.example.yaml")
    pal_policy = PALPolicy.from_yaml(pal_policy_path)
    pal_guard = PALGuard(pal_policy)

    risk_manager: PortfolioRiskManager | None = None
    max_var = None
    strategy_name = "auto"
    write_heartbeat(
        mode=f"daily:{mode}",
        extra={
            "stage": "scan_start",
            "series": series,
            "outstanding": OutstandingOrdersState.load().summary(),
            "force_run": force_run_enabled,
            "window_et": getattr(args, "window_et", None),
        },
    )
    outcome = scan_series(
        series=series,
        client=client,
        min_ev=0.01,
        contracts=10,
        pal_guard=pal_guard,
        driver_fixtures=driver_fixtures,
        strategy_name=strategy_name,
        maker_only=True,
        allow_tails=False,
        risk_manager=risk_manager,
        max_var=max_var,
        offline=offline_mode,
        sizing_mode="kelly",
        kelly_cap=args.kelly_cap,
        daily_loss_cap=args.daily_loss_cap,
        mispricing_only=args.mispricing_only,
        max_legs=args.max_legs,
        prob_sum_gap_threshold=args.prob_sum_gap_threshold,
        model_version=args.model_version,
    )

    proposals = outcome.proposals
    monitors = outcome.monitors
    if fill_alpha_auto:
        monitors["fill_alpha_auto"] = fill_alpha_value
    else:
        monitors.setdefault("fill_alpha", fill_alpha_value)
    monitors.setdefault("model_version", args.model_version)
    exposure_summary = _compute_exposure_summary(proposals)
    cdf_path = _write_cdf_diffs(outcome.cdf_diffs)
    should_archive = args.report or args.paper_ledger

    orderbook_ids: set[str] = set()
    if args.paper_ledger or args.report:
        orderbook_ids.update({proposal.market_id for proposal in proposals})
    if should_archive:
        orderbook_ids.update({market.id for market in outcome.markets})

    orderbooks: dict[str, Orderbook] = {}
    for market_id in sorted(orderbook_ids):
        try:
            orderbooks[market_id] = client.get_orderbook(market_id)
        except Exception:  # pragma: no cover
            continue

    ledger: PaperLedger | None = None
    if proposals and (args.paper_ledger or args.report):
        ledger_books = {
            proposal.market_id: orderbooks[proposal.market_id]
            for proposal in proposals
            if proposal.market_id in orderbooks
        }
        estimator = FillRatioEstimator(fill_alpha_value) if fill_alpha_value is not None else None
        event_lookup: dict[str, str] = {}
        if outcome.events and outcome.markets:
            event_tickers = {event.id: event.ticker for event in outcome.events}
            for market in outcome.markets:
                label = event_tickers.get(market.event_id) or market.ticker
                event_lookup[market.id] = label
        slippage_mode = (args.slippage_mode or "top").lower()
        impact_cap_arg = args.impact_cap
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
            ledger_books,
            fill_estimator=estimator,
            ledger_series=series,
            market_event_lookup=event_lookup,
            mode=slippage_mode,
            slippage_model=slippage_model,
        )

    if ledger:
        drawdown.record_pnl(ledger.total_expected_pnl())

    proposals_path = write_proposals(series=series, proposals=proposals, output_dir=Path("exec/proposals"))
    manifest_path: Path | None = None
    scorecard_summary_path: Path | None = None
    scorecard_cdf_path: Path | None = None
    scorecard_records: list[dict[str, object]] | None = None
    if should_archive and outcome.series is not None:
        manifest_path = _archive_and_replay(
            client=client,
            series=outcome.series,
            events=outcome.events,
            markets=outcome.markets,
            orderbooks=orderbooks,
            proposals_path=proposals_path,
            driver_fixtures=driver_fixtures,
            scanner_fixtures=fixtures_root,
            model_metadata=outcome.model_metadata,
        )
        if manifest_path:
            try:
                scorecard = build_replay_scorecard(
                    manifest_path=manifest_path,
                    model_version=args.model_version,
                    driver_fixtures=driver_fixtures,
                )
                scorecard_dir = Path("reports/_artifacts/scorecards")
                scorecard_dir.mkdir(parents=True, exist_ok=True)
                scorecard_summary_path = scorecard_dir / f"{series.upper()}_summary.parquet"
                scorecard.summary.write_parquet(scorecard_summary_path)
                scorecard_cdf_path = scorecard_dir / f"{series.upper()}_cdf.parquet"
                scorecard.cdf_deltas.write_parquet(scorecard_cdf_path)
                scorecard_records = scorecard.summary.sort("mean_abs_cdf_delta", descending=True).to_dicts()
            except Exception:  # pragma: no cover - scoreboard best-effort
                scorecard_summary_path = None
                scorecard_cdf_path = None
                scorecard_records = None

    _write_latest_manifest(manifest_path)

    if ledger and (args.paper_ledger or args.report):
        ledger.write_artifacts(Path("reports/_artifacts"), manifest_path=manifest_path)

    if proposals:
        _attach_series_metadata(
            proposals=proposals,
            series=series,
            driver_fixtures=driver_fixtures,
            offline=offline_mode,
        )

    scorecard_summary_section = (
        scorecard_records
        if scorecard_records is not None
        else ([] if manifest_path else None)
    )

    broker_status = None
    if proposals:
        try:
            broker_status = execute_broker(
                broker_mode=getattr(args, "broker", "dry"),
                proposals=proposals,
                args=args,
                monitors=outcome.monitors,
                quiet=False,
                go_status=log.get("quality_gates", {}).get("go", True),
            )
        except RuntimeError as exc:
            broker_status = {"mode": getattr(args, "broker", "dry"), "orders_recorded": 0, "error": str(exc)}
            log.setdefault("broker", broker_status)

    outstanding_summary = OutstandingOrdersState.load().summary()
    pilot_metadata = {
        "mode": getattr(args, "broker", "dry"),
        "kelly_cap": getattr(args, "kelly_cap", None),
        "max_var": getattr(args, "max_var", None),
        "fill_alpha": fill_alpha_value,
        "outstanding_total": sum(outstanding_summary.values()),
        "force_run": force_run_enabled,
    }
    window_et = getattr(args, "window_et", None)
    if window_et:
        pilot_metadata["window_et"] = window_et
    report_path = None
    if args.report:
        report_path = write_markdown_report(
            series=series,
            proposals=proposals,
            ledger=ledger,
            output_dir=Path("reports") / series.upper(),
            monitors=monitors,
            exposure_summary=exposure_summary,
            manifest_path=manifest_path,
            go_status=log.get("quality_gates", {}).get("go", True),
            fill_alpha=fill_alpha_value,
            mispricings=outcome.mispricings,
            model_metadata=outcome.model_metadata,
            scorecard_summary=scorecard_summary_section,
            outstanding_summary=outstanding_summary,
            pilot_metadata=pilot_metadata,
        )

    write_heartbeat(
        mode=f"daily:{mode}",
        monitors=monitors,
        extra={
            "outstanding": outstanding_summary,
            "broker": broker_status,
            "series": series,
            "force_run": force_run_enabled,
            "window_et": window_et,
        },
    )


    log.setdefault("scan_results", {})[mode] = {
        "proposals": len(proposals),
        "proposals_path": str(proposals_path),
        "report_path": str(report_path) if report_path else None,
        "ledger_stats": ledger.to_dict() if ledger else None,
        "monitors": monitors,
        "exposure": exposure_summary,
        "cdf_diffs_path": str(cdf_path) if cdf_path else None,
        "archive_manifest": str(manifest_path) if manifest_path else None,
        "mispricings": outcome.mispricings,
        "model_metadata": outcome.model_metadata,
        "scorecard_summary_path": str(scorecard_summary_path) if scorecard_summary_path else None,
        "scorecard_cdf_path": str(scorecard_cdf_path) if scorecard_cdf_path else None,
        "broker": broker_status,
        "outstanding": outstanding_summary,
    }


def resolve_series(mode: str) -> str | None:
    if mode == "pre_cpi":
        return "CPI"
    if mode == "pre_claims":
        return "CLAIMS"
    if mode == "teny_close":
        return "TNEY"
    if mode == "weather_cycle":
        return "WEATHER"
    return None


def run_step(name: str, log: dict[str, object], func, metadata: dict | None = None) -> None:
    entry = {"step": name, "started": datetime.now(tz=UTC).isoformat()}
    if metadata:
        entry["metadata"] = metadata
    try:
        func()
    except Exception as exc:  # pragma: no cover - instrumentation
        entry["status"] = "error"
        entry["error"] = str(exc)
        log.setdefault("steps", []).append(entry)
        raise
    else:
        entry["status"] = "ok"
        entry["completed"] = datetime.now(tz=UTC).isoformat()
        log.setdefault("steps", []).append(entry)


def write_log(mode: str, log: dict[str, object], now_utc: datetime) -> None:
    date_dir = PROC_ROOT / "logs" / now_utc.date().isoformat()
    date_dir.mkdir(parents=True, exist_ok=True)
    filename = date_dir / f"{now_utc.strftime('%H%M%S')}_{mode}.json"
    from kalshi_alpha.utils.secrets import ensure_safe_payload

    ensure_safe_payload(log)
    filename.write_text(json.dumps(log, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
