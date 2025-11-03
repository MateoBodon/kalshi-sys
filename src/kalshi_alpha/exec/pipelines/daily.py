"""Daily orchestration pipeline for ladder strategies."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from kalshi_alpha.core.archive.scorecards import build_replay_scorecard
from kalshi_alpha.core.execution.fillratio import FillRatioEstimator, load_alpha, tune_alpha
from kalshi_alpha.core.execution.slippage import SlippageModel, load_slippage_model
from kalshi_alpha.core.gates import QualityGateResult, load_quality_gate_config, run_quality_gates
from kalshi_alpha.core.kalshi_api import KalshiPublicClient
from kalshi_alpha.core.pricing.align import SkipScan
from kalshi_alpha.core.risk import PALGuard, PALPolicy, PortfolioRiskManager, drawdown
from kalshi_alpha.datastore import ingest as datastore_ingest
from kalshi_alpha.datastore.paths import PROC_ROOT, RAW_ROOT
from kalshi_alpha.drivers import macro_calendar
from kalshi_alpha.exec.gate_utils import resolve_quality_gate_config_path, write_go_no_go
from kalshi_alpha.exec.heartbeat import (
    heartbeat_stale,
    kill_switch_engaged,
    resolve_kill_switch_path,
    write_heartbeat,
)
from kalshi_alpha.exec.ledger import PaperLedger, simulate_fills
from kalshi_alpha.exec.pipelines.calendar import RunWindow, resolve_run_window
from kalshi_alpha.exec.reports import write_markdown_report
from kalshi_alpha.exec.runners.scan_ladders import (
    _apply_ev_honesty_gate,
    _archive_and_replay,
    _attach_series_metadata,
    _clear_dry_orders_start,
    _compute_ev_honesty_rows,
    _compute_exposure_summary,
    _load_replay_for_ev_honesty,
    _write_cdf_diffs,
    execute_broker,
    scan_series,
    write_proposals,
)
from kalshi_alpha.exec.state.orders import OutstandingOrdersState
from kalshi_alpha.strategies.claims import calibrate as calibrate_claims
from kalshi_alpha.strategies.cpi import calibrate as calibrate_cpi
from kalshi_alpha.strategies.teny import calibrate as calibrate_teny
from kalshi_alpha.strategies.weather import calibrate as calibrate_weather
from kalshi_alpha.utils.secrets import ensure_safe_payload

if TYPE_CHECKING:  # pragma: no cover
    from kalshi_alpha.exec.pipelines.today import ScheduledRun

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
        "--snap-to-window",
        choices=["off", "wait", "print"],
        default="off",
        help="Align execution with the next scan window (wait, print, or off).",
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
        "--ev-honesty-shrink",
        type=float,
        default=0.9,
        help="Maker EV shrink factor applied for EV honesty (0-1).",
    )
    parser.add_argument(
        "--broker",
        choices=["dry", "live"],
        default="dry",
        help="Broker adapter to use when generating orders.",
    )
    parser.add_argument(
        "--clear-dry-orders-start",
        action="store_true",
        help="Clear outstanding DRY orders before scanning.",
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


def _resolve_fill_alpha_value(fill_alpha_arg: object, series: str) -> tuple[float, bool]:  # noqa: PLR0912
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
        raw = fill_alpha_arg.strip().lower()
        if raw == "auto":
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
                candidate = float(raw)
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
    return candidate, auto


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    modes = MODE_SEQUENCE if args.mode == "full" else [args.mode]
    for mode in modes:
        run_mode(mode, args)
        if getattr(args, "snap_to_window", "off") == "print":
            break


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

    snap_result = _apply_snap_option(mode=mode, args=args, now_utc=now_utc)
    if snap_result is None:
        log.setdefault("scan_notes", {})[mode] = "snap_print"
        write_log(mode, log, now_utc)
        return

    now_utc, target_date, run_window, snapped = snap_result
    log["timestamp"] = now_utc.isoformat()
    log["window"] = run_window.to_dict(now_utc)
    if snapped:
        log.setdefault("snap_notes", []).append("waited_for_window")

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
        now_utc = datetime.now(tz=UTC)
        _refresh_heartbeat("post_calibrations")
        series = resolve_series(mode)
        fill_alpha_value = DEFAULT_FILL_ALPHA
        fill_alpha_auto = False
        extra_monitors: dict[str, object] = {}
        extra_monitors["online"] = bool(args.online)
        if series is not None:
            fill_alpha_value, fill_alpha_auto = _resolve_fill_alpha_value(args.fill_alpha, series)
            log.setdefault("fill_alpha", {})[series] = fill_alpha_value
            key = "fill_alpha_auto" if fill_alpha_auto else "fill_alpha"
            extra_monitors[key] = fill_alpha_value
            extra_monitors["ev_honesty_shrink"] = getattr(args, "ev_honesty_shrink", 0.9)
        if force_run_enabled:
            extra_monitors["force_run"] = True
        gate_result = run_quality_gate_step(args, now_utc, log, monitors=extra_monitors)
        if not gate_result.go and not getattr(args, "allow_no_go", False):
            raise SystemExit(1)
        target_date = target_date if getattr(args, "when", None) else now_utc.astimezone(ET).date()
        run_window = resolve_run_window(mode=mode, target_date=target_date, now=now_utc, proc_root=PROC_ROOT)
        log["window"] = run_window.to_dict(now_utc)

        scan_allowed = run_window.scan_allowed(now_utc)
        if not scan_allowed and not force_run_enabled:
            reason = "outside window"
            if run_window.notes:
                reason = ",".join(run_window.notes)
            log.setdefault("scan_notes", {})[mode] = reason
            return
        elif series is None:
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

    def history_bounds(sequence: Sequence[dict[str, object]] | None) -> tuple[date, date] | None:
        if not sequence:
            return None
        dates: list[date] = []
        for row in sequence:
            raw_date = row.get("date")
            if isinstance(raw_date, str):
                try:
                    dates.append(date.fromisoformat(raw_date))
                except ValueError:
                    continue
        if not dates:
            return None
        return min(dates), max(dates)

    teny_bounds = history_bounds(histories["teny"])
    if teny_bounds is not None:
        start_date, end_date = teny_bounds
        try:
            macro_calendar.emit_day_dummies(
                start_date,
                end_date,
                offline=args.offline,
                fixtures_dir=fixtures_root if args.offline else None,
            )
        except Exception:  # pragma: no cover - macro calendar is non-critical for calibration
            log.setdefault("calibration_warnings", {})["teny_macro_calendar"] = "emit_failed"

    def calibrate_wrapper(
        name: str,
        func: Callable[[Sequence[dict[str, object]]], None],
        history: Sequence[dict[str, object]] | None,
    ) -> None:
        if not history:
            log.setdefault("calibration_skipped", {})[name] = "missing_history"
            if heartbeat_cb:
                heartbeat_cb("post_calibrate", {"calibration": name, "status": "skipped"})
            return
        run_step(name=f"calibrate_{name}", log=log, func=lambda: func(list(history)))
        if heartbeat_cb:
            heartbeat_cb("post_calibrate", {"calibration": name})

    calibrate_wrapper("cpi", calibrate_cpi, histories["cpi"])
    calibrate_wrapper("claims", calibrate_claims, histories["claims"])
    calibrate_wrapper("teny", calibrate_teny, histories["teny"])
    calibrate_wrapper("weather", calibrate_weather, histories["weather"])


def _evaluate_quality_gates(
    args: argparse.Namespace,
    now_utc: datetime,
    *,
    monitors: dict[str, float] | None = None,
    apply_side_effects: bool = True,
) -> tuple[QualityGateResult, dict[str, object]]:
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
        if apply_side_effects:
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
    return combined, log_entry


def run_quality_gate_step(
    args: argparse.Namespace,
    now_utc: datetime,
    log: dict[str, object],
    *,
    monitors: dict[str, float] | None = None,
) -> QualityGateResult:
    combined, log_entry = _evaluate_quality_gates(
        args,
        now_utc,
        monitors=monitors,
        apply_side_effects=True,
    )
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
        return date.fromisoformat(value)
    except ValueError as exc:  # pragma: no cover - surfaced via argparse
        raise argparse.ArgumentTypeError(f"Invalid date for --when: {value}") from exc


def _apply_fill_realism_gate(
    log: dict[str, object],
    ledger: PaperLedger | None,
    monitors: dict[str, object],
) -> tuple[float | None, QualityGateResult | None]:
    if ledger is None or not ledger.records:
        return None, None
    deltas: list[float] = []
    for record in ledger.records:
        alpha_row_value = getattr(record, "alpha_row", None)
        if alpha_row_value is None:
            continue
        deltas.append(abs(float(record.fill_ratio) - float(alpha_row_value)))
    if not deltas:
        return None, None
    metric = statistics.median(deltas)
    monitors.setdefault("fill_realism_median", metric)
    quality = log.setdefault("quality_gates", {"go": True, "reasons": [], "details": {}})
    details = quality.setdefault("details", {})
    details["fill_realism_median"] = metric
    reasons = quality.setdefault("reasons", [])
    go_flag = quality.get("go", True)
    if metric > 0.10:
        if "fill_realism_miss" not in reasons:
            reasons.append("fill_realism_miss")
        go_flag = False
    quality["go"] = go_flag
    quality["reasons"] = reasons
    quality["details"] = details
    result = QualityGateResult(go=go_flag, reasons=list(reasons), details=dict(details))
    write_go_no_go(result)
    return metric, result


def _compute_next_window(
    *,
    mode: str,
    start_date: date,
    now_utc: datetime,
    max_days: int = 14,
) -> tuple[date, RunWindow]:
    """Find the next run window on or after ``start_date`` that closes in the future."""

    candidate_date = start_date
    for _ in range(max_days):
        window = resolve_run_window(mode=mode, target_date=candidate_date, now=now_utc, proc_root=PROC_ROOT)
        if window.scan_open and window.scan_close and now_utc <= window.scan_close:
            return candidate_date, window
        candidate_date += timedelta(days=1)
    # Fallback to the last computed window if none suitable were found
    return candidate_date - timedelta(days=1), window


def _format_window_line(
    prefix: str,
    run_window: ScheduledRun,
    *,
    include_notes: bool = False,
) -> str:
    scan_open = run_window.scan_open.astimezone(ET).isoformat() if run_window.scan_open else "n/a"
    scan_close = run_window.scan_close.astimezone(ET).isoformat() if run_window.scan_close else "n/a"
    line = f"{prefix}: open={scan_open} close={scan_close}"
    if include_notes and run_window.notes:
        line += f" notes={','.join(run_window.notes)}"
    return line


def _print_next_window(mode: str, window_date: date, run_window: ScheduledRun) -> None:
    print(f"[snap] Next window for {mode.upper()} on {window_date.isoformat()}")
    print(_format_window_line("[snap] Window (ET)", run_window, include_notes=True))


def _apply_snap_option(
    *,
    mode: str,
    args: argparse.Namespace,
    now_utc: datetime,
) -> tuple[datetime, date, RunWindow, bool] | None:
    """Handle snap-to-window logic. Returns (now, date, window, waited) or None if we should exit."""

    target_date: date = args.when if getattr(args, "when", None) else now_utc.astimezone(ET).date()
    run_window = resolve_run_window(mode=mode, target_date=target_date, now=now_utc, proc_root=PROC_ROOT)
    option = getattr(args, "snap_to_window", "off")
    if option == "off":
        return now_utc, target_date, run_window, False

    next_date, next_window = _compute_next_window(mode=mode, start_date=target_date, now_utc=now_utc)
    if option == "print":
        _print_next_window(mode, next_date, next_window)
        return None

    if option == "wait":
        if next_window.scan_open is None:
            _print_next_window(mode, next_date, next_window)
            return now_utc, next_date, next_window, False
        wait_seconds = max(0.0, (next_window.scan_open - now_utc).total_seconds())
        if wait_seconds > 0:
            print(f"[snap] Waiting {wait_seconds:.0f}s for next {mode.upper()} window ...")
            time.sleep(wait_seconds)
        updated_now = datetime.now(tz=UTC)
        refreshed_window = resolve_run_window(
            mode=mode,
            target_date=next_date,
            now=updated_now,
            proc_root=PROC_ROOT,
        )
        return updated_now, next_date, refreshed_window, True

    raise ValueError(f"Unsupported snap-to-window option: {option}")


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

    outstanding_start = _clear_dry_orders_start(
        enabled=getattr(args, "clear_dry_orders_start", False),
        broker_mode=getattr(args, "broker", "dry"),
        quiet=False,
    )

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
    try:
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
            ev_honesty_shrink=getattr(args, "ev_honesty_shrink", 0.9),
        )
    except SkipScan as exc:
        reason_text = getattr(exc, "reason", str(exc))
        log.setdefault("scan_notes", {})[mode] = reason_text
        log.setdefault("errors", []).append(reason_text)
        write_go_no_go(
            QualityGateResult(
                go=False,
                reasons=[reason_text],
                details={"mode": mode},
            )
        )
        outstanding_summary = OutstandingOrdersState.load().summary()
        write_heartbeat(
            mode=f"daily:{mode}",
            extra={
                "stage": "scan_skipped",
                "series": series,
                "reason": reason_text,
                "outstanding": outstanding_summary,
                "force_run": force_run_enabled,
                "window_et": getattr(args, "window_et", None),
            },
        )
        return

    proposals = outcome.proposals
    books_at_scan = dict(getattr(outcome, "books_at_scan", {}))
    book_snapshot_started_at = getattr(outcome, "book_snapshot_started_at", None)
    book_snapshot_completed_at = getattr(outcome, "book_snapshot_completed_at", None)
    monitors = dict(outcome.monitors or {})
    monitors["online"] = bool(args.online)
    monitors.setdefault("orderbook_snapshots", len(books_at_scan))
    if book_snapshot_started_at is not None:
        monitors.setdefault("book_snapshot_started_at", book_snapshot_started_at.isoformat())
    if book_snapshot_completed_at is not None:
        monitors.setdefault("book_snapshot_completed_at", book_snapshot_completed_at.isoformat())
    monitors.setdefault(
        "outstanding_orders_start_total",
        sum(outstanding_start.values()),
    )
    monitors.setdefault(
        "outstanding_orders_start_breakdown",
        dict(sorted(outstanding_start.items())),
    )
    if fill_alpha_auto:
        monitors["fill_alpha_auto"] = fill_alpha_value
    else:
        monitors.setdefault("fill_alpha", fill_alpha_value)
    monitors.setdefault("model_version", args.model_version)
    exposure_summary = _compute_exposure_summary(proposals)
    cdf_path = _write_cdf_diffs(outcome.cdf_diffs)
    should_archive = args.report or args.paper_ledger or force_run_enabled
    if should_archive and outcome.markets:
        expected_ids = {market.id for market in outcome.markets}
        missing_ids = expected_ids.difference(books_at_scan.keys())
        if missing_ids:
            monitors["orderbook_snapshot_missing"] = len(missing_ids)

    ledger: PaperLedger | None = None
    if proposals and (args.paper_ledger or args.report):
        ledger_books = {
            proposal.market_id: books_at_scan[proposal.market_id]
            for proposal in proposals
            if proposal.market_id in books_at_scan
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
            elif slippage_mode == "depth":
                calibrated = load_slippage_model(series, mode=slippage_mode)
                if calibrated is not None:
                    slippage_model = calibrated
            if slippage_model is None:
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
    replay_path: Path | None = None
    scorecard_summary_path: Path | None = None
    scorecard_cdf_path: Path | None = None
    scorecard_records: list[dict[str, object]] | None = None
    if should_archive and outcome.series is not None:
        archive_result = _archive_and_replay(
            client=client,
            series=outcome.series,
            events=outcome.events,
            markets=outcome.markets,
            orderbooks=books_at_scan,
            proposals_path=proposals_path,
            driver_fixtures=driver_fixtures,
            scanner_fixtures=fixtures_root,
            model_metadata=outcome.model_metadata,
        )
        if isinstance(archive_result, tuple):
            manifest_path, replay_path = archive_result
        else:
            manifest_path = archive_result
            replay_path = None
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
                except ValueError:
                    archiver_ts = None
                if archiver_ts is not None:
                    latency = (archiver_ts - book_snapshot_completed_at).total_seconds() * 1000.0
                    monitors["book_latency_ms"] = round(max(0.0, latency), 3)

    if replay_path:
        replay_records = _load_replay_for_ev_honesty(replay_path)
        ev_rows, ev_max_delta = _compute_ev_honesty_rows(proposals, replay_records)
        if ev_rows:
            monitors["ev_honesty_table"] = ev_rows
            monitors["ev_honesty_max_delta"] = ev_max_delta
            monitors["ev_honesty_count"] = len(ev_rows)
            monitors.setdefault("ev_per_contract_diff_max", ev_max_delta)
    _apply_ev_honesty_gate(monitors, threshold=0.10)

    outcome.monitors = monitors

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
                monitors=monitors,
                quiet=False,
                go_status=log.get("quality_gates", {}).get("go", True),
            )
        except RuntimeError as exc:
            broker_status = {"mode": getattr(args, "broker", "dry"), "orders_recorded": 0, "error": str(exc)}
            log.setdefault("broker", broker_status)

    throttle_rows = 0
    if ledger:
        throttle_rows = sum(1 for record in ledger.records if getattr(record, "size_throttled", False))
        if throttle_rows:
            monitors["size_throttled_rows"] = throttle_rows
    fill_realism_metric, realism_result = _apply_fill_realism_gate(log, ledger, monitors)

    outstanding_summary = OutstandingOrdersState.load().summary()
    pilot_metadata = {
        "mode": getattr(args, "broker", "dry"),
        "kelly_cap": getattr(args, "kelly_cap", None),
        "max_var": getattr(args, "max_var", None),
        "fill_alpha": fill_alpha_value,
        "outstanding_total": sum(outstanding_summary.values()),
        "force_run": force_run_enabled,
        "online": bool(args.online),
    }
    window_et = getattr(args, "window_et", None)
    if window_et:
        pilot_metadata["window_et"] = window_et
    if throttle_rows:
        pilot_metadata["size_throttled_rows"] = throttle_rows
    if fill_realism_metric is not None:
        pilot_metadata["fill_realism_median"] = fill_realism_metric
    quality_entry = log.get("quality_gates")
    if realism_result is not None:
        write_go_no_go(realism_result)
    elif isinstance(quality_entry, dict):
        write_go_no_go(
            QualityGateResult(
                go=quality_entry.get("go", True),
                reasons=list(quality_entry.get("reasons", [])),
                details=dict(quality_entry.get("details", {})),
            )
        )
    report_path = None
    write_report = bool(args.report or force_run_enabled)
    if write_report:
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


def run_step(
    name: str,
    log: dict[str, object],
    func: Callable[[], object | None],
    metadata: dict | None = None,
) -> None:
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
    ensure_safe_payload(log)
    filename.write_text(json.dumps(log, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
