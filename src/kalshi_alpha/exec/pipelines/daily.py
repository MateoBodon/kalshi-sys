"""Daily orchestration pipeline for ladder strategies."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from kalshi_alpha.core.kalshi_api import KalshiPublicClient
from kalshi_alpha.core.risk import PALGuard, PALPolicy, PortfolioRiskManager
from kalshi_alpha.datastore import ingest as datastore_ingest
from kalshi_alpha.datastore.paths import PROC_ROOT
from kalshi_alpha.exec.ledger import PaperLedger, simulate_fills
from kalshi_alpha.exec.reports import write_markdown_report
from kalshi_alpha.exec.runners.scan_ladders import (
    _attach_series_metadata,
    scan_series,
    write_proposals,
)
from kalshi_alpha.strategies.claims import calibrate as calibrate_claims
from kalshi_alpha.strategies.cpi import calibrate as calibrate_cpi
from kalshi_alpha.strategies.teny import calibrate as calibrate_teny
from kalshi_alpha.strategies.weather import calibrate as calibrate_weather

ET = ZoneInfo("America/New_York")

MODE_SEQUENCE = ["pre_cpi", "pre_claims", "teny_close", "weather_cycle"]


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
    return parser.parse_args(argv)


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
    }
    try:
        run_ingest(args, log)
        run_calibrations(args, log)
        window_state = evaluate_window(mode, now_utc)
        log["window"] = window_state

        if not window_state.get("scan_allowed", True):
            log.setdefault("scan_notes", {})[mode] = window_state.get(
                "reason", "outside window"
            )
        else:
            run_scan(mode, args, log)
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


def run_calibrations(args: argparse.Namespace, log: dict[str, object]) -> None:
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
            return
        run_step(name=f"calibrate_{name}", log=log, func=lambda: func(history))

    calibrate_wrapper("cpi", calibrate_cpi, histories["cpi"])
    calibrate_wrapper("claims", calibrate_claims, histories["claims"])
    calibrate_wrapper("teny", calibrate_teny, histories["teny"])
    calibrate_wrapper("weather", calibrate_weather, histories["weather"])


def evaluate_window(mode: str, now_utc: datetime) -> dict[str, object]:
    now_et = now_utc.astimezone(ET)
    result = {"scan_allowed": True, "freeze_active": False}
    if mode == "pre_cpi":
        start = time(6, 0)
        end = time(8, 20)
        freeze_start = datetime.combine(now_et.date(), start, tzinfo=ET) - timedelta(hours=24)
        result["freeze_active"] = now_et >= freeze_start
        result.update(check_window(now_et, start, end))
    elif mode == "pre_claims":
        start = time(6, 0)
        end = time(8, 25)
        freeze_time = datetime.combine(now_et.date(), time(18, 0), tzinfo=ET)
        weekday = now_et.weekday()
        result["freeze_active"] = weekday > 2 or (weekday == 2 and now_et >= freeze_time)
        result.update(check_window(now_et, start, end))
    elif mode == "teny_close":
        start = time(14, 30)
        end = time(15, 25)
        freeze_time = datetime.combine(now_et.date(), time(13, 30), tzinfo=ET)
        result["freeze_active"] = now_et >= freeze_time
        result.update(check_window(now_et, start, end))
    elif mode == "weather_cycle":
        cycle_hours = {0, 6, 12, 18}
        hour = now_utc.hour
        cycle = min(cycle_hours, key=lambda h: abs(hour - h))
        cycle_start = datetime.combine(now_utc.date(), time(cycle, 0), tzinfo=UTC)
        window_start = cycle_start
        window_end = cycle_start + timedelta(minutes=45)
        result["freeze_active"] = now_utc >= cycle_start
        result["scan_allowed"] = window_start <= now_utc <= window_end
        if not result["scan_allowed"]:
            result["reason"] = "outside weather cycle window"
    return result


def check_window(now_et: datetime, start: time, end: time) -> dict[str, object]:
    start_dt = datetime.combine(now_et.date(), start, tzinfo=ET)
    end_dt = datetime.combine(now_et.date(), end, tzinfo=ET)
    allowed = start_dt <= now_et <= end_dt
    payload = {"scan_allowed": allowed}
    if not allowed:
        payload["reason"] = f"window {start.strftime('%H:%M')}-{end.strftime('%H:%M')} ET"
    return payload


def run_scan(mode: str, args: argparse.Namespace, log: dict[str, object]) -> None:
    fixtures_root = Path(args.scanner_fixtures)
    driver_fixtures = Path(args.driver_fixtures)
    client = KalshiPublicClient(offline_dir=fixtures_root / "kalshi", use_offline=args.offline)

    pal_policy_path = Path("configs/pal_policy.yaml")
    if not pal_policy_path.exists():
        pal_policy_path = Path("configs/pal_policy.example.yaml")
    pal_policy = PALPolicy.from_yaml(pal_policy_path)
    pal_guard = PALGuard(pal_policy)

    risk_manager: PortfolioRiskManager | None = None
    max_var = None
    strategy_name = "auto"
    series = resolve_series(mode)
    if series is None:
        log.setdefault("scan_notes", {})[mode] = "series_not_supported"
        return

    proposals = scan_series(
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
        offline=args.offline,
        sizing_mode="kelly",
        kelly_cap=args.kelly_cap,
    )

    if not proposals:
        log.setdefault("scan_results", {})[mode] = {"proposals": 0}
        return

    _attach_series_metadata(
        proposals=proposals,
        series=series,
        driver_fixtures=driver_fixtures,
        offline=args.offline,
    )

    ledger: PaperLedger | None = None
    if args.paper_ledger or args.report:
        orderbooks = {
            proposal.market_id: client.get_orderbook(proposal.market_id) for proposal in proposals
        }
        ledger = simulate_fills(
            proposals,
            orderbooks,
            artifacts_dir=Path("reports/_artifacts"),
        )

    proposals_path = write_proposals(series=series, proposals=proposals, output_dir=Path("exec/proposals"))
    report_path = None
    if args.report:
        report_path = write_markdown_report(
            series=series,
            proposals=proposals,
            ledger=ledger,
            output_dir=Path("reports") / series.upper(),
        )

    log.setdefault("scan_results", {})[mode] = {
        "proposals": len(proposals),
        "proposals_path": str(proposals_path),
        "report_path": str(report_path) if report_path else None,
        "ledger_stats": ledger.to_dict() if ledger else None,
    }


def resolve_series(mode: str) -> str | None:
    if mode == "pre_cpi":
        return "CPI"
    if mode == "pre_claims":
        return None  # placeholder until claims scanner added
    if mode == "teny_close":
        return None
    if mode == "weather_cycle":
        return None
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
    filename.write_text(json.dumps(log, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
