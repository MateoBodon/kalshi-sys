"""Daily orchestration pipeline for ladder strategies."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from kalshi_alpha.core.gates import QualityGateResult, load_quality_gate_config, run_quality_gates
from kalshi_alpha.core.kalshi_api import KalshiPublicClient
from kalshi_alpha.core.risk import PALGuard, PALPolicy, PortfolioRiskManager
from kalshi_alpha.datastore import ingest as datastore_ingest
from kalshi_alpha.datastore.paths import PROC_ROOT, RAW_ROOT
from kalshi_alpha.exec.ledger import PaperLedger, simulate_fills
from kalshi_alpha.exec.pipelines.calendar import resolve_run_window
from kalshi_alpha.exec.reports import write_markdown_report
from kalshi_alpha.exec.runners.scan_ladders import (
    _attach_series_metadata,
    _compute_exposure_summary,
    _write_cdf_diffs,
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
    parser.add_argument(
        "--when",
        type=_parse_date,
        help="Override target date (YYYY-MM-DD) for calendar scheduling.",
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
        gate_result = run_quality_gate_step(now_utc, log)
        if not gate_result.go:
            raise SystemExit(1)
        target_date: date = args.when if args.when else now_utc.astimezone(ET).date()
        run_window = resolve_run_window(mode=mode, target_date=target_date, now=now_utc, proc_root=PROC_ROOT)
        log["window"] = run_window.to_dict(now_utc)

        if not run_window.scan_allowed(now_utc):
            reason = "outside window"
            if run_window.notes:
                reason = ",".join(run_window.notes)
            log.setdefault("scan_notes", {})[mode] = reason
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


def run_quality_gate_step(now_utc: datetime, log: dict[str, object]) -> QualityGateResult:
    config = load_quality_gate_config(_resolve_quality_gate_config_path())
    result = run_quality_gates(config=config, now=now_utc, proc_root=PROC_ROOT, raw_root=RAW_ROOT)
    log["quality_gates"] = {
        "go": result.go,
        "reasons": result.reasons,
        "details": result.details,
    }
    _write_go_no_go(result)
    return result


def _resolve_quality_gate_config_path() -> Path:
    primary = Path("configs/quality_gates.yaml")
    if primary.exists():
        return primary
    fallback = Path("configs/quality_gates.example.yaml")
    if fallback.exists():
        return fallback
    return primary


def _write_go_no_go(result: QualityGateResult) -> Path:
    artifacts_dir = Path("reports/_artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    payload = {"go": bool(result.go), "reasons": list(result.reasons)}
    output_path = artifacts_dir / "go_no_go.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:  # pragma: no cover - surfaced via argparse
        raise argparse.ArgumentTypeError(f"Invalid date for --when: {value}") from exc


def run_scan(mode: str, args: argparse.Namespace, log: dict[str, object]) -> None:
    fixtures_root = Path(args.scanner_fixtures)
    driver_fixtures = Path(args.driver_fixtures)
    offline_mode = args.offline or not args.online
    client = KalshiPublicClient(offline_dir=fixtures_root / "kalshi", use_offline=offline_mode)

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
    )

    proposals = outcome.proposals
    monitors = outcome.monitors
    exposure_summary = _compute_exposure_summary(proposals)
    cdf_path = _write_cdf_diffs(outcome.cdf_diffs)

    if not proposals:
        log.setdefault("scan_results", {})[mode] = {
            "proposals": 0,
            "exposure": exposure_summary,
            "cdf_diffs_path": str(cdf_path) if cdf_path else None,
        }
        return

    _attach_series_metadata(
        proposals=proposals,
        series=series,
        driver_fixtures=driver_fixtures,
        offline=offline_mode,
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
            monitors=monitors,
            exposure_summary=exposure_summary,
        )

    log.setdefault("scan_results", {})[mode] = {
        "proposals": len(proposals),
        "proposals_path": str(proposals_path),
        "report_path": str(report_path) if report_path else None,
        "ledger_stats": ledger.to_dict() if ledger else None,
        "monitors": monitors,
        "exposure": exposure_summary,
        "cdf_diffs_path": str(cdf_path) if cdf_path else None,
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
