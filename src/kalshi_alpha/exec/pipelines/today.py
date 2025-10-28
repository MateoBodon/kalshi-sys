"""Autonomous "today" orchestration that selects daily modes based on calendars."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from kalshi_alpha.exec.pipelines import daily
from kalshi_alpha.exec.pipelines.calendar import ET, resolve_run_window
from kalshi_alpha.exec.state.orders import OutstandingOrdersState


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run appropriate daily modes for today.")
    env = parser.add_mutually_exclusive_group(required=True)
    env.add_argument("--offline", action="store_true", help="Run using offline fixtures.")
    env.add_argument("--online", action="store_true", help="Run against live data.")
    parser.add_argument("--report", action="store_true", help="Render reports after scans.")
    parser.add_argument("--paper-ledger", action="store_true", help="Simulate paper ledger fills.")
    parser.add_argument("--include-weather", action="store_true", help="Include weather cycle run.")
    parser.add_argument(
        "--driver-fixtures",
        default="tests/fixtures",
        help="Driver fixtures root when running offline.",
    )
    parser.add_argument(
        "--scanner-fixtures",
        default="tests/data_fixtures",
        help="Scanner fixtures root when running offline.",
    )
    parser.add_argument(
        "--kelly-cap",
        type=float,
        default=0.25,
        help="Override default Kelly cap forwarded to the daily pipeline.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Forward force-refresh flag to the daily pipeline.",
    )
    parser.add_argument(
        "--daily-loss-cap",
        type=float,
        help="Daily drawdown cap forwarded to daily pipeline (USD).",
    )
    parser.add_argument(
        "--weekly-loss-cap",
        type=float,
        help="Weekly drawdown cap forwarded to daily pipeline (USD).",
    )
    parser.add_argument(
        "--fill-alpha",
        type=float,
        default=0.6,
        help="Fill ratio alpha forwarded to daily pipeline (0-1).",
    )
    parser.add_argument(
        "--slippage-mode",
        default="top",
        choices=["top", "depth", "mid"],
        help="Slippage mode forwarded to daily pipeline.",
    )
    parser.add_argument(
        "--impact-cap",
        type=float,
        default=0.02,
        help="Impact cap forwarded to daily pipeline (probability points).",
    )
    parser.add_argument(
        "--broker",
        choices=["dry", "live"],
        default="dry",
        help="Broker adapter forwarded to the daily pipeline.",
    )
    parser.add_argument(
        "--model-version",
        choices=["v0", "v15"],
        default="v15",
        help="Strategy model version forwarded to the daily pipeline.",
    )
    parser.add_argument(
        "--kill-switch-file",
        help="Path to kill-switch sentinel file forwarded to daily runs.",
    )
    return parser.parse_args(argv)


def _fmt_float(value: float) -> str:
    """Format floats for CLI forwarding without trailing zeros."""
    return format(value, "g")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    now = _now()
    runs = _plan_runs(now, include_weather=args.include_weather)

    _print_outstanding("[today]")

    badge_path = _badge_path()
    print(f"[today] GO/NO-GO badge -> {badge_path}")

    if not runs:
        print("[today] No modes scheduled for current calendar.")
        return

    env_flag = "--offline" if args.offline else "--online"
    base_flags = [
        env_flag,
        "--driver-fixtures",
        args.driver_fixtures,
        "--scanner-fixtures",
        args.scanner_fixtures,
        "--kelly-cap",
        _fmt_float(args.kelly_cap),
    ]
    if args.report:
        base_flags.append("--report")
    if args.paper_ledger:
        base_flags.append("--paper-ledger")
    if args.force_refresh:
        base_flags.append("--force-refresh")
    if args.daily_loss_cap is not None:
        base_flags.extend(["--daily-loss-cap", _fmt_float(args.daily_loss_cap)])
    if args.weekly_loss_cap is not None:
        base_flags.extend(["--weekly-loss-cap", _fmt_float(args.weekly_loss_cap)])
    if args.fill_alpha is not None:
        base_flags.extend(["--fill-alpha", _fmt_float(args.fill_alpha)])
    if args.slippage_mode:
        base_flags.extend(["--slippage-mode", args.slippage_mode])
    if args.impact_cap is not None:
        base_flags.extend(["--impact-cap", _fmt_float(args.impact_cap)])
    if args.broker:
        base_flags.extend(["--broker", args.broker])
    if args.model_version:
        base_flags.extend(["--model-version", args.model_version])
    if args.kill_switch_file:
        base_flags.extend(["--kill-switch-file", args.kill_switch_file])

    for mode_run in runs:
        mode, run_date = mode_run.mode, mode_run.run_date
        run_args = ["--mode", mode, "--when", run_date.isoformat(), *base_flags]
        print(f"[today] Running {mode} for {run_date.isoformat()} ...")
        daily.main(run_args)
        _print_manifest_link()
        _print_outstanding("[today]")

    final_manifest = _load_latest_manifest()
    if final_manifest:
        print(f"[today] Manifest used: {final_manifest}")

    go_flag = _load_go_status(badge_path)
    if go_flag is False:
        raise SystemExit(1)


def _print_outstanding(prefix: str) -> None:
    state = OutstandingOrdersState.load()
    summary = state.summary()
    dry = summary.get("dry", 0)
    live = summary.get("live", 0)
    total = dry + live
    print(f"{prefix} Outstanding orders -> total={total} (dry={dry}, live={live})")


@dataclass(frozen=True)
class ScheduledRun:
    mode: str
    run_date: date


def _plan_runs(
    now: datetime,
    *,
    include_weather: bool,
    proc_root: Path | None = None,
) -> list[ScheduledRun]:
    today_et = now.astimezone(ET).date()
    selections: list[ScheduledRun] = []
    seen: set[tuple[str, date]] = set()

    def add(mode: str, run_date: date) -> None:
        key = (mode, run_date)
        if key not in seen:
            selections.append(ScheduledRun(mode, run_date))
            seen.add(key)

    # CPI eve/day
    try:
        cpi_window = resolve_run_window(mode="pre_cpi", target_date=today_et, now=now, proc_root=proc_root)
        if cpi_window.reference:
            release_date = cpi_window.reference.astimezone(ET).date()
            diff = (release_date - today_et).days
            if diff == 0 or (diff == 1 and cpi_window.freeze_active(now)):
                add("pre_cpi", release_date)
    except Exception:  # pragma: no cover - calendar missing
        pass

    # Claims (Wed PM / Thu AM)
    try:
        claims_window = resolve_run_window(mode="pre_claims", target_date=today_et, now=now)
        if claims_window.reference:
            claims_date = claims_window.reference.astimezone(ET).date()
            diff = (claims_date - today_et).days
            if diff == 0:
                add("pre_claims", claims_date)
            elif diff == 1 and claims_window.freeze_active(now):
                add("pre_claims", claims_date)
    except Exception:  # pragma: no cover
        pass

    # Ten-year close (business days only)
    try:
        if today_et.weekday() < 5:
            teny_window = resolve_run_window(mode="teny_close", target_date=today_et, now=now)
            if teny_window.reference and teny_window.reference.astimezone(ET).date() == today_et:
                add("teny_close", today_et)
    except Exception:  # pragma: no cover
        pass

    if include_weather:
        add("weather_cycle", today_et)

    return selections


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _badge_path() -> Path:
    return Path("reports/_artifacts/go_no_go.json")


def _latest_manifest_marker() -> Path:
    return Path("reports/_artifacts/latest_manifest.txt")


def _load_latest_manifest() -> Path | None:
    marker = _latest_manifest_marker()
    if not marker.exists():
        return None
    content = marker.read_text(encoding="utf-8").strip()
    if not content:
        return None
    return Path(content)


def _print_manifest_link() -> None:
    manifest = _load_latest_manifest()
    if manifest is not None:
        print(f"[today] Archived manifest: {manifest}")


def _load_go_status(path: Path) -> bool | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    value = payload.get("go")
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    return bool(value)


if __name__ == "__main__":
    main()
