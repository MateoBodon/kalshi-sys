"""Weekly orchestration wrapper running daily pipeline modes in sequence."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from kalshi_alpha.datastore.paths import PROC_ROOT
from kalshi_alpha.exec.heartbeat import write_heartbeat
from kalshi_alpha.exec.pipelines import daily
from kalshi_alpha.exec.pipelines.calendar import ET, RunWindow, resolve_run_window
from kalshi_alpha.exec.state.orders import OutstandingOrdersState
from kalshi_alpha.utils.family import resolve_family


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the weekly ladder scan sequence.")
    env = parser.add_mutually_exclusive_group(required=True)
    env.add_argument("--offline", action="store_true", help="Use offline fixtures for drivers and API.")
    env.add_argument("--online", action="store_true", help="Use live drivers and API fetchers.")
    parser.add_argument("--paper", action="store_true", help="Enable reports and paper ledger outputs.")
    parser.add_argument("--report", action="store_true", help="Render markdown reports for each run.")
    parser.add_argument("--paper-ledger", action="store_true", help="Simulate paper fills for each run.")
    parser.add_argument("--include-weather", action="store_true", help="Append the weather cycle run.")
    parser.add_argument(
        "--broker",
        choices=["dry", "live"],
        default="dry",
        help="Broker adapter forwarded to daily runs.",
    )
    parser.add_argument(
        "--driver-fixtures",
        default="tests/fixtures",
        help="Driver fixture root forwarded to daily pipeline when offline.",
    )
    parser.add_argument(
        "--scanner-fixtures",
        default="tests/data_fixtures",
        help="Scanner fixture root forwarded to daily pipeline when offline.",
    )
    parser.add_argument(
        "--kelly-cap",
        type=float,
        default=0.25,
        help="Kelly cap forwarded to daily pipeline runs.",
    )
    parser.add_argument("--force-refresh", action="store_true", help="Forward force refresh flag.")
    parser.add_argument("--daily-loss-cap", type=float, help="Daily loss cap forwarded to runs (USD).")
    parser.add_argument("--weekly-loss-cap", type=float, help="Weekly loss cap forwarded to runs (USD).")
    parser.add_argument(
        "--fill-alpha",
        type=float,
        default=0.6,
        help="Fill alpha forwarded to daily pipeline runs (0-1).",
    )
    parser.add_argument(
        "--slippage-mode",
        default="top",
        choices=["top", "depth", "mid"],
        help="Slippage model forwarded to daily pipeline runs.",
    )
    parser.add_argument(
        "--impact-cap",
        type=float,
        default=0.02,
        help="Impact cap forwarded to daily pipeline runs (probability points).",
    )
    parser.add_argument(
        "--model-version",
        choices=["v0", "v15"],
        default="v15",
        help="Strategy model version used across weekly runs.",
    )
    parser.add_argument(
        "--modes",
        nargs="*",
        choices=["pre_cpi", "pre_claims", "teny_close", "weather_cycle"],
        help="Explicit mode sequence; defaults to CPI, Claims, TenY (and weather if enabled).",
    )
    parser.add_argument(
        "--preset",
        choices=["paper_live"],
        help="Named run preset overriding manual mode selection.",
    )
    parser.add_argument(
        "--kill-switch-file",
        help="Path to kill-switch sentinel file forwarded to daily runs.",
    )
    parser.add_argument(
        "--force-run",
        action="store_true",
        help="Forward --force-run to daily pipeline runs (DRY broker only).",
    )
    parser.add_argument(
        "--family",
        choices=["index", "macro", "all"],
        default=None,
        help="Family focus (default: FAMILY env or index). Index family skips macro pipelines.",
    )
    return parser.parse_args(argv)


def _fmt(value: float) -> str:
    return format(value, "g")


@dataclass(frozen=True)
class WeekRun:
    mode: str
    run_date: date | None
    window_et: str | None = None
    auto_resolved: bool = False
    original_date: date | None = None


def _default_modes(include_weather: bool) -> list[str]:
    sequence = ["pre_cpi", "pre_claims", "teny_close"]
    if include_weather:
        sequence.append("weather_cycle")
    return sequence


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    family = resolve_family(getattr(args, "family", None))
    if family == "index":
        print("[week] FAMILY=index -> skipping macro weekly run")
        return
    now = datetime.now(tz=UTC)
    write_heartbeat(
        mode="week:start",
        extra={
            "stage": "start",
            "timestamp": now.isoformat(),
            "outstanding": OutstandingOrdersState.load().summary(),
        },
    )

    if args.preset == "paper_live":
        schedule = _paper_live_schedule(now, include_weather=args.include_weather)
    else:
        modes = list(dict.fromkeys(args.modes or _default_modes(args.include_weather)))
        if not modes:
            print("[week] no modes requested")
            return
        schedule = [WeekRun(mode=mode, run_date=None) for mode in modes]

    if not schedule:
        print("[week] no modes requested")
        return

    env_flag = "--offline" if args.offline else "--online"
    base_flags = [
        env_flag,
        "--driver-fixtures",
        args.driver_fixtures,
        "--scanner-fixtures",
        args.scanner_fixtures,
        "--kelly-cap",
        _fmt(args.kelly_cap),
        "--model-version",
        args.model_version,
    ]
    if args.force_refresh:
        base_flags.append("--force-refresh")
    if args.force_run:
        base_flags.append("--force-run")
    if args.daily_loss_cap is not None:
        base_flags.extend(["--daily-loss-cap", _fmt(args.daily_loss_cap)])
    if args.weekly_loss_cap is not None:
        base_flags.extend(["--weekly-loss-cap", _fmt(args.weekly_loss_cap)])
    if args.fill_alpha is not None:
        base_flags.extend(["--fill-alpha", _fmt(args.fill_alpha)])
    if args.slippage_mode:
        base_flags.extend(["--slippage-mode", args.slippage_mode])
    if args.impact_cap is not None:
        base_flags.extend(["--impact-cap", _fmt(args.impact_cap)])

    report_flag = args.report or args.paper
    paper_flag = args.paper_ledger or args.paper
    if args.preset == "paper_live":
        report_flag = True
        paper_flag = True
    if report_flag:
        base_flags.append("--report")
    if paper_flag:
        base_flags.append("--paper-ledger")
    if args.broker:
        base_flags.extend(["--broker", args.broker])
    if args.preset == "paper_live":
        base_flags.append("--allow-no-go")
    if args.kill_switch_file:
        base_flags.extend(["--kill-switch-file", args.kill_switch_file])

    print(f"[week] starting weekly run at {now.isoformat()}")
    _print_outstanding("[week]")
    for run in schedule:
        mode = run.mode
        run_date = run.run_date
        write_heartbeat(
            mode=f"week:{mode}",
            extra={
                "stage": "pre_daily",
                "run_date": run_date.isoformat() if run_date else "auto",
                "window_et": run.window_et,
                "outstanding": OutstandingOrdersState.load().summary(),
            },
        )
        run_args = ["--mode", mode, *base_flags]
        if run_date is not None:
            run_args.extend(["--when", run_date.isoformat()])
        if run.window_et:
            run_args.extend(["--window-et", run.window_et])
        print(f"[week] running mode: {mode} ({run_date.isoformat() if run_date else 'auto'})")
        if run.auto_resolved and run.window_et:
            original = run.original_date.isoformat() if run.original_date else "unknown"
            print(f"[week] auto-resolved {mode} window {original} -> {run.window_et}")
        elif run.window_et:
            print(f"[week] scheduled {mode} window {run.window_et}")
        daily.main(run_args)
        _print_outstanding("[week]")


def _print_outstanding(prefix: str) -> None:
    state = OutstandingOrdersState.load()
    summary = state.summary()
    dry = summary.get("dry", 0)
    live = summary.get("live", 0)
    total = dry + live
    print(f"{prefix} Outstanding orders -> total={total} (dry={dry}, live={live})")


def _paper_live_schedule(now: datetime, *, include_weather: bool) -> list[WeekRun]:
    today_et = now.astimezone(ET).date()
    raw: list[tuple[str, date, str | None]] = []

    claims_window = resolve_run_window(mode="pre_claims", target_date=today_et, now=now)
    release_date = _window_date(claims_window, fallback=_next_weekday(today_et, target_weekday=3))
    freeze_date = _previous_business_day(release_date)
    raw.append(("pre_claims", freeze_date, "freeze"))
    raw.append(("pre_claims", release_date, "release"))

    cpi_window = resolve_run_window(mode="pre_cpi", target_date=release_date, now=now, proc_root=PROC_ROOT)
    cpi_release = _window_date(cpi_window, fallback=release_date)
    cpi_eve = _previous_business_day(cpi_release)
    raw.append(("pre_cpi", cpi_eve, "freeze"))
    raw.append(("pre_cpi", cpi_release, "release"))

    week_start = release_date - timedelta(days=release_date.weekday())
    for offset in range(5):
        ten_y_day = week_start + timedelta(days=offset)
        if ten_y_day.weekday() >= 5:
            continue
        raw.append(("teny_close", ten_y_day, None))
        if include_weather:
            raw.append(("weather_cycle", ten_y_day, None))

    ordered: list[tuple[str, date, str | None]] = []
    seen: set[tuple[str, date, str | None]] = set()
    for mode, when, phase in sorted(raw, key=lambda item: (item[1], item[0], item[2] or "")):
        key = (mode, when, phase)
        if key not in seen:
            ordered.append(key)
            seen.add(key)

    schedule: list[WeekRun] = []
    for mode, target_date, phase in ordered:
        schedule.append(_auto_resolve_week_run(mode, phase, target_date, now))
    return schedule


def _window_date(window: RunWindow | None, *, fallback: date) -> date:
    if window and window.reference is not None:
        return window.reference.astimezone(ET).date()
    return fallback


def _auto_resolve_week_run(mode: str, phase: str | None, target_date: date, now: datetime) -> WeekRun:
    current_date = target_date
    original = target_date
    auto = False
    window: RunWindow | None = None
    for _ in range(90):
        window = resolve_run_window(mode=mode, target_date=current_date, now=now, proc_root=PROC_ROOT)
        scan_close = window.scan_close
        if scan_close is None or now <= scan_close:
            break
        current_date = _advance_target_date(mode, phase, window, current_date)
        auto = True
    else:  # pragma: no cover - safety guard
        if window is None:
            window = resolve_run_window(mode=mode, target_date=target_date, now=now, proc_root=PROC_ROOT)

    resolved_date = _phase_adjusted_date(mode, phase, window, current_date)
    window_et = _format_window_range(window)
    return WeekRun(
        mode=mode,
        run_date=resolved_date,
        window_et=window_et,
        auto_resolved=auto,
        original_date=original if auto else None,
    )


def _advance_target_date(mode: str, phase: str | None, window: RunWindow, current_date: date) -> date:
    if mode == "teny_close":
        next_date = current_date + timedelta(days=1)
        while next_date.weekday() >= 5:
            next_date += timedelta(days=1)
        return next_date
    if mode == "pre_claims":
        reference = window.reference.astimezone(ET).date() if window.reference else current_date
        return reference + timedelta(days=7)
    if mode == "pre_cpi":
        reference = window.reference.astimezone(ET).date() if window.reference else current_date
        return reference + timedelta(days=1)
    return current_date + timedelta(days=1)


def _phase_adjusted_date(mode: str, phase: str | None, window: RunWindow, fallback: date) -> date:
    if window.reference is not None:
        ref_date = window.reference.astimezone(ET).date()
    else:
        ref_date = fallback
    if mode == "pre_cpi":
        if phase == "freeze":
            return _previous_business_day(ref_date)
        return ref_date
    if mode == "pre_claims":
        if phase == "freeze":
            return _previous_business_day(ref_date)
        return ref_date
    if mode == "teny_close":
        return ref_date
    return fallback


def _format_window_range(window: RunWindow) -> str | None:
    if window.scan_open is None or window.scan_close is None:
        return None
    open_et = window.scan_open.astimezone(ET)
    close_et = window.scan_close.astimezone(ET)
    if open_et.date() == close_et.date():
        return f"{open_et.strftime('%Y-%m-%d %H:%M')}–{close_et.strftime('%H:%M')} ET"
    return (
        f"{open_et.strftime('%Y-%m-%d %H:%M')} ET -> "
        f"{close_et.strftime('%Y-%m-%d %H:%M')} ET"
    )


def _previous_business_day(candidate: date) -> date:
    current = candidate - timedelta(days=1)
    while current.weekday() >= 5:
        current -= timedelta(days=1)
    return current


def _next_weekday(start: date, target_weekday: int) -> date:
    delta = (target_weekday - start.weekday()) % 7
    return start + timedelta(days=delta)


if __name__ == "__main__":  # pragma: no cover
    main()
