"""Weekly orchestration wrapper running daily pipeline modes in sequence."""

from __future__ import annotations

import argparse
from datetime import UTC, date, datetime, timedelta

from kalshi_alpha.exec.pipelines import daily
from kalshi_alpha.exec.pipelines.calendar import ET, resolve_run_window
from kalshi_alpha.datastore.paths import PROC_ROOT
from kalshi_alpha.exec.state.orders import OutstandingOrdersState


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
    return parser.parse_args(argv)


def _fmt(value: float) -> str:
    return format(value, "g")


def _default_modes(include_weather: bool) -> list[str]:
    sequence = ["pre_cpi", "pre_claims", "teny_close"]
    if include_weather:
        sequence.append("weather_cycle")
    return sequence


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    now = datetime.now(tz=UTC)

    if args.preset == "paper_live":
        schedule = _paper_live_schedule(now, include_weather=args.include_weather)
    else:
        modes = list(dict.fromkeys(args.modes or _default_modes(args.include_weather)))
        if not modes:
            print("[week] no modes requested")
            return
        schedule = [(mode, None) for mode in modes]

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
    for mode, run_date in schedule:
        run_args = ["--mode", mode, *base_flags]
        if run_date is not None:
            run_args.extend(["--when", run_date.isoformat()])
        print(f"[week] running mode: {mode} ({run_date.isoformat() if run_date else 'auto'})")
        daily.main(run_args)
        _print_outstanding("[week]")


def _print_outstanding(prefix: str) -> None:
    state = OutstandingOrdersState.load()
    summary = state.summary()
    dry = summary.get("dry", 0)
    live = summary.get("live", 0)
    total = dry + live
    print(f"{prefix} Outstanding orders -> total={total} (dry={dry}, live={live})")


def _paper_live_schedule(now: datetime, *, include_weather: bool) -> list[tuple[str, date | None]]:
    today_et = now.astimezone(ET).date()
    runs: list[tuple[str, date]] = []

    claims_window = resolve_run_window(mode="pre_claims", target_date=today_et, now=now)
    release_date = _window_date(claims_window, fallback=_next_weekday(today_et, target_weekday=3))
    freeze_date = _previous_business_day(release_date)
    runs.append(("pre_claims", freeze_date))
    runs.append(("pre_claims", release_date))

    cpi_window = resolve_run_window(mode="pre_cpi", target_date=release_date, now=now, proc_root=PROC_ROOT)
    cpi_release = _window_date(cpi_window, fallback=release_date)
    cpi_eve = _previous_business_day(cpi_release)
    runs.append(("pre_cpi", cpi_eve))
    runs.append(("pre_cpi", cpi_release))

    week_start = release_date - timedelta(days=release_date.weekday())
    for offset in range(5):
        ten_y_day = week_start + timedelta(days=offset)
        if ten_y_day.weekday() >= 5:
            continue
        runs.append(("teny_close", ten_y_day))
        if include_weather:
            runs.append(("weather_cycle", ten_y_day))

    ordered: list[tuple[str, date]] = []
    seen: set[tuple[str, date]] = set()
    for mode, when in sorted(runs, key=lambda item: (item[1], item[0])):
        key = (mode, when)
        if key not in seen:
            ordered.append(key)
            seen.add(key)
    return ordered


def _window_date(window, *, fallback: date) -> date:
    if window and window.reference is not None:
        return window.reference.astimezone(ET).date()
    return fallback


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
