"""Weekly orchestration wrapper running daily pipeline modes in sequence."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

from kalshi_alpha.exec.pipelines import daily


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
    modes = list(dict.fromkeys(args.modes or _default_modes(args.include_weather)))
    if not modes:
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
    if report_flag:
        base_flags.append("--report")
    if paper_flag:
        base_flags.append("--paper-ledger")

    print(f"[week] starting weekly run at {datetime.now(tz=UTC).isoformat()}")
    for mode in modes:
        print(f"[week] running mode: {mode}")
        daily.main(["--mode", mode, *base_flags])


if __name__ == "__main__":  # pragma: no cover
    main()
