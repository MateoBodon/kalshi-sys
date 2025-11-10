"""Command-line entry point for runtime monitors."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from kalshi_alpha.exec.heartbeat import resolve_kill_switch_path

from .runtime import (
    ALPHA_STATE_PATH,
    LEDGER_PATH,
    MONITOR_ARTIFACTS_DIR,
    TELEMETRY_ROOT,
    MonitorResult,
    RuntimeMonitorConfig,
    build_report_summary,
    compute_runtime_monitors,
    write_monitor_artifacts,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    defaults = RuntimeMonitorConfig()
    parser = argparse.ArgumentParser(description="Compute execution runtime monitors.")
    parser.add_argument(
        "--telemetry-root",
        type=Path,
        default=TELEMETRY_ROOT,
        help="Telemetry base directory. Defaults to data/raw/kalshi.",
    )
    parser.add_argument(
        "--ledger-path",
        type=Path,
        default=LEDGER_PATH,
        help="Aggregated ledger parquet path. Defaults to data/proc/ledger_all.parquet.",
    )
    parser.add_argument(
        "--alpha-state-path",
        type=Path,
        default=ALPHA_STATE_PATH,
        help="Fill alpha state JSON. Defaults to data/proc/state/fill_alpha.json.",
    )
    parser.add_argument(
        "--drawdown-state-dir",
        type=Path,
        default=None,
        help="Override state directory for drawdown tracking. Defaults to data/proc.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=MONITOR_ARTIFACTS_DIR,
        help="Directory for monitor JSON artifacts. Defaults to reports/_artifacts/monitors.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("REPORT.md"),
        help="Path to REPORT.md for summary injection.",
    )
    parser.add_argument("--no-report", action="store_true", help="Skip updating REPORT.md.")
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=defaults.telemetry_lookback_hours,
        help="Telemetry lookback window in hours.",
    )
    parser.add_argument(
        "--ledger-lookback-days",
        type=int,
        default=defaults.ledger_lookback_days,
        help="Ledger lookback window in days.",
    )
    parser.add_argument(
        "--daily-loss-cap",
        type=float,
        default=defaults.daily_loss_cap,
        help="Daily drawdown cap in USD.",
    )
    parser.add_argument(
        "--weekly-loss-cap",
        type=float,
        default=defaults.weekly_loss_cap,
        help="Weekly drawdown cap in USD.",
    )
    parser.add_argument(
        "--kill-switch-file",
        type=Path,
        default=None,
        help="Path to kill-switch sentinel file (default: data/proc/state/kill_switch).",
    )
    parser.add_argument(
        "--ws-disconnect-threshold",
        type=float,
        default=defaults.ws_disconnect_rate_threshold,
        help="Maximum websocket disconnects per hour before alert.",
    )
    parser.add_argument(
        "--auth-error-threshold",
        type=int,
        default=defaults.auth_error_streak_threshold,
        help="Maximum tolerated auth reject streak.",
    )
    parser.add_argument(
        "--fill-min-contracts",
        type=int,
        default=defaults.fill_min_contracts,
        help="Minimum requested contracts to include a series in fill monitors.",
    )
    parser.add_argument(
        "--seq-threshold",
        type=float,
        default=defaults.seq_cusum_threshold,
        help="Sequential CuSum alert threshold in Δbps.",
    )
    parser.add_argument(
        "--seq-drift",
        type=float,
        default=defaults.seq_cusum_drift,
        help="Sequential CuSum drift parameter in Δbps.",
    )
    parser.add_argument(
        "--seq-min-sample",
        type=int,
        default=defaults.seq_min_sample,
        help="Minimum trades per series for sequential guard evaluation.",
    )
    parser.add_argument(
        "--freeze-series",
        nargs="+",
        help="Override the list of series evaluated for freeze windows (default: CPI, CLAIMS, TENY and ledger-derived series).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    moment = datetime.now(tz=UTC)
    kill_switch_path = resolve_kill_switch_path(args.kill_switch_file)
    config = RuntimeMonitorConfig(
        telemetry_lookback_hours=args.lookback_hours,
        ledger_lookback_days=args.ledger_lookback_days,
        daily_loss_cap=args.daily_loss_cap,
        weekly_loss_cap=args.weekly_loss_cap,
        ws_disconnect_rate_threshold=args.ws_disconnect_threshold,
        auth_error_streak_threshold=args.auth_error_threshold,
        fill_min_contracts=args.fill_min_contracts,
        kill_switch_path=kill_switch_path,
        seq_cusum_threshold=args.seq_threshold,
        seq_cusum_drift=args.seq_drift,
        seq_min_sample=args.seq_min_sample,
        freeze_series=tuple(s.upper() for s in args.freeze_series) if args.freeze_series else None,
    )

    results = compute_runtime_monitors(
        config=config,
        telemetry_root=args.telemetry_root,
        ledger_path=args.ledger_path,
        alpha_state_path=args.alpha_state_path,
        drawdown_state_dir=args.drawdown_state_dir,
        now=moment,
    )

    write_monitor_artifacts(results, artifacts_dir=args.artifacts_dir, generated_at=moment)
    summary_lines = build_report_summary(results)
    if not summary_lines:
        summary_lines = ["- ⬜ No runtime monitor data"]
    for line in summary_lines:
        print(line)

    if not args.no_report:
        _update_report(args.report, summary_lines, moment)
    _maybe_notify(summary_lines, results)
    return 0


def _update_report(report_path: Path, summary_lines: Sequence[str], generated_at: datetime) -> None:
    marker_start = "<!-- monitors:start -->"
    marker_end = "<!-- monitors:end -->"
    generated_line = f"_Generated {generated_at.strftime('%Y-%m-%d %H:%M:%S %Z')}_"
    body = "\n".join([generated_line, "", *summary_lines])

    if report_path.exists():
        contents = report_path.read_text(encoding="utf-8")
    else:
        contents = "# Execution Snapshot\n"

    if marker_start in contents and marker_end in contents:
        pre, remainder = contents.split(marker_start, 1)
        _, post = remainder.split(marker_end, 1)
        new_contents = (
            pre
            + marker_start
            + "\n"
            + body
            + "\n"
            + marker_end
            + post
        )
    else:
        section = (
            "\n\n## Runtime Monitors\n"
            f"{marker_start}\n{body}\n{marker_end}\n"
        )
        new_contents = contents.rstrip() + section

    report_path.write_text(new_contents.rstrip() + "\n", encoding="utf-8")


def _maybe_notify(summary: Sequence[str], results: Sequence[MonitorResult]) -> None:
    webhook = os.getenv("KALSHI_MONITOR_SLACK_WEBHOOK")
    if not webhook:
        return
    if not webhook.lower().startswith("https://"):
        print("[monitors] Slack webhook must be https://", file=sys.stderr)
        return
    worst = next((r for r in results if r.status == "ALERT"), None)
    title = "Runtime monitors: ALERT" if worst else "Runtime monitors: OK"
    payload = {
        "text": "\n".join([title, *summary]),
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(  # noqa: S310
        webhook,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(request, timeout=5)  # noqa: S310
    except Exception as exc:  # pragma: no cover - notification is best effort
        print(f"[monitors] Slack notification failed: {exc}", file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
