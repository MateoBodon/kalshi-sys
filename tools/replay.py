"""Replay recorded Kalshi sessions to validate EV parity."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

import polars as pl

from kalshi_alpha.core.archive.replay import replay_manifest
from kalshi_alpha.risk.var_index import SERIES_FAMILY

ET = ZoneInfo("America/New_York")
RAW_KALSHI_ROOT = Path("data/raw/kalshi")
DEFAULT_FAMILIES = ("SPX", "NDX")
DEFAULT_HOURS = (10, 11, 12, 13, 14, 15, 16)
REPLAY_FILENAME = "replay_ev.parquet"
SUMMARY_DIRNAME = "replay"
SUMMARY_TEMPLATE = "replay_summary_{date}.json"
PLOT_TEMPLATE = "replay_delta_{date}.png"

SERIES_WINDOW_TYPE = {
    "INXU": "hourly",
    "NASDAQ100U": "hourly",
    "INX": "close",
    "NASDAQ100": "close",
}


@dataclass(slots=True)
class ReplayRun:
    manifest_path: Path
    series: str
    family: str
    window_type: str
    event_time: datetime
    generated_at: datetime
    window_start: datetime
    window_end: datetime
    hour_et: int

    @property
    def label(self) -> str:
        et_time = self.event_time.astimezone(ET)
        return f"{self.series} {et_time:%Y-%m-%d %H:%M}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay recorded Kalshi maker sessions (T-5m..T-2s).")
    parser.add_argument(
        "--families",
        default=",".join(DEFAULT_FAMILIES),
        help="Comma-separated family tickers (default: SPX,NDX).",
    )
    parser.add_argument(
        "--date",
        default="yesterday",
        help="Trading day in US/Eastern (YYYY-MM-DD|today|yesterday).",
    )
    parser.add_argument(
        "--hours",
        default=",".join(str(hour) for hour in DEFAULT_HOURS),
        help="Comma-separated ET hours to replay (default: 10-16). Include 16 for EOD windows.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.15,
        help="ΔEV parity allowance per contract (USD, default: 0.15).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/_artifacts"),
        help="Artifact directory (parquet + summary + plots).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    target_date = _resolve_date(args.date)
    families = _parse_families(args.families)
    hours = _parse_hours(args.hours)
    runs = _discover_runs(
        target_date=target_date,
        families=families,
        hours=hours,
        raw_root=RAW_KALSHI_ROOT,
    )
    if not runs:
        raise SystemExit(
            f"no replay manifests found for {target_date.isoformat()} "
            f"(families={sorted(families)}, hours={sorted(hours)})"
        )
    frames: list[pl.DataFrame] = []
    for run in runs:
        frame = _replay_single(run)
        if frame is not None and not frame.is_empty():
            frames.append(frame)
    if not frames:
        raise SystemExit("replay produced no rows; ensure proposals/orderbooks exist for requested date")
    combined = pl.concat(frames, how="diagonal")
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / REPLAY_FILENAME
    combined.write_parquet(output_path)
    summary = _summarize(combined, target_date, epsilon=args.epsilon)
    summary_dir = out_dir / SUMMARY_DIRNAME
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / SUMMARY_TEMPLATE.format(date=target_date.isoformat())
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_path = summary_dir / PLOT_TEMPLATE.format(date=target_date.isoformat())
    _write_plot(summary["windows"], args.epsilon, plot_path)
    print(f"[replay] wrote {output_path}")
    print(f"[replay] summary stored at {summary_path}")
    print(f"[replay] max ΔEV by type: {json.dumps(summary['window_type_max'], sort_keys=True)}")
    return 0


def _resolve_date(label: str) -> date:
    today_et = datetime.now(tz=ET).date()
    if not label or label.lower() == "today":
        return today_et
    if label.lower() == "yesterday":
        return today_et - timedelta(days=1)
    return date.fromisoformat(label)


def _parse_families(raw: str) -> set[str]:
    tokens = [token.strip().upper() for token in (raw or "").split(",") if token.strip()]
    families = {token for token in tokens if token}
    return families or set(DEFAULT_FAMILIES)


def _parse_hours(raw: str) -> set[int]:
    if not raw or raw.lower() == "all":
        return set(range(0, 24))
    result: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if token.isdigit():
            hour = int(token)
            if 0 <= hour <= 23:
                result.add(hour)
    return result or set(DEFAULT_HOURS)


def _discover_runs(
    *,
    target_date: date,
    families: set[str],
    hours: set[int],
    raw_root: Path,
) -> list[ReplayRun]:
    day_dir = raw_root / target_date.isoformat()
    if not day_dir.exists():
        return []
    runs: list[ReplayRun] = []
    for timestamp_dir in sorted(path for path in day_dir.iterdir() if path.is_dir()):
        generated_at = _timestamp_from_dir(timestamp_dir.name, target_date)
        for series_dir in sorted(path for path in timestamp_dir.iterdir() if path.is_dir()):
            series = series_dir.name.upper()
            window_type = SERIES_WINDOW_TYPE.get(series)
            family = SERIES_FAMILY.get(series)
            if window_type is None or family is None or family not in families:
                continue
            manifest_path = series_dir / "manifest.json"
            markets_path = series_dir / "markets.json"
            if not manifest_path.exists() or not markets_path.exists():
                continue
            manifest = _read_json(manifest_path)
            if not isinstance(manifest, dict):
                continue
            generated_at_manifest = manifest.get("generated_at")
            generated_at_dt = _parse_timestamp(generated_at_manifest) if generated_at_manifest else generated_at
            if generated_at_dt is None:
                continue
            market_hour = _market_hour(markets_path)
            if market_hour is None or market_hour not in hours:
                continue
            event_time = datetime.combine(target_date, time(hour=market_hour), tzinfo=ET)
            window_start = event_time - timedelta(minutes=5)
            window_end = event_time - timedelta(seconds=2)
            generated_at_local = generated_at_dt.astimezone(ET)
            if not (window_start <= generated_at_local <= window_end):
                continue
            runs.append(
                ReplayRun(
                    manifest_path=manifest_path,
                    series=series,
                    family=family,
                    window_type=window_type,
                    event_time=event_time,
                    generated_at=generated_at_local,
                    window_start=window_start,
                    window_end=window_end,
                    hour_et=market_hour,
                )
            )
    runs.sort(key=lambda run: (run.event_time, run.generated_at, run.series))
    return runs


def _market_hour(path: Path) -> int | None:
    payload = _read_json(path)
    if isinstance(payload, list) and payload:
        sample = payload[0]
    elif isinstance(payload, dict):
        sample = payload
    else:
        return None
    identifier = str(sample.get("ticker") or sample.get("id") or "")
    match = re.search(r"H(\d{2})(\d{2})", identifier)
    if not match:
        return None
    hour = int(match.group(1))
    return hour


def _timestamp_from_dir(name: str, base_date: date) -> datetime | None:
    if not name.isdigit() or len(name) not in (6, 9):
        return None
    hour = int(name[:2])
    minute = int(name[2:4])
    second = int(name[4:6])
    return datetime(
        year=base_date.year,
        month=base_date.month,
        day=base_date.day,
        hour=hour,
        minute=minute,
        second=second,
        tzinfo=ET,
    )


def _replay_single(run: ReplayRun) -> pl.DataFrame | None:
    try:
        replay_path = replay_manifest(run.manifest_path)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[replay] replay_manifest failed for {run.manifest_path}: {exc}")
        return None
    if not replay_path.exists():
        return None
    try:
        frame = pl.read_parquet(replay_path)
    except Exception as exc:  # pragma: no cover - corrupted artifact
        print(f"[replay] unable to read parquet for {run.manifest_path}: {exc}")
        return None
    if frame.is_empty():
        return frame
    per_contract = (
        (pl.col("maker_ev_original") / pl.col("contracts").clip(lower_bound=1))
        .fill_null(0.0)
        .alias("maker_ev_per_contract_original")
    )
    delta_raw = pl.col("maker_ev_per_contract_replay") - per_contract
    enriched = frame.with_columns(
        [
            pl.lit(run.family).alias("family"),
            pl.lit(run.window_type).alias("window_type"),
            pl.lit(run.event_time.astimezone(ET).isoformat()).alias("event_time_et"),
            pl.lit(run.generated_at.isoformat()).alias("generated_at_et"),
            pl.lit(run.window_start.isoformat()).alias("window_start_et"),
            pl.lit(run.window_end.isoformat()).alias("window_end_et"),
            pl.lit(run.label).alias("window_label"),
            pl.lit(run.hour_et).alias("hour_et"),
            per_contract,
            delta_raw.alias("delta_per_contract"),
            delta_raw.abs().alias("abs_delta_per_contract"),
        ]
    )
    return enriched


def _summarize(frame: pl.DataFrame, target_date: date, *, epsilon: float) -> dict[str, object]:
    lookup: dict[str, dict[str, object]] = {}
    worst_rows: list[dict[str, object]] = []
    partitions = frame.partition_by("window_label", maintain_order=True)
    for subset in partitions:
        window_label = subset["window_label"][0]
        max_abs_delta = float(subset["abs_delta_per_contract"].max())
        metrics = {
            "series": subset["series"][0],
            "family": subset["family"][0],
            "window_type": subset["window_type"][0],
            "event_time": subset["event_time_et"][0],
            "window_start": subset["window_start_et"][0],
            "window_end": subset["window_end_et"][0],
            "generated_at": subset["generated_at_et"][0],
            "hour_et": int(subset["hour_et"][0]) if "hour_et" in subset.columns else None,
            "records": subset.height,
            "max_abs_delta": max_abs_delta,
            "mean_abs_delta": float(subset["abs_delta_per_contract"].mean()),
            "threshold_breach": max_abs_delta > epsilon,
            "window_label": window_label,
        }
        window_worst = (
            subset.sort("abs_delta_per_contract", descending=True)
            .head(3)
            .select(
                [
                    "market_id",
                    "market_ticker",
                    "strike",
                    "side",
                    "manifest",
                    "maker_ev_per_contract_original",
                    "maker_ev_per_contract_replay",
                    "delta_per_contract",
                    "abs_delta_per_contract",
                ]
            )
            .to_dicts()
        )
        metrics["worst_bins"] = window_worst
        lookup[window_label] = metrics
        worst_rows.extend(window_worst)
    window_type_max: dict[str, float] = {}
    if not lookup:
        return {
            "date": target_date.isoformat(),
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "epsilon": epsilon,
            "runs": 0,
            "rows": 0,
            "windows": [],
            "window_type_max": window_type_max,
            "worst_bins": [],
        }
    for metrics in lookup.values():
        window_type = str(metrics["window_type"])
        window_type_max[window_type] = max(window_type_max.get(window_type, 0.0), float(metrics["max_abs_delta"]))
    worst_rows.sort(key=lambda row: row["abs_delta_per_contract"], reverse=True)
    summary = {
        "date": target_date.isoformat(),
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "epsilon": epsilon,
        "runs": len(partitions),
        "rows": frame.height,
        "windows": list(lookup.values()),
        "window_type_max": window_type_max,
        "worst_bins": worst_rows[:5],
    }
    return summary


def _write_plot(windows: Sequence[dict[str, object]], epsilon: float, output: Path) -> None:
    if not windows:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"")
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:  # pragma: no cover - matplotlib optional
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"")
        return
    labels = [window["window_label"] for window in windows]
    values = [float(window["max_abs_delta"]) for window in windows]
    indices = range(len(labels))
    colors = ["#1f77b4" if window["window_type"] == "hourly" else "#d62728" for window in windows]
    fig, ax = plt.subplots(figsize=(max(4, len(labels)), 3))
    ax.bar(indices, values, color=colors)
    ax.axhline(epsilon, color="#ff7f0e", linestyle="--", linewidth=1.0, label=f"ε ({epsilon:.2f})")
    ax.set_ylabel("|ΔEV| per contract (USD)")
    ax.set_title("Replay Parity by Window")
    ax.set_xticks(list(indices))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _read_json(path: Path) -> dict | list | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _parse_timestamp(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    return None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
