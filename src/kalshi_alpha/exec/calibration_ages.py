"""Calibration age inspection and reporting for index ladders."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

CALIBRATION_ROOT = Path("data/proc/calib/index")
DEFAULT_MAX_AGE_DAYS = 14.0
DEFAULT_HOURLY_HOURS = (10, 11, 12, 13, 14, 15, 16)

SERIES_SLUGS = {
    "INXU": "spx",
    "INX": "spx",
    "NASDAQ100U": "ndx",
    "NASDAQ100": "ndx",
}

HOURLY_SERIES = ("INXU", "NASDAQ100U")
CLOSE_SERIES = ("INX", "NASDAQ100")
SERIES_ORDER = ("INXU", "NASDAQ100U", "INX", "NASDAQ100")


@dataclass(slots=True)
class CalibrationAgeResult:
    series: str
    horizon: str
    file_path: str
    mtime_iso: str | None
    age_hours: float | None
    status: str
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "series": self.series,
            "horizon": self.horizon,
            "file_path": self.file_path,
            "mtime_iso": self.mtime_iso,
            "age_hours": self.age_hours,
            "status": self.status,
            "reason": self.reason,
        }


@dataclass(slots=True)
class CalibrationSeriesSummary:
    series: str
    status: str
    age_hours: float | None
    age_days: float | None
    reason: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "series": self.series,
            "status": self.status,
            "age_hours": self.age_hours,
            "age_days": self.age_days,
            "reason": self.reason,
        }


def _ensure_utc(moment: datetime | None) -> datetime:
    if moment is None:
        return datetime.now(tz=UTC)
    return moment.astimezone(UTC) if moment.tzinfo else moment.replace(tzinfo=UTC)


def _format_age(age_hours: float | None) -> str:
    return "n/a" if age_hours is None else f"{age_hours:.1f}"


def _format_timestamp(value: str | None) -> str:
    return value or "n/a"


def _hourly_candidates(root: Path, slug: str, hour: int) -> list[Path]:
    variant = root / slug / "hourly" / f"{hour:02d}00" / "params.json"
    aggregate = root / slug / "hourly" / "params.json"
    legacy = root / slug / "noon" / "params.json"
    return [variant, aggregate, legacy]


def _close_candidates(root: Path, slug: str) -> list[Path]:
    return [root / slug / "close" / "params.json"]


def _age_from_mtime(path: Path, now_utc: datetime) -> tuple[float | None, str | None]:
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    except OSError:
        return None, None
    age_seconds = (now_utc - mtime).total_seconds()
    age_hours = max(age_seconds, 0.0) / 3600.0
    return age_hours, mtime.isoformat()


def _evaluate_candidates(
    *,
    series: str,
    horizon: str,
    candidates: Sequence[Path],
    now_utc: datetime,
    max_age_hours: float,
) -> CalibrationAgeResult:
    primary = candidates[0]
    chosen: Path | None = None
    chosen_index = 0
    missing_primary: list[str] = []
    for idx, candidate in enumerate(candidates):
        if candidate.exists():
            chosen = candidate
            chosen_index = idx
            break
        missing_primary.append(candidate.as_posix())

    file_path = (chosen or primary).as_posix()
    if chosen is None:
        reason = "missing: " + ", ".join(missing_primary)
        return CalibrationAgeResult(
            series=series,
            horizon=horizon,
            file_path=file_path,
            mtime_iso=None,
            age_hours=None,
            status="MISSING",
            reason=reason,
        )

    age_hours, mtime_iso = _age_from_mtime(chosen, now_utc)
    if age_hours is None:
        return CalibrationAgeResult(
            series=series,
            horizon=horizon,
            file_path=file_path,
            mtime_iso=None,
            age_hours=None,
            status="MISSING",
            reason="mtime_unavailable",
        )

    reason_parts: list[str] = []
    if chosen_index > 0 and missing_primary:
        reason_parts.append("fallback_used: " + chosen.as_posix())
        reason_parts.append("missing_primary: " + ", ".join(missing_primary))

    status = "OK"
    if age_hours > max_age_hours:
        status = "STALE"
        reason_parts.append(f"age_hours {age_hours:.1f} > {max_age_hours:.1f}")

    reason = "; ".join(reason_parts) if reason_parts else None
    return CalibrationAgeResult(
        series=series,
        horizon=horizon,
        file_path=file_path,
        mtime_iso=mtime_iso,
        age_hours=age_hours,
        status=status,
        reason=reason,
    )


def inspect_calibration_ages(
    *,
    now: datetime | None = None,
    root: Path = CALIBRATION_ROOT,
    max_age_days: float = DEFAULT_MAX_AGE_DAYS,
    series: Sequence[str] | None = None,
    hourly_hours: Sequence[int] = DEFAULT_HOURLY_HOURS,
) -> list[CalibrationAgeResult]:
    now_utc = _ensure_utc(now)
    max_age_hours = max_age_days * 24.0
    series_list = [s.upper() for s in (series or SERIES_ORDER)]
    results: list[CalibrationAgeResult] = []

    for entry in series_list:
        slug = SERIES_SLUGS.get(entry)
        if not slug:
            continue
        if entry in HOURLY_SERIES:
            for hour in hourly_hours:
                horizon = f"hourly-{int(hour):02d}00"
                candidates = _hourly_candidates(root, slug, int(hour))
                results.append(
                    _evaluate_candidates(
                        series=entry,
                        horizon=horizon,
                        candidates=candidates,
                        now_utc=now_utc,
                        max_age_hours=max_age_hours,
                    )
                )
        else:
            candidates = _close_candidates(root, slug)
            results.append(
                _evaluate_candidates(
                    series=entry,
                    horizon="close",
                    candidates=candidates,
                    now_utc=now_utc,
                    max_age_hours=max_age_hours,
                )
            )
    return results


def summarize_by_series(
    results: Iterable[CalibrationAgeResult],
) -> dict[str, CalibrationSeriesSummary]:
    grouped: dict[str, list[CalibrationAgeResult]] = {}
    for entry in results:
        grouped.setdefault(entry.series, []).append(entry)

    summaries: dict[str, CalibrationSeriesSummary] = {}
    for series, entries in grouped.items():
        missing = [item for item in entries if item.status == "MISSING"]
        stale = [item for item in entries if item.status == "STALE"]
        ok = [item for item in entries if item.status == "OK"]
        if missing:
            reason = "missing: " + ", ".join(item.file_path for item in missing)
            summaries[series] = CalibrationSeriesSummary(
                series=series,
                status="MISSING",
                age_hours=None,
                age_days=None,
                reason=reason,
            )
            continue
        if stale:
            age_hours = max(item.age_hours or 0.0 for item in stale)
            reason = "stale: " + ", ".join(item.file_path for item in stale)
            summaries[series] = CalibrationSeriesSummary(
                series=series,
                status="STALE",
                age_hours=age_hours,
                age_days=age_hours / 24.0 if age_hours is not None else None,
                reason=reason,
            )
            continue
        age_hours = max((item.age_hours or 0.0) for item in ok) if ok else None
        summaries[series] = CalibrationSeriesSummary(
            series=series,
            status="OK",
            age_hours=age_hours,
            age_days=age_hours / 24.0 if age_hours is not None else None,
            reason=None,
        )
    return summaries


def render_markdown(
    results: Sequence[CalibrationAgeResult],
    *,
    asof_date: date,
    generated_at: datetime,
) -> str:
    lines = [
        f"# Calibration Ages ({asof_date.isoformat()})",
        "",
        f"Generated at (UTC): {generated_at.astimezone(UTC).isoformat()}",
        "",
        "| Series | Horizon | File | MTime (UTC) | Age (hours) | Status | Reason |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in results:
        lines.append(
            "| {series} | {horizon} | {file_path} | {mtime} | {age} | {status} | {reason} |".format(
                series=entry.series,
                horizon=entry.horizon,
                file_path=entry.file_path,
                mtime=_format_timestamp(entry.mtime_iso),
                age=_format_age(entry.age_hours),
                status=entry.status,
                reason=entry.reason or "",
            )
        )
    return "\n".join(lines) + "\n"


def write_report(
    results: Sequence[CalibrationAgeResult],
    *,
    asof_date: date,
    output_path: Path,
    generated_at: datetime,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_markdown(results, asof_date=asof_date, generated_at=generated_at),
        encoding="utf-8",
    )
    return output_path


def _parse_date(value: str | None, *, now_et: datetime) -> date:
    if value:
        return date.fromisoformat(value)
    return now_et.date()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report calibration ages for index ladders.")
    parser.add_argument("--asof-date", help="Override as-of date (YYYY-MM-DD).")
    parser.add_argument("--root", type=Path, default=CALIBRATION_ROOT, help="Calibration root directory.")
    parser.add_argument("--output", type=Path, default=None, help="Override output markdown path.")
    parser.add_argument(
        "--max-age-days",
        type=float,
        default=DEFAULT_MAX_AGE_DAYS,
        help="Staleness threshold in days (default: %(default)s).",
    )
    parser.add_argument(
        "--series",
        nargs="+",
        default=None,
        help="Restrict to series (INXU, NASDAQ100U, INX, NASDAQ100).",
    )
    parser.add_argument(
        "--hourly-hours",
        nargs="+",
        type=int,
        default=list(DEFAULT_HOURLY_HOURS),
        help="Hourly target hours to check (default: 10 11 12 13 14 15 16).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    now_utc = datetime.now(tz=UTC)
    now_et = now_utc.astimezone(ET)
    asof = _parse_date(args.asof_date, now_et=now_et)
    output = (
        Path(args.output)
        if args.output is not None
        else Path("reports/calibration") / f"calibration_ages_{asof.isoformat()}.md"
    )
    results = inspect_calibration_ages(
        now=now_utc,
        root=Path(args.root),
        max_age_days=float(args.max_age_days),
        series=args.series,
        hourly_hours=tuple(args.hourly_hours),
    )
    write_report(results, asof_date=asof, output_path=output, generated_at=now_utc)
    print(f"[calibration_ages] wrote {output}")
    return 0


__all__ = [
    "CALIBRATION_ROOT",
    "CalibrationAgeResult",
    "CalibrationSeriesSummary",
    "DEFAULT_HOURLY_HOURS",
    "DEFAULT_MAX_AGE_DAYS",
    "HOURLY_SERIES",
    "SERIES_ORDER",
    "SERIES_SLUGS",
    "inspect_calibration_ages",
    "main",
    "render_markdown",
    "summarize_by_series",
    "write_report",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
