"""CLI to ingest data snapshots across drivers."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import requests

from kalshi_alpha.core.datastore import ProcessedWriter
from kalshi_alpha.datastore.paths import PROC_ROOT
from kalshi_alpha.drivers import bls_cpi, cleveland_nowcast, dol_claims, treasury_yields
from kalshi_alpha.drivers.aaa_gas import fetch as aaa_fetch
from kalshi_alpha.drivers.aaa_gas import ingest as aaa_ingest
from kalshi_alpha.drivers.nws_cli import fetch_daily_climate_report, fetch_station_metadata


@dataclass
class IngestContext:
    offline: bool
    force_refresh: bool
    fixtures_root: Path | None
    session: requests.Session
    processed_writer: ProcessedWriter
    quiet: bool = False
    artifacts: list[Path] = field(default_factory=list)

    def fixture(self, name: str) -> Path | None:
        if self.fixtures_root is None:
            return None
        path = self.fixtures_root / name
        return path if path.exists() else None

    def record_artifact(self, path: Path | None) -> None:
        if path is not None:
            self.artifacts.append(path)

    def log(self, message: str) -> None:
        if not self.quiet:
            print(message)


def ingest_bls_cpi(ctx: IngestContext) -> None:
    ctx.log("[bls_cpi] fetching release calendar and latest release")
    fixtures = ctx.fixture("bls_cpi")
    calendar = bls_cpi.fetch_release_calendar(
        offline=ctx.offline,
        fixtures_dir=fixtures,
        force_refresh=ctx.force_refresh,
        session=ctx.session,
    )
    release = bls_cpi.fetch_latest_release(
        offline=ctx.offline,
        fixtures_dir=fixtures,
        force_refresh=ctx.force_refresh,
        session=ctx.session,
    )
    if calendar:
        records = [
            {
                "release_datetime": dt.astimezone(UTC),
                "release_date": dt.astimezone(UTC).date(),
            }
            for dt in calendar
        ]
        latest_calendar = max(record["release_datetime"] for record in records)
        calendar_path = ctx.processed_writer.write_parquet(
            "bls_cpi/calendar",
            pl.DataFrame(records),
            timestamp=latest_calendar,
        )
        ctx.record_artifact(calendar_path)
    release_ts = release.release_datetime.astimezone(UTC)
    release_frame = pl.DataFrame(
        {
            "release_datetime": [release_ts],
            "period": [release.period],
            "mom_sa": [release.mom_sa],
            "yoy_sa": [release.yoy_sa],
        }
    )
    release_path = ctx.processed_writer.write_parquet(
        "bls_cpi/latest_release",
        release_frame,
        timestamp=release_ts,
    )
    ctx.record_artifact(release_path)


def ingest_dol_claims(ctx: IngestContext) -> None:
    ctx.log("[dol_claims] fetching latest ETA-539 report")
    fixtures = ctx.fixture("dol_claims")
    report = dol_claims.fetch_latest_report(
        offline=ctx.offline,
        fixtures_dir=fixtures,
        force_refresh=ctx.force_refresh,
        session=ctx.session,
    )
    frame = dol_claims.latest_claims_dataframe(report)
    path = ctx.processed_writer.write_parquet(
        "dol_claims/latest_report",
        frame,
        timestamp=report.week_ending,
    )
    ctx.record_artifact(path)


def ingest_treasury(ctx: IngestContext) -> None:
    ctx.log("[treasury_yields] fetching daily par yields")
    fixtures = ctx.fixture("treasury_yields")
    yields = treasury_yields.fetch_daily_yields(
        offline=ctx.offline,
        fixtures_dir=fixtures,
        force_refresh=ctx.force_refresh,
        session=ctx.session,
    )
    if not yields:
        ctx.log("[treasury_yields] no yields returned; skipping processed snapshot")
        return
    frame = treasury_yields.yields_to_frame(yields)
    latest_as_of = max(entry.as_of for entry in yields)
    path = ctx.processed_writer.write_parquet(
        "treasury_yields/daily",
        frame,
        timestamp=latest_as_of,
    )
    ctx.record_artifact(path)


def ingest_cleveland(ctx: IngestContext) -> None:
    ctx.log("[cleveland_nowcast] fetching monthly headline/core nowcasts")
    fixtures = ctx.fixture("cleveland_nowcast")
    series = cleveland_nowcast.fetch_nowcast(
        offline=ctx.offline,
        fixtures_dir=fixtures,
        force_refresh=ctx.force_refresh,
        session=ctx.session,
    )
    records = [
        {
            "series": name,
            "label": item.label,
            "as_of": item.as_of.astimezone(UTC),
            "value": item.value,
        }
        for name, item in series.items()
    ]
    latest_as_of = max(record["as_of"] for record in records)
    frame = pl.DataFrame(records)
    path = ctx.processed_writer.write_parquet(
        "cleveland_nowcast/monthly",
        frame,
        timestamp=latest_as_of,
    )
    ctx.record_artifact(path)


def ingest_nws(ctx: IngestContext) -> None:
    ctx.log("[nws_cli] fetching station metadata and climate reports")
    fixtures = ctx.fixture("nws_cli")
    metadata = fetch_station_metadata(
        offline=ctx.offline,
        fixtures_dir=fixtures,
        force_refresh=ctx.force_refresh,
        session=ctx.session,
    )
    meta_records = [
        {"station_id": config.station_id, "name": config.name, "wban": config.wban}
        for config in metadata.values()
    ]
    if meta_records:
        meta_path = ctx.processed_writer.write_parquet(
            "nws_cli/stations",
            pl.DataFrame(meta_records),
            timestamp=datetime.now(tz=UTC),
        )
        ctx.record_artifact(meta_path)
    climate_records: list[dict[str, object]] = []
    for station_id in metadata.keys():
        try:
            report = fetch_daily_climate_report(
                station_id,
                offline=ctx.offline,
                fixtures_dir=fixtures,
                force_refresh=ctx.force_refresh,
                session=ctx.session,
            )
        except Exception as exc:  # pragma: no cover - robustness
            ctx.log(f"[nws_cli] failed to fetch DCR for {station_id}: {exc}")
            continue
        climate_records.append(
            {
                "station_id": report.station_id,
                "record_date": report.record_date,
                "high_temp_f": report.high_temp_f,
                "low_temp_f": report.low_temp_f,
            }
        )
    if climate_records:
        latest_date = max(record["record_date"] for record in climate_records)
        timestamp = datetime.combine(latest_date, datetime.min.time(), tzinfo=UTC)
        climate_path = ctx.processed_writer.write_parquet(
            "nws_cli/daily_climate",
            pl.DataFrame(climate_records),
            timestamp=timestamp,
        )
        ctx.record_artifact(climate_path)


def ingest_aaa(ctx: IngestContext) -> None:
    ctx.log("[aaa_gas] ingesting AAA gasoline prices")
    fixtures = ctx.fixture("aaa")
    bootstrap_csv = fixtures / "AAA_daily_gas_price_regular_sample.csv" if fixtures else None
    if bootstrap_csv and not aaa_fetch.DAILY_PATH.exists():
        aaa_ingest.bootstrap_from_csv(bootstrap_csv)
    result = aaa_fetch.fetch_latest(
        offline=ctx.offline,
        fixtures_dir=fixtures,
        force_refresh=ctx.force_refresh,
        session=ctx.session,
    )
    ctx.record_artifact(aaa_fetch.DAILY_PATH if aaa_fetch.DAILY_PATH.exists() else None)
    ctx.record_artifact(aaa_fetch.MONTHLY_PATH if aaa_fetch.MONTHLY_PATH.exists() else None)
    if result:
        ctx.log(
            f"[aaa_gas] latest price {result.price:.3f} USD "
            f"as of {result.as_of_date.isoformat()}"
        )


SourceFunc = Callable[[IngestContext], None]

SOURCES: dict[str, SourceFunc] = {
    "bls_cpi": ingest_bls_cpi,
    "dol_claims": ingest_dol_claims,
    "treasury_yields": ingest_treasury,
    "cleveland_nowcast": ingest_cleveland,
    "nws_cli": ingest_nws,
    "aaa_gas": ingest_aaa,
}


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Datastore ingestion CLI.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--online", action="store_true", help="Use live HTTP fetchers.")
    mode.add_argument("--offline", action="store_true", help="Use offline fixtures.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ingest all configured sources.",
    )
    parser.add_argument(
        "--source",
        dest="sources",
        action="append",
        choices=sorted(SOURCES.keys()),
        help="Specific source(s) to ingest. May be repeated.",
    )
    parser.add_argument(
        "--fixtures",
        type=Path,
        help="Root directory for offline fixtures.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Bypass cached HTTP responses and revalidate upstream.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress stdout progress logs.",
    )
    return parser.parse_args(argv)


def _resolve_sources(args: argparse.Namespace) -> list[str]:
    if args.all and args.sources:
        raise ValueError("--all may not be combined with --source")
    if args.all or not args.sources:
        return list(SOURCES.keys())
    return list(dict.fromkeys(args.sources))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        sources = _resolve_sources(args)
    except ValueError as exc:  # pragma: no cover - argument error surface
        raise SystemExit(str(exc)) from exc

    fixtures = args.fixtures
    if args.offline and fixtures is None:
        default_fixtures = Path("tests/fixtures")
        if default_fixtures.exists():
            fixtures = default_fixtures
        else:
            raise ValueError("--fixtures is required for offline ingestion")

    session = requests.Session()
    writer = ProcessedWriter(PROC_ROOT)
    ctx = IngestContext(
        offline=args.offline,
        force_refresh=args.force_refresh,
        fixtures_root=fixtures,
        session=session,
        processed_writer=writer,
        quiet=args.quiet,
    )

    for source in sources:
        handler = SOURCES[source]
        handler(ctx)

    if not args.quiet:
        if ctx.artifacts:
            ctx.log("Artifacts written:")
            for path in ctx.artifacts:
                ctx.log(f"  - {path}")
        else:
            ctx.log("No processed artifacts were produced.")


if __name__ == "__main__":
    main()
