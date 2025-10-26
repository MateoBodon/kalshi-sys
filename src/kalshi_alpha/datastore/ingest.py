"""CLI to ingest data snapshots across drivers."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import requests

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

    def fixture(self, name: str) -> Path | None:
        if self.fixtures_root is None:
            return None
        path = self.fixtures_root / name
        return path if path.exists() else None


def ingest_bls_cpi(ctx: IngestContext) -> None:
    fixtures = ctx.fixture("bls_cpi")
    bls_cpi.fetch_release_calendar(
        offline=ctx.offline,
        fixtures_dir=fixtures,
        force_refresh=ctx.force_refresh,
        session=ctx.session,
    )
    bls_cpi.fetch_latest_release(
        offline=ctx.offline,
        fixtures_dir=fixtures,
        force_refresh=ctx.force_refresh,
        session=ctx.session,
    )


def ingest_dol_claims(ctx: IngestContext) -> None:
    fixtures = ctx.fixture("dol_claims")
    dol_claims.fetch_latest_report(
        offline=ctx.offline,
        fixtures_dir=fixtures,
        force_refresh=ctx.force_refresh,
        session=ctx.session,
    )


def ingest_treasury(ctx: IngestContext) -> None:
    fixtures = ctx.fixture("treasury_yields")
    treasury_yields.fetch_daily_yields(
        offline=ctx.offline,
        fixtures_dir=fixtures,
        force_refresh=ctx.force_refresh,
        session=ctx.session,
    )


def ingest_cleveland(ctx: IngestContext) -> None:
    fixtures = ctx.fixture("cleveland_nowcast")
    cleveland_nowcast.fetch_nowcast(
        offline=ctx.offline,
        fixtures_dir=fixtures,
        force_refresh=ctx.force_refresh,
        session=ctx.session,
    )


def ingest_nws(ctx: IngestContext) -> None:
    fixtures = ctx.fixture("nws_cli")
    metadata = fetch_station_metadata(
        offline=ctx.offline,
        fixtures_dir=fixtures,
        force_refresh=ctx.force_refresh,
        session=ctx.session,
    )
    for station in metadata.keys():
        try:
            fetch_daily_climate_report(
                station,
                offline=ctx.offline,
                fixtures_dir=fixtures,
                force_refresh=ctx.force_refresh,
                session=ctx.session,
            )
        except Exception as exc:  # pragma: no cover - logging only
            print(f"Failed to fetch NWS report for {station}: {exc}")


def ingest_aaa(ctx: IngestContext) -> None:
    fixtures = ctx.fixture("aaa")
    bootstrap_csv = fixtures / "AAA_daily_gas_price_regular_sample.csv" if fixtures else None
    if bootstrap_csv and not aaa_fetch.DAILY_PATH.exists():
        aaa_ingest.bootstrap_from_csv(bootstrap_csv)
    aaa_fetch.fetch_latest(
        offline=ctx.offline,
        fixtures_dir=fixtures,
        force_refresh=ctx.force_refresh,
        session=ctx.session,
    )


SOURCES: dict[str, Callable[[IngestContext], None]] = {
    "bls_cpi": ingest_bls_cpi,
    "dol_claims": ingest_dol_claims,
    "treasury_yields": ingest_treasury,
    "cleveland_nowcast": ingest_cleveland,
    "nws_cli": ingest_nws,
    "aaa_gas": ingest_aaa,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Datastore ingestion CLI.")
    parser.add_argument("--offline", action="store_true", help="Use offline fixtures.")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore HTTP cache metadata.")
    parser.add_argument("--fixtures", type=Path, help="Root directory for offline fixtures.")
    parser.add_argument(
        "--source",
        action="append",
        choices=sorted(SOURCES.keys()),
        help="Specific sources to ingest. Omit to run all.",
    )
    args = parser.parse_args(argv)
    session = requests.Session()
    ctx = IngestContext(
        offline=args.offline,
        force_refresh=args.force_refresh,
        fixtures_root=args.fixtures,
        session=session,
    )
    sources = args.source or list(SOURCES.keys())
    for source in sources:
        SOURCES[source](ctx)


if __name__ == "__main__":
    main()
