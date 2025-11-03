from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from kalshi_alpha.datastore import ingest as datastore_ingest
from kalshi_alpha.drivers import bls_cpi, cleveland_nowcast, dol_claims, treasury_yields
from kalshi_alpha.drivers.aaa_gas import fetch as aaa_fetch
from kalshi_alpha.drivers.aaa_gas import ingest as aaa_ingest
from kalshi_alpha.drivers.nws_cli import fetch_daily_climate_report, fetch_station_metadata


def test_bls_cpi_offline_fetch(offline_fixtures_root: Path) -> None:
    fixtures = offline_fixtures_root / "bls_cpi"
    calendar = bls_cpi.fetch_release_calendar(offline=True, fixtures_dir=fixtures)
    latest = bls_cpi.fetch_latest_release(offline=True, fixtures_dir=fixtures)
    assert calendar and calendar[0].tzinfo is not None
    assert getattr(calendar[0].tzinfo, "key", "") == "America/New_York"
    assert latest.mom_sa == 0.32


def test_dol_claims_offline_fetch(offline_fixtures_root: Path) -> None:
    fixtures = offline_fixtures_root / "dol_claims"
    report = dol_claims.fetch_latest_report(offline=True, fixtures_dir=fixtures)
    assert report.initial_claims_nsa == 220000


def test_treasury_yields_offline_fetch(offline_fixtures_root: Path) -> None:
    fixtures = offline_fixtures_root / "treasury_yields"
    yields = treasury_yields.fetch_daily_yields(offline=True, fixtures_dir=fixtures)
    assert treasury_yields.dgs10_latest_rate(yields) == 4.35


def test_treasury_yields_persists_latest_parquet(
    tmp_path: Path,
    offline_fixtures_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixtures = offline_fixtures_root / "treasury_yields"
    proc_root = tmp_path / "proc" / "treasury_yields"
    monkeypatch.setattr(treasury_yields, "PROC_ROOT", proc_root)

    yields = treasury_yields.fetch_daily_yields(offline=True, fixtures_dir=fixtures)
    frame = treasury_yields.yields_to_frame(yields)
    as_of_dates = sorted({value for value in frame.get_column("as_of").to_list()})
    assert len(as_of_dates) >= 2

    latest_date = as_of_dates[-1]
    prior_date = as_of_dates[-2]

    latest_path = proc_root / "latest.parquet"
    daily_latest = proc_root / "daily" / f"{latest_date.isoformat()}.parquet"
    daily_prior = proc_root / "daily" / f"{prior_date.isoformat()}.parquet"

    assert latest_path.exists()
    assert daily_latest.exists()
    assert daily_prior.exists()

    today_frame = treasury_yields.today_close()
    today_dates = set(today_frame.get_column("as_of").to_list())
    assert today_dates == {latest_date}

    yesterday_frame = treasury_yields.yesterday_close()
    yesterday_dates = set(yesterday_frame.get_column("as_of").to_list())
    assert yesterday_dates == {prior_date}


def test_cleveland_nowcast_offline_fetch(offline_fixtures_root: Path) -> None:
    fixtures = offline_fixtures_root / "cleveland_nowcast"
    series = cleveland_nowcast.fetch_nowcast(offline=True, fixtures_dir=fixtures)
    assert "headline" in series and series["headline"].value == 0.31


def test_nws_cli_offline_fetch(offline_fixtures_root: Path) -> None:
    fixtures = offline_fixtures_root / "nws_cli"
    metadata = fetch_station_metadata(offline=True, fixtures_dir=fixtures)
    assert "KBOS" in metadata
    record = fetch_daily_climate_report("KBOS", offline=True, fixtures_dir=fixtures)
    assert record.high_temp_f == 62


def test_aaa_ingest_and_fetch_offline(tmp_path: Path, offline_fixtures_root: Path) -> None:
    fixtures = offline_fixtures_root / "aaa"
    daily_path = tmp_path / "proc" / "aaa_daily.parquet"
    monthly_path = tmp_path / "proc" / "aaa_monthly.parquet"
    daily_path.parent.mkdir(parents=True, exist_ok=True)

    # Redirect ingest paths to tmp directory
    aaa_ingest.PROC_ROOT = daily_path.parent
    aaa_fetch.PROC_ROOT = daily_path.parent
    aaa_fetch.DAILY_PATH = daily_path
    aaa_fetch.MONTHLY_PATH = monthly_path

    csv_path = fixtures / "AAA_daily_gas_price_regular_sample.csv"
    aaa_ingest.bootstrap_from_csv(csv_path)

    result = aaa_fetch.fetch_latest(offline=True, fixtures_dir=fixtures)
    assert result.price == 3.42
    assert daily_path.exists() and monthly_path.exists()
    frame = pl.read_parquet(daily_path)
    assert "price" in frame.columns
    avg = aaa_fetch.mtd_average(result.as_of_date)
    assert avg is not None


def test_datastore_ingest_offline(offline_fixtures_root: Path) -> None:
    datastore_ingest.main(
        [
            "--offline",
            "--fixtures",
            str(offline_fixtures_root),
            "--source",
            "bls_cpi",
        ]
    )
