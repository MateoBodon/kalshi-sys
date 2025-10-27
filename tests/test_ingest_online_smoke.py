from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from kalshi_alpha.datastore import ingest as datastore_ingest
from kalshi_alpha.drivers.aaa_gas import fetch as aaa_fetch


def _latest_parquet(directory: Path) -> Path:
    files = sorted(directory.glob("*.parquet"))
    assert files, f"No parquet files found in {directory}"
    return files[-1]


def test_ingest_offline_smoke(offline_fixtures_root: Path, isolated_data_roots: tuple[Path, Path]) -> None:
    raw_root, proc_root = isolated_data_roots
    datastore_ingest.main(
        [
            "--all",
            "--offline",
            "--fixtures",
            str(offline_fixtures_root),
            "--force-refresh",
        ]
    )

    run_date = datetime.now(tz=UTC).date()
    raw_day_dir = raw_root / f"{run_date.year:04d}" / f"{run_date.month:02d}" / f"{run_date.day:02d}"
    assert raw_day_dir.exists()
    for namespace in ("bls_cpi", "dol_claims", "treasury_yields", "cleveland_nowcast", "nws_cli", "aaa"):
        ns_dir = raw_day_dir / namespace
        assert ns_dir.exists(), f"Missing raw snapshot dir for {namespace}"
        assert any(ns_dir.iterdir()), f"Raw snapshot dir {ns_dir} is empty"

    calendar_frame = pl.read_parquet(_latest_parquet(proc_root / "bls_cpi" / "calendar"))
    assert set(calendar_frame.columns) == {"release_datetime", "release_date"}
    assert calendar_frame.height >= 1

    release_frame = pl.read_parquet(_latest_parquet(proc_root / "bls_cpi" / "latest_release"))
    assert {"release_datetime", "period", "mom_sa", "yoy_sa"} <= set(release_frame.columns)
    assert release_frame.height == 1

    claims_frame = pl.read_parquet(_latest_parquet(proc_root / "dol_claims" / "latest_report"))
    assert {"week_ending", "initial_claims_nsa", "initial_claims_sa"} <= set(claims_frame.columns)

    yields_frame = pl.read_parquet(_latest_parquet(proc_root / "treasury_yields" / "daily"))
    assert {"as_of", "maturity", "rate"} <= set(yields_frame.columns)
    assert yields_frame.height > 0

    nowcast_frame = pl.read_parquet(_latest_parquet(proc_root / "cleveland_nowcast" / "monthly"))
    assert {"series", "label", "as_of", "value"} <= set(nowcast_frame.columns)
    assert nowcast_frame.height >= 2

    station_frame = pl.read_parquet(_latest_parquet(proc_root / "nws_cli" / "stations"))
    assert {"station_id", "name", "wban"} <= set(station_frame.columns)
    assert station_frame.height > 0

    climate_frame = pl.read_parquet(_latest_parquet(proc_root / "nws_cli" / "daily_climate"))
    assert {"station_id", "record_date", "high_temp_f", "low_temp_f"} <= set(climate_frame.columns)
    assert climate_frame.height > 0

    assert aaa_fetch.DAILY_PATH.exists()
    aaa_daily = pl.read_parquet(aaa_fetch.DAILY_PATH)
    assert {"date", "price"} <= set(aaa_daily.columns)

    assert aaa_fetch.MONTHLY_PATH.exists()
    aaa_monthly = pl.read_parquet(aaa_fetch.MONTHLY_PATH)
    assert {"month", "avg_price"} <= set(aaa_monthly.columns)
