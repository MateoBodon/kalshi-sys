from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def fixtures_root() -> Path:
    return Path(__file__).parent / "data_fixtures"


@pytest.fixture
def offline_fixtures_root() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def pal_policy_path(tmp_path: Path) -> Path:
    policy_src = Path("configs/pal_policy.example.yaml")
    policy_copy = tmp_path / "pal_policy.yaml"
    policy_copy.write_text(policy_src.read_text(encoding="utf-8"), encoding="utf-8")
    return policy_copy


@pytest.fixture
def isolated_data_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Patch datastore roots so ingestion writes into a temporary sandbox."""
    raw_root = tmp_path / "data" / "raw"
    proc_root = tmp_path / "data" / "proc"
    bootstrap_root = tmp_path / "data" / "bootstrap"
    for path in (raw_root, proc_root, bootstrap_root):
        path.mkdir(parents=True, exist_ok=True)

    from kalshi_alpha.core import kalshi_ws
    from kalshi_alpha.core.archive import archiver
    from kalshi_alpha.core.risk import drawdown as drawdown_module
    from kalshi_alpha.datastore import ingest as datastore_ingest
    from kalshi_alpha.datastore import paths as datastore_paths
    from kalshi_alpha.datastore import snapshots
    from kalshi_alpha.drivers import bls_cpi, cleveland_nowcast, dol_claims, macro_calendar, nws_cli, treasury_yields
    from kalshi_alpha.drivers.aaa_gas import fetch as aaa_fetch
    from kalshi_alpha.drivers.aaa_gas import ingest as aaa_ingest
    from kalshi_alpha.exec.pipelines import calendar as pipeline_calendar
    from kalshi_alpha.exec.runners import scan_ladders
    from kalshi_alpha.exec.scanners import scan_index_close, scan_index_hourly
    from kalshi_alpha.strategies import claims as claims_strategy
    from kalshi_alpha.strategies import cpi as cpi_strategy
    from kalshi_alpha.strategies import index as index_strategy
    from kalshi_alpha.strategies import teny as teny_strategy
    from kalshi_alpha.strategies import weather as weather_strategy
    from kalshi_alpha.strategies.index import close_range as index_close_strategy
    from kalshi_alpha.strategies.index import hourly_above_below as index_hourly_strategy

    monkeypatch.setattr(datastore_paths, "RAW_ROOT", raw_root)
    monkeypatch.setattr(datastore_paths, "PROC_ROOT", proc_root)
    monkeypatch.setattr(datastore_paths, "BOOTSTRAP_ROOT", bootstrap_root)
    monkeypatch.setattr(snapshots, "RAW_ROOT", raw_root)
    monkeypatch.setattr(datastore_ingest, "PROC_ROOT", proc_root)

    bls_cache = raw_root / "_cache" / "bls_cpi"
    bls_cache.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(bls_cpi, "CACHE_ROOT", bls_cache)

    dol_cache = raw_root / "_cache" / "dol_claims" / "eta539.csv"
    dol_cache.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(dol_claims, "CACHE_PATH", dol_cache)

    treasury_cache = raw_root / "_cache" / "treasury" / "daily_yields.csv"
    treasury_cache.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(treasury_yields, "CACHE_PATH", treasury_cache)

    cle_cache = raw_root / "_cache" / "cleveland_nowcast"
    cle_cache.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cleveland_nowcast, "CACHE_DIR", cle_cache)
    monkeypatch.setattr(cleveland_nowcast, "MONTHLY_CACHE_PATH", cle_cache / "nowcast_month.json")
    monkeypatch.setattr(cleveland_nowcast, "PAGE_CACHE_PATH", cle_cache / "nowcast.html")

    nws_cache = raw_root / "_cache" / "nws_cli"
    nws_cache.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(nws_cli, "RAW_CACHE", nws_cache)

    aaa_daily = proc_root / "aaa_daily.parquet"
    aaa_monthly = proc_root / "aaa_monthly.parquet"
    monkeypatch.setattr(aaa_fetch, "PROC_ROOT", proc_root)
    monkeypatch.setattr(aaa_fetch, "DAILY_PATH", aaa_daily)
    monkeypatch.setattr(aaa_fetch, "MONTHLY_PATH", aaa_monthly)
    monkeypatch.setattr(aaa_ingest, "PROC_ROOT", proc_root)
    monkeypatch.setattr(aaa_ingest, "BOOTSTRAP_ROOT", bootstrap_root)

    macro_root = proc_root / "macro_calendar"
    macro_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(macro_calendar, "PROC_ROOT", macro_root)
    monkeypatch.setattr(macro_calendar, "DEFAULT_OUTPUT", macro_root / "latest.parquet")

    monkeypatch.setattr(cpi_strategy, "CALIBRATION_PATH", proc_root / "cpi_calib.parquet")
    monkeypatch.setattr(claims_strategy, "CALIBRATION_PATH", proc_root / "claims_calib.parquet")
    monkeypatch.setattr(teny_strategy, "CALIBRATION_PATH", proc_root / "teny_calib.parquet")
    monkeypatch.setattr(weather_strategy, "CALIBRATION_PATH", proc_root / "weather_calib.parquet")
    index_calib_root = proc_root / "calib" / "index"
    index_calib_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(index_strategy, "HOURLY_CALIBRATION_PATH", index_calib_root)
    monkeypatch.setattr(index_strategy, "NOON_CALIBRATION_PATH", index_calib_root)
    monkeypatch.setattr(index_strategy, "CLOSE_CALIBRATION_PATH", index_calib_root)
    monkeypatch.setattr(index_hourly_strategy, "HOURLY_CALIBRATION_PATH", index_calib_root)
    monkeypatch.setattr(index_hourly_strategy, "NOON_CALIBRATION_PATH", index_calib_root)
    monkeypatch.setattr(index_close_strategy, "CLOSE_CALIBRATION_PATH", index_calib_root)
    monkeypatch.setattr(scan_index_hourly, "HOURLY_CALIBRATION_PATH", index_calib_root)
    monkeypatch.setattr(scan_index_close, "CLOSE_CALIBRATION_PATH", index_calib_root)
    monkeypatch.setattr(scan_ladders, "PROC_ROOT", proc_root)
    monkeypatch.setattr(scan_ladders, "RAW_ROOT", raw_root)
    monkeypatch.setattr(archiver, "RAW_ROOT", raw_root)
    monkeypatch.setattr(pipeline_calendar, "PROC_ROOT", proc_root)
    monkeypatch.setattr(drawdown_module, "PROC_ROOT", proc_root)

    raw_orderbook_root = raw_root / "kalshi" / "orderbook"
    proc_imbalance_root = proc_root / "kalshi" / "orderbook_imbalance"
    raw_orderbook_root.mkdir(parents=True, exist_ok=True)
    proc_imbalance_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(kalshi_ws, "RAW_ORDERBOOK_ROOT", raw_orderbook_root)
    monkeypatch.setattr(kalshi_ws, "PROC_IMBALANCE_ROOT", proc_imbalance_root)

    return raw_root, proc_root


INDEX_ACTIVE_PATTERNS = (
    "test_index_",
    "/test_u_hourly_rotation.py",
    "/test_index_fee_rules.py",
    "/test_scoreboard.py",
)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        path = str(item.fspath)
        if any(pattern in path for pattern in INDEX_ACTIVE_PATTERNS):
            continue
        item.add_marker(pytest.mark.legacy)
