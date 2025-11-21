from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


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


@pytest.fixture(autouse=True)
def _mock_external_clients(
    monkeypatch: pytest.MonkeyPatch,
    fixtures_root: Path,
    request: pytest.FixtureRequest,
) -> None:
    """Default offline stubs so tests never hit live APIs unless marked slow."""

    if request.node.get_closest_marker("slow"):
        return

    # Polygon indices client -------------------------------------------------
    from kalshi_alpha.drivers.polygon_index import client as polygon_client

    original_polygon_init = polygon_client.PolygonIndicesClient.__init__

    class _StubResponse:
        def __init__(self, payload: dict[str, Any]) -> None:
            self.status_code = 200
            self._payload = payload
            self.text = "offline-stub"

        def json(self) -> dict[str, Any]:
            return self._payload

    class _StubPolygonSession:
        def request(
            self,
            method: str,
            url: str,
            *,
            params: dict[str, Any] | None = None,
            headers: dict[str, str] | None = None,
            timeout: float | None = None,
        ) -> _StubResponse:
            payload: dict[str, Any] = {"status": "OK", "results": []}
            if "snapshot" in url:
                payload["results"] = {
                    "ticker": "I:SPX",
                    "lastQuote": {"p": 0.0},
                    "prevDay": {"c": 0.0},
                    "lastUpdated": 0,
                }
            return _StubResponse(payload)

    def _offline_polygon_init(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("api_key", "offline-test-key")
        kwargs.setdefault("session", _StubPolygonSession())
        kwargs.setdefault("ws_url", None)
        return original_polygon_init(self, *args, **kwargs)

    monkeypatch.setattr(polygon_client.PolygonIndicesClient, "__init__", _offline_polygon_init)

    # Kalshi public client ---------------------------------------------------
    from kalshi_alpha.core import kalshi_api

    original_kalshi_init = kalshi_api.KalshiPublicClient.__init__
    original_kalshi_get = kalshi_api.KalshiPublicClient._get

    def _offline_kalshi_init(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("offline_dir", fixtures_root / "kalshi")
        kwargs.setdefault("use_offline", True)
        kwargs.setdefault("timeout", 0.2)
        return original_kalshi_init(self, *args, **kwargs)

    def _offline_kalshi_get(
        self,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        cache_key: tuple[Any, ...],
        offline_stub: str | None,
        force_refresh: bool,
    ) -> Any:
        try:
            return original_kalshi_get(
                self,
                endpoint,
                params=params,
                cache_key=cache_key,
                offline_stub=offline_stub,
                force_refresh=force_refresh,
            )
        except FileNotFoundError:
            return {}

    monkeypatch.setattr(kalshi_api.KalshiPublicClient, "__init__", _offline_kalshi_init)
    monkeypatch.setattr(kalshi_api.KalshiPublicClient, "_get", _offline_kalshi_get)


INDEX_ACTIVE_PATTERNS = (
    "test_index_",
    "/test_u_hourly_rotation.py",
    "/test_index_fee_rules.py",
    "/test_scoreboard.py",
    "/test_pilot_runners.py",
    "/test_micro_runner.py",
    "/test_family_switch.py",
    "/test_index_windows_guard.py",
    "/test_fast_index_scans.py",
    "/test_time_awareness.py",
    "/test_index_panel_polygon.py",
    "/test_model_polygon.py",
    "/test_backtest_index_polygon.py",
    "/test_kalshi_index_history.py",
    "/test_fill_model.py",
    "/test_ws_lifecycle.py",
    "/test_polygon_index_ws.py",
    "/test_index_paper_ledger.py",
    "/test_scoreboard_index_paper.py",
    "/test_freshness_gate_index.py",
)
SLOW_PATTERNS = (
    "live_smoke",
    "ws_smoke",
    "ingest_online",
    "polygon_ws",
    "tob_recorder",
    "live_hourly",
    "live_run",
    "market_discovery_online",
)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run network/slow tests",
    )
    parser.addoption(
        "--run-legacy",
        action="store_true",
        default=False,
        help="run the full legacy suite (skipped by default for fast CI)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: marks tests that hit live APIs or take >60s")
    config.addinivalue_line("markers", "legacy: legacy suite skipped by default")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_slow: pytest.MarkDecorator | None = None
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="use --run-slow to execute network/slow tests")

    run_legacy = config.getoption("--run-legacy")
    for item in items:
        nodeid = item.nodeid.lower()
        if any(pattern in nodeid for pattern in SLOW_PATTERNS):
            item.add_marker(pytest.mark.slow)
        if skip_slow and "slow" in item.keywords:
            item.add_marker(skip_slow)
        path = str(item.fspath)
        if any(pattern in path for pattern in INDEX_ACTIVE_PATTERNS):
            continue
        item.add_marker(pytest.mark.legacy)
        if not run_legacy:
            item.add_marker(pytest.mark.skip(reason="use --run-legacy to include legacy suite"))
