from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from kalshi_alpha.core.kalshi_api import KalshiPublicClient
from kalshi_alpha.core.risk import PALGuard, PALPolicy
from kalshi_alpha.exec.runners import scan_ladders
from kalshi_alpha.exec.scanners import scan_index_close, scan_index_hourly
from kalshi_alpha.strategies import index as index_strategy
from kalshi_alpha.strategies.index import cdf as index_cdf
from kalshi_alpha.strategies.index import close_range, hourly_above_below

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def _configure_index_calibration(proc_root: Path) -> None:
    base = proc_root / "calib" / "index"
    index_strategy.HOURLY_CALIBRATION_PATH = base
    index_strategy.NOON_CALIBRATION_PATH = base
    index_strategy.CLOSE_CALIBRATION_PATH = base
    scan_index_hourly.HOURLY_CALIBRATION_PATH = base
    scan_index_close.CLOSE_CALIBRATION_PATH = base
    hourly_above_below.HOURLY_CALIBRATION_PATH = base
    close_range.CLOSE_CALIBRATION_PATH = base
    index_cdf._load_calibration_cached.cache_clear()


def _copy_calibration(src: Path, proc_root: Path, symbol: str, horizon: str) -> None:
    dest = proc_root / "calib" / "index" / symbol / horizon / "params.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(src.read_bytes())


def test_scan_series_over_index_fixture_produces_deterministic_ev(
    fixtures_root: Path,
    isolated_data_roots: tuple[Path, Path],
) -> None:
    _, proc_root = isolated_data_roots
    _configure_index_calibration(proc_root)
    _copy_calibration(Path("tests/fixtures/index/spx/hourly/params.json"), proc_root, "spx", "hourly")

    client = KalshiPublicClient(offline_dir=fixtures_root / "kalshi", use_offline=True)
    pal_guard = PALGuard(PALPolicy(series="INXU", default_max_loss=10_000.0))
    now_et = datetime(2025, 11, 3, 11, 55, tzinfo=ET)
    outcome = scan_ladders.scan_series(
        series="INXU",
        client=client,
        min_ev=0.0,
        contracts=1,
        pal_guard=pal_guard,
        driver_fixtures=fixtures_root / "drivers",
        strategy_name="auto",
        maker_only=True,
        allow_tails=False,
        risk_manager=None,
        max_var=None,
        offline=True,
        sizing_mode="kelly",
        kelly_cap=0.25,
        now_override=now_et.astimezone(UTC),
    )
    assert len(outcome.proposals) == 2
    first = outcome.proposals[0]
    assert first.maker_ev == pytest.approx(0.55139, rel=1e-6)
    assert outcome.roll_info is not None
    assert outcome.roll_info["target_hour_label"] == "H1200"
    assert outcome.execution_metrics is None
    assert outcome.monitors.get("index_rules_ok") is True


def test_stale_freshness_blocks_execution(
    fixtures_root: Path,
    isolated_data_roots: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, proc_root = isolated_data_roots
    _configure_index_calibration(proc_root)
    _copy_calibration(Path("tests/fixtures/index/spx/hourly/params.json"), proc_root, "spx", "hourly")

    fixtures_path = fixtures_root.resolve()
    pal_policy_path = Path("configs/pal_policy.example.yaml").resolve()
    quality_example = Path("configs/quality_gates.example.yaml").resolve()
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    gate_config_path = configs_dir / "quality_gates.example.yaml"
    gate_config_path.write_text(
        quality_example.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    artifacts_dir = tmp_path / "reports" / "_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    freshness_payload = {
        "required_feeds_ok": False,
        "feeds": [
            {
                "id": "polygon_index.websocket",
                "label": "Polygon index websocket",
                "required": True,
                "ok": False,
                "age_minutes": 5.0,
                "reason": "STALE>2s",
            }
        ],
    }
    (artifacts_dir / "freshness.json").write_text(json.dumps(freshness_payload, indent=2), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    scan_ladders.main(
        [
            "--series",
            "INXU",
            "--fixtures-root",
            str(fixtures_path),
            "--offline",
            "--min-ev",
            "0.0",
            "--contracts",
            "1",
            "--maker-only",
            "--pal-policy",
            str(pal_policy_path),
        ]
    )

    go_artifact = json.loads((artifacts_dir / "go_no_go.json").read_text(encoding="utf-8"))
    assert go_artifact["go"] is False

    orders_path = proc_root / "state" / "orders.json"
    payload = json.loads(orders_path.read_text(encoding="utf-8"))
    cancel_entry = payload.get("cancel_all")
    assert cancel_entry is not None
    assert cancel_entry.get("reason") == "quality_gate_no_go"


def test_clock_skew_blocks_execution(
    fixtures_root: Path,
    isolated_data_roots: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, proc_root = isolated_data_roots
    _configure_index_calibration(proc_root)
    _copy_calibration(Path("tests/fixtures/index/spx/hourly/params.json"), proc_root, "spx", "hourly")

    fixtures_path = fixtures_root.resolve()
    pal_policy_path = Path("configs/pal_policy.example.yaml").resolve()
    quality_example = Path("configs/quality_gates.example.yaml").resolve()
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    gate_config_path = configs_dir / "quality_gates.example.yaml"
    gate_config_path.write_text(
        quality_example.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    artifacts_dir = tmp_path / "reports" / "_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    freshness_payload = {
        "required_feeds_ok": True,
        "feeds": [],
    }
    (artifacts_dir / "freshness.json").write_text(json.dumps(freshness_payload, indent=2), encoding="utf-8")

    monkeypatch.setattr(scan_ladders, "_clock_skew_seconds", lambda *_: 5.0)
    monkeypatch.chdir(tmp_path)

    scan_ladders.main(
        [
            "--series",
            "INXU",
            "--fixtures-root",
            str(fixtures_path),
            "--offline",
            "--min-ev",
            "0.0",
            "--contracts",
            "1",
            "--maker-only",
            "--pal-policy",
            str(pal_policy_path),
        ]
    )

    go_artifact = json.loads((artifacts_dir / "go_no_go.json").read_text(encoding="utf-8"))
    assert go_artifact["go"] is False
    assert "clock_skew_exceeded" in go_artifact.get("reasons", [])


def test_index_rule_mismatch_forces_no_go(
    fixtures_root: Path,
    isolated_data_roots: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, proc_root = isolated_data_roots
    _configure_index_calibration(proc_root)
    _copy_calibration(Path("tests/fixtures/index/spx/hourly/params.json"), proc_root, "spx", "hourly")

    fixtures_path = fixtures_root.resolve()
    pal_policy_path = Path("configs/pal_policy.example.yaml").resolve()
    quality_example = Path("configs/quality_gates.example.yaml").resolve()
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    gate_config_path = configs_dir / "quality_gates.example.yaml"
    gate_config_path.write_text(
        quality_example.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    artifacts_dir = tmp_path / "reports" / "_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    freshness_payload = {
        "required_feeds_ok": True,
        "feeds": [],
    }
    (artifacts_dir / "freshness.json").write_text(json.dumps(freshness_payload, indent=2), encoding="utf-8")

    from kalshi_alpha.config import IndexRule

    monkeypatch.setattr(
        scan_ladders,
        "lookup_index_rule",
        lambda series: IndexRule(
            series=series.upper(),
            display_name=series.upper(),
            evaluation_time_et="13:00:00 ET",
            evaluation_clause="incorrect",
            timing_clause="",
            fallback_clause="settle after print",
            reference_source="S&P",
            primary_window_et="",
            tick_size_usd=0.01,
            position_limit_usd=7000000,
        ),
    )

    monkeypatch.chdir(tmp_path)
    scan_ladders.main(
        [
            "--series",
            "INXU",
            "--fixtures-root",
            str(fixtures_path),
            "--offline",
            "--min-ev",
            "0.0",
            "--contracts",
            "1",
            "--maker-only",
            "--pal-policy",
            str(pal_policy_path),
        ]
    )

    go_artifact = json.loads((artifacts_dir / "go_no_go.json").read_text(encoding="utf-8"))
    assert go_artifact["go"] is False
    assert "index_rules_mismatch" in go_artifact.get("reasons", [])


def test_close_scan_includes_event_tags(
    fixtures_root: Path,
    isolated_data_roots: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, proc_root = isolated_data_roots
    _configure_index_calibration(proc_root)
    _copy_calibration(Path("tests/fixtures/index/ndx/close/params.json"), proc_root, "ndx", "close")

    def _fake_calendar(moment: datetime, *, path: Path | None = None) -> tuple[str, ...]:
        target_date = moment.astimezone(ET).date()
        return ("FOMC",) if target_date == date(2024, 10, 21) else ()

    monkeypatch.setattr(scan_ladders, "calendar_tags_for", _fake_calendar)

    client = KalshiPublicClient(offline_dir=fixtures_root / "kalshi", use_offline=True)
    pal_guard = PALGuard(PALPolicy(series="NASDAQ100", default_max_loss=10_000.0))
    now_et = datetime(2024, 10, 21, 15, 55, tzinfo=ET)
    outcome = scan_ladders.scan_series(
        series="NASDAQ100",
        client=client,
        min_ev=0.0,
        contracts=1,
        pal_guard=pal_guard,
        driver_fixtures=fixtures_root / "drivers",
        strategy_name="auto",
        maker_only=True,
        allow_tails=False,
        risk_manager=None,
        max_var=None,
        offline=True,
        sizing_mode="kelly",
        kelly_cap=0.25,
        now_override=now_et.astimezone(UTC),
    )
    assert outcome.model_metadata.get("event_tags") == ("FOMC",)
