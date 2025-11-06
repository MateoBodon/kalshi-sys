from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import polars as pl

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


def _write_polygon_snapshot(proc_root: Path, snapshot_ts: datetime) -> None:
    target_dir = proc_root / "polygon_index"
    target_dir.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame({"snapshot_ts": [snapshot_ts]})
    frame.write_parquet(target_dir / "snapshot.parquet")


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
    assert first.maker_ev == pytest.approx(0.5613895748, rel=1e-6)
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
    _write_polygon_snapshot(proc_root, datetime.now(tz=UTC))

    fixtures_path = fixtures_root.resolve()
    pal_policy_path = Path("configs/pal_policy.example.yaml").resolve()
    gate_config_path = Path("configs/quality_gates.index.yaml").resolve()

    artifacts_dir = tmp_path / "reports" / "_artifacts"
    monitors_dir = artifacts_dir / "monitors"
    monitors_dir.mkdir(parents=True, exist_ok=True)
    freshness_payload = {
        "name": "data_freshness",
        "status": "ALERT",
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "metrics": {
            "required_feeds_ok": False,
            "required_feeds": ["polygon_index.websocket"],
            "stale_feeds": ["polygon_index.websocket"],
            "feeds": [
                {
                    "id": "polygon_index.websocket",
                    "label": "Polygon index websocket",
                    "required": True,
                    "ok": False,
                    "age_minutes": 5.0,
                    "reason": "STALE>2s",
                    "details": {"threshold_seconds": 2.0},
                }
            ],
        },
    }
    (monitors_dir / "freshness.json").write_text(json.dumps(freshness_payload, indent=2), encoding="utf-8")

    configs_dir = tmp_path / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    legacy_config = "\n".join(
        [
            "data_freshness:",
            "  - name: macro_feed",
            "    namespace: macro/latest",
            "    timestamp_field: as_of",
            "    max_age_minutes: 10",
            "    require_et: true",
            "monitors:",
            "  tz_not_et: 0",
            "  non_monotone_ladders: 0",
        ]
    )
    (configs_dir / "quality_gates.yaml").write_text(legacy_config + "\n", encoding="utf-8")

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
            "--report",
            "--quality-gates-config",
            str(gate_config_path),
            "--pal-policy",
            str(pal_policy_path),
        ]
    )

    go_artifact = json.loads((artifacts_dir / "go_no_go.json").read_text(encoding="utf-8"))
    assert go_artifact["go"] is False
    reasons = go_artifact.get("reasons", [])
    assert "STALE_FEEDS" in reasons
    assert "polygon_ws_stale" in reasons

    proposals_dir = tmp_path / "exec" / "proposals" / "INXU"
    proposal_files = sorted(proposals_dir.glob("*.json"))
    assert proposal_files, "expected proposals artifact"
    proposals_payload = json.loads(proposal_files[-1].read_text(encoding="utf-8"))
    assert proposals_payload.get("proposals") == []

    orders_path = proc_root / "state" / "orders.json"
    payload = json.loads(orders_path.read_text(encoding="utf-8"))
    cancel_entry = payload.get("cancel_all")
    assert cancel_entry is not None
    assert cancel_entry.get("reason") == "polygon_ws_stale"

    report_dir = tmp_path / "reports" / "INXU"
    report_files = sorted(report_dir.glob("*.md"))
    assert report_files, "expected markdown report for INXU"
    report_text = report_files[-1].read_text(encoding="utf-8")
    assert "polygon_ws_freshness" in report_text
    assert "ops_cancel_at" in report_text
    assert "fee_path" in report_text


def test_macro_stale_allows_execution_with_index_gates(
    fixtures_root: Path,
    isolated_data_roots: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, proc_root = isolated_data_roots
    _configure_index_calibration(proc_root)
    _copy_calibration(Path("tests/fixtures/index/spx/hourly/params.json"), proc_root, "spx", "hourly")
    _write_polygon_snapshot(proc_root, datetime.now(tz=UTC))

    fixtures_path = fixtures_root.resolve()
    pal_policy_path = Path("configs/pal_policy.example.yaml").resolve()
    gate_config_path = Path("configs/quality_gates.index.yaml").resolve()

    artifacts_dir = tmp_path / "reports" / "_artifacts"
    monitors_dir = artifacts_dir / "monitors"
    monitors_dir.mkdir(parents=True, exist_ok=True)
    freshness_payload = {
        "name": "data_freshness",
        "status": "ALERT",
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "metrics": {
            "required_feeds_ok": True,
            "required_feeds": ["polygon_index.websocket"],
            "stale_feeds": ["macro_calendar.latest"],
            "feeds": [
                    {
                        "id": "polygon_index.websocket",
                        "label": "Polygon index websocket",
                        "required": True,
                        "ok": True,
                        "age_minutes": 0.0005,
                        "reason": None,
                        "details": {"threshold_seconds": 2.0},
                    },
                {
                    "id": "macro_calendar.latest",
                    "label": "Macro calendar",
                    "required": False,
                    "ok": False,
                    "age_minutes": 45.0,
                    "reason": "STALE>10m",
                },
            ],
        },
    }
    (monitors_dir / "freshness.json").write_text(json.dumps(freshness_payload, indent=2), encoding="utf-8")

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
            "--report",
            "--quality-gates-config",
            str(gate_config_path),
            "--pal-policy",
            str(pal_policy_path),
        ]
    )

    go_artifact = json.loads((artifacts_dir / "go_no_go.json").read_text(encoding="utf-8"))
    assert go_artifact["go"] is True
    reasons = go_artifact.get("reasons", [])
    assert "STALE_FEEDS" not in reasons
    assert "polygon_ws_stale" not in reasons

    proposals_dir = tmp_path / "exec" / "proposals" / "INXU"
    proposal_files = sorted(proposals_dir.glob("*.json"))
    assert proposal_files, "expected proposals artifact"
    proposals_payload = json.loads(proposal_files[-1].read_text(encoding="utf-8"))
    proposals = proposals_payload.get("proposals") or []
    assert proposals, "expected proposals when macro feeds are stale but polygon OK"
    first = proposals[0]
    metadata = first.get("metadata") or {}
    components = metadata.get("ev_components")
    assert components is not None
    per_contract = components.get("per_contract") or {}
    total = components.get("total") or {}
    assert per_contract, "expected per-contract EV components"
    assert total, "expected total EV components"
    assert per_contract.get("gross") == pytest.approx(
        per_contract.get("net", 0.0) + per_contract.get("fee", 0.0),
        rel=1e-9,
        abs=1e-9,
    )
    assert total.get("gross") == pytest.approx(
        total.get("net", 0.0) + total.get("fee", 0.0),
        rel=1e-9,
        abs=1e-9,
    )
    assert components.get("liquidity") == "maker"

    report_dir = tmp_path / "reports" / "INXU"
    report_files = sorted(report_dir.glob("*.md"))
    assert report_files, "expected markdown report for INXU"
    report_text = report_files[-1].read_text(encoding="utf-8")
    assert "ops_cancel_at" in report_text
    assert "polygon_ws_freshness" in report_text
    assert "fee_path" in report_text


def test_clock_skew_blocks_execution(
    fixtures_root: Path,
    isolated_data_roots: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, proc_root = isolated_data_roots
    _configure_index_calibration(proc_root)
    _copy_calibration(Path("tests/fixtures/index/spx/hourly/params.json"), proc_root, "spx", "hourly")
    _write_polygon_snapshot(proc_root, datetime.now(tz=UTC))

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
    monitors_dir = artifacts_dir / "monitors"
    monitors_dir.mkdir(parents=True, exist_ok=True)
    freshness_payload = {
        "name": "data_freshness",
        "status": "OK",
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "metrics": {
            "required_feeds_ok": True,
            "required_feeds": ["polygon_index.websocket"],
            "stale_feeds": [],
            "feeds": [
                {
                    "id": "polygon_index.websocket",
                    "label": "Polygon index websocket",
                    "required": True,
                    "ok": True,
                    "age_minutes": 0.0005,
                    "reason": None,
                    "details": {"threshold_seconds": 2.0},
                }
            ],
        },
    }
    (monitors_dir / "freshness.json").write_text(json.dumps(freshness_payload, indent=2), encoding="utf-8")

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
    _write_polygon_snapshot(proc_root, datetime.now(tz=UTC))

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
    monitors_dir = artifacts_dir / "monitors"
    monitors_dir.mkdir(parents=True, exist_ok=True)
    freshness_payload = {
        "name": "data_freshness",
        "status": "OK",
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "metrics": {
            "required_feeds_ok": True,
            "required_feeds": ["polygon_index.websocket"],
            "stale_feeds": [],
            "feeds": [
                {
                    "id": "polygon_index.websocket",
                    "label": "Polygon index websocket",
                    "required": True,
                    "ok": True,
                    "age_minutes": 0.0004,
                    "reason": None,
                    "details": {"threshold_seconds": 2.0},
                }
            ],
        },
    }
    (monitors_dir / "freshness.json").write_text(json.dumps(freshness_payload, indent=2), encoding="utf-8")

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
