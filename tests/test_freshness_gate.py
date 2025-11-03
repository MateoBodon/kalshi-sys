from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
import yaml

from kalshi_alpha.exec.monitors.freshness import (
    FRESHNESS_ARTIFACT_PATH,
    load_artifact,
    summarize_artifact,
    write_freshness_artifact,
)
from kalshi_alpha.exec.reports.ramp import RampPolicyConfig, compute_ramp_policy


def _write_snapshot(directory: Path, frame: pl.DataFrame, now: datetime) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    slug = now.strftime("%Y%m%dT%H%M%S")
    path = directory / f"{slug}.parquet"
    frame.write_parquet(path)


@dataclass(slots=True)
class FeedSnapshotOptions:
    claims_days_old: int = 1
    treasury_has_dgs: bool = True
    aaa_price: float = 3.2
    weather_days_old: int = 0
    weather_stations: Sequence[str] = ("KBOS",)


def _seed_feed_snapshots(
    proc_root: Path,
    now: datetime,
    options: FeedSnapshotOptions | None = None,
) -> None:
    opts = options or FeedSnapshotOptions()
    release_time = now - timedelta(days=7)
    bls_frame = pl.DataFrame(
        {
            "release_datetime": [release_time],
            "period": ["2025-10"],
            "mom_sa": [0.2],
            "yoy_sa": [3.0],
        }
    )
    _write_snapshot(proc_root / "bls_cpi" / "latest_release", bls_frame, now)

    claims_date = (now - timedelta(days=opts.claims_days_old)).date()
    claims_frame = pl.DataFrame(
        {
            "week_ending": [claims_date],
            "initial_claims": [210_000],
        }
    )
    _write_snapshot(proc_root / "dol_claims" / "latest_report", claims_frame, now)

    treasury_date = (now - timedelta(days=1)).date()
    treasury_rows: list[dict[str, object]] = []
    if opts.treasury_has_dgs:
        treasury_rows.append({"as_of": treasury_date, "maturity": "DGS10", "rate": 4.35})
    treasury_rows.append({"as_of": treasury_date, "maturity": "10 YR", "rate": 4.33})
    treasury_frame = pl.DataFrame(treasury_rows)
    _write_snapshot(proc_root / "treasury_yields" / "daily", treasury_frame, now)

    cleveland_frame = pl.DataFrame(
        {
            "series": ["headline"],
            "label": ["Headline"],
            "as_of": [now - timedelta(days=5)],
            "value": [3.1],
        }
    )
    _write_snapshot(proc_root / "cleveland_nowcast" / "monthly", cleveland_frame, now)

    aaa_path = proc_root / "aaa_daily.parquet"
    aaa_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"date": [now.date()], "price": [opts.aaa_price]}).write_parquet(aaa_path)

    if opts.weather_stations:
        weather_date = (now - timedelta(days=opts.weather_days_old)).date()
        weather_records = [
            {
                "station_id": station,
                "record_date": weather_date,
                "high_temp_f": 72,
                "low_temp_f": 58,
            }
            for station in opts.weather_stations
        ]
        weather_frame = pl.DataFrame(weather_records)
        _write_snapshot(proc_root / "nws_cli" / "daily_climate", weather_frame, now)


def _seed_monitor_artifacts(monitors_dir: Path, now: datetime) -> None:
    monitors_dir.mkdir(parents=True, exist_ok=True)
    monitors_dir.joinpath("ev_gap.json").write_text(
        json.dumps(
            {
                "name": "ev_gap",
                "status": "OK",
                "metrics": {"mean_delta_bps": 0.5},
                "generated_at": now.isoformat(),
            }
        ),
        encoding="utf-8",
    )


def _seed_ledger(proc_root: Path, now: datetime) -> Path:
    ledger_path = proc_root / "ledger_all.parquet"
    frame = pl.DataFrame(
        {
            "series": ["CPI"],
            "ev_expected_bps": [10.0],
            "ev_realized_bps": [11.0],
            "expected_fills": [500],
            "timestamp_et": [now - timedelta(hours=1)],
        }
    )
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(ledger_path)
    return ledger_path


def _freshness_config(path: Path, *, active_stations: Iterable[str]) -> Path:
    payload = {
        "feeds": {
            "nws_daily_climate": {"active_stations": list(active_stations)},
        }
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_freshness_stale_feed_blocks_ramp(
    tmp_path: Path,
    isolated_data_roots: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _, proc_root = isolated_data_roots
    now = datetime(2025, 11, 3, 12, 0, tzinfo=UTC)

    _seed_feed_snapshots(
        proc_root,
        now,
        FeedSnapshotOptions(claims_days_old=10, weather_stations=("KBOS",)),
    )
    monitors_dir = tmp_path / "reports" / "_artifacts" / "monitors"
    _seed_monitor_artifacts(monitors_dir, now)
    config_path = _freshness_config(tmp_path / "freshness.yaml", active_stations=("KBOS",))
    freshness_path = monitors_dir / FRESHNESS_ARTIFACT_PATH.name
    write_freshness_artifact(
        config_path=config_path,
        output_path=freshness_path,
        proc_root=proc_root,
        now=now,
    )
    ledger_path = _seed_ledger(proc_root, now)

    policy = compute_ramp_policy(
        ledger_path=ledger_path,
        artifacts_dir=tmp_path / "reports" / "_artifacts",
        monitor_artifacts_dir=monitors_dir,
        drawdown_state_dir=proc_root,
        config=RampPolicyConfig(
            min_fills=0,
            min_delta_bps=-10,
            min_t_stat=-10,
            seq_guard_threshold=100.0,
            seq_guard_min_sample=0,
            ledger_max_age_minutes=120,
            monitor_max_age_minutes=120,
        ),
        now=now,
    )

    assert policy["overall"]["decision"] == "NO_GO"
    assert "STALE_FEEDS" in policy["overall"]["global_reasons"]
    stale_feeds = policy["data_freshness"]["stale_feeds"]
    assert "dol_claims.latest_report" in stale_feeds


def test_freshness_aaa_out_of_range(
    tmp_path: Path,
    isolated_data_roots: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _, proc_root = isolated_data_roots
    now = datetime(2025, 11, 3, 12, 0, tzinfo=UTC)

    _seed_feed_snapshots(
        proc_root,
        now,
        FeedSnapshotOptions(aaa_price=6.7, weather_stations=()),
    )
    monitors_dir = tmp_path / "reports" / "_artifacts" / "monitors"
    monitors_dir.mkdir(parents=True, exist_ok=True)
    output_path = monitors_dir / FRESHNESS_ARTIFACT_PATH.name
    write_freshness_artifact(
        output_path=output_path,
        proc_root=proc_root,
        now=now,
    )
    artifact = load_artifact(output_path)
    summary = summarize_artifact(artifact, artifact_path=output_path)
    aaa_entry = next(
        entry
        for entry in summary["feeds"]
        if entry.get("id") == "aaa_gas.daily"
    )
    assert aaa_entry["ok"] is False
    assert aaa_entry.get("reason") == "AAA_OUT_OF_RANGE"


def test_freshness_teny_series_mismatch(
    tmp_path: Path,
    isolated_data_roots: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _, proc_root = isolated_data_roots
    now = datetime(2025, 11, 3, 12, 0, tzinfo=UTC)

    _seed_feed_snapshots(
        proc_root,
        now,
        FeedSnapshotOptions(treasury_has_dgs=False, weather_stations=()),
    )
    monitors_dir = tmp_path / "reports" / "_artifacts" / "monitors"
    monitors_dir.mkdir(parents=True, exist_ok=True)
    output_path = monitors_dir / FRESHNESS_ARTIFACT_PATH.name
    write_freshness_artifact(
        output_path=output_path,
        proc_root=proc_root,
        now=now,
    )
    artifact = load_artifact(output_path)
    summary = summarize_artifact(artifact, artifact_path=output_path)
    teny_entry = next(
        entry
        for entry in summary["feeds"]
        if entry.get("id") == "treasury_10y.daily"
    )
    assert teny_entry["ok"] is False
    assert teny_entry.get("reason") == "TENY_SERIES_MISMATCH"


def test_freshness_weather_gate_respects_active_stations(
    tmp_path: Path,
    isolated_data_roots: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _, proc_root = isolated_data_roots
    now = datetime(2025, 11, 3, 12, 0, tzinfo=UTC)

    # No active stations → should not block GO.
    _seed_feed_snapshots(
        proc_root,
        now,
        FeedSnapshotOptions(weather_stations=()),
    )
    monitors_dir = tmp_path / "reports" / "_artifacts" / "monitors"
    _seed_monitor_artifacts(monitors_dir, now)
    inactive_config = _freshness_config(tmp_path / "freshness_inactive.yaml", active_stations=())
    write_freshness_artifact(
        config_path=inactive_config,
        output_path=monitors_dir / FRESHNESS_ARTIFACT_PATH.name,
        proc_root=proc_root,
        now=now,
    )
    ledger_path = _seed_ledger(proc_root, now)
    policy = compute_ramp_policy(
        ledger_path=ledger_path,
        artifacts_dir=tmp_path / "reports" / "_artifacts",
        monitor_artifacts_dir=monitors_dir,
        drawdown_state_dir=proc_root,
        config=RampPolicyConfig(
            min_fills=0,
            min_delta_bps=-10,
            min_t_stat=-10,
            seq_guard_threshold=100.0,
            seq_guard_min_sample=0,
            ledger_max_age_minutes=120,
            monitor_max_age_minutes=120,
        ),
        now=now,
    )
    assert policy["overall"]["required_feeds_ok"] is True

    # Activate KBOS but make it stale → force NO-GO.
    (monitors_dir / FRESHNESS_ARTIFACT_PATH.name).unlink()
    _seed_feed_snapshots(
        proc_root,
        now,
        FeedSnapshotOptions(weather_days_old=5, weather_stations=("KBOS",)),
    )
    active_config = _freshness_config(tmp_path / "freshness_active.yaml", active_stations=("KBOS",))
    write_freshness_artifact(
        config_path=active_config,
        output_path=monitors_dir / FRESHNESS_ARTIFACT_PATH.name,
        proc_root=proc_root,
        now=now,
    )
    policy = compute_ramp_policy(
        ledger_path=ledger_path,
        artifacts_dir=tmp_path / "reports" / "_artifacts",
        monitor_artifacts_dir=monitors_dir,
        drawdown_state_dir=proc_root,
        config=RampPolicyConfig(
            min_fills=0,
            min_delta_bps=-10,
            min_t_stat=-10,
            seq_guard_threshold=100.0,
            seq_guard_min_sample=0,
            ledger_max_age_minutes=120,
            monitor_max_age_minutes=120,
        ),
        now=now,
    )
    assert policy["overall"]["decision"] == "NO_GO"
    reason = policy["data_freshness"]["feeds"][-1].get("reason")
    assert reason and "KBOS" in reason
