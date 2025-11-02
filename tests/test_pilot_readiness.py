from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from kalshi_alpha.core.risk import drawdown
from kalshi_alpha.exec.reports.ramp import RampPolicyConfig, compute_ramp_policy, write_ramp_outputs
from kalshi_alpha.exec.runners import scan_ladders


def test_pilot_ramp_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    now = datetime(2025, 11, 2, tzinfo=UTC)

    ledger_path = tmp_path / "data" / "proc" / "ledger_all.parquet"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = pl.DataFrame(
        {
            "series": ["CPI"] * 6 + ["CLAIMS"] * 6,
            "ev_expected_bps": [10.0] * 6 + [8.0] * 6,
            "ev_realized_bps": [18.0, 17.5, 18.6, 18.1, 17.9, 18.4, 6.0, 6.2, 5.8, 6.1, 5.9, 6.0],
            "expected_fills": [60, 55, 50, 52, 58, 62, 20, 18, 22, 19, 21, 23],
            "timestamp_et": [now - timedelta(days=idx) for idx in range(1, 7)]
            + [now - timedelta(days=idx) for idx in range(1, 7)],
        }
    )
    ledger.write_parquet(ledger_path)

    artifacts_dir = tmp_path / "reports" / "_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.joinpath("go_no_go_1.json").write_text(
        json.dumps(
            {
                "go": False,
                "series": "CLAIMS",
                "timestamp": (now - timedelta(days=2)).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    monitors_dir = tmp_path / "reports" / "_artifacts" / "monitors"
    monitors_dir.mkdir(parents=True, exist_ok=True)
    monitors_dir.joinpath("ev_gap.json").write_text(
        json.dumps(
            {
                "name": "ev_gap",
                "status": "OK",
                "metrics": {"mean_delta_bps": 1.0},
                "generated_at": now.isoformat(),
            }
        ),
        encoding="utf-8",
    )

    proc_root = tmp_path / "data" / "proc"
    drawdown_state_dir = proc_root
    drawdown.record_pnl(500.0, timestamp=now, state_dir=drawdown_state_dir)

    config = RampPolicyConfig(
        lookback_days=30,
        min_fills=200,
        min_delta_bps=6.0,
        min_t_stat=1.5,
        go_multiplier=1.5,
        base_multiplier=1.0,
        seq_guard_threshold=100.0,
    )

    policy = compute_ramp_policy(
        ledger_path=ledger_path,
        artifacts_dir=artifacts_dir,
        monitor_artifacts_dir=monitors_dir,
        drawdown_state_dir=drawdown_state_dir,
        config=config,
        now=now,
    )

    assert policy["drawdown"]["ok"] is True
    series = {entry["series"]: entry for entry in policy["series"]}
    assert series["CPI"]["recommendation"] == "GO"
    assert series["CPI"]["size_multiplier"] == config.go_multiplier
    assert series["CLAIMS"]["recommendation"] == "NO_GO"
    assert "guardrail_breaches" in series["CLAIMS"]
    assert series["CLAIMS"]["size_multiplier"] == config.base_multiplier
    assert policy["overall"]["global_reasons"] == []

    json_path = tmp_path / "reports" / "pilot_ready.json"
    markdown_path = tmp_path / "reports" / "pilot_readiness.md"
    write_ramp_outputs(policy, json_path=json_path, markdown_path=markdown_path)
    assert json.loads(json_path.read_text(encoding="utf-8"))["series"]
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Pilot Ramp Readiness" in markdown
    assert "| CPI" in markdown


def test_ramp_policy_stale_ledger_no_go(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    now = datetime(2025, 11, 2, 15, 0, tzinfo=UTC)

    ledger_path = tmp_path / "data" / "proc" / "ledger_all.parquet"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = pl.DataFrame(
        {
            "series": ["CPI", "CLAIMS"],
            "ev_expected_bps": [10.0, 8.0],
            "ev_realized_bps": [11.0, 8.5],
            "expected_fills": [10, 12],
            "timestamp_et": [now - timedelta(days=1), now - timedelta(days=1)],
        }
    )
    ledger.write_parquet(ledger_path)
    stale_ts = (now - timedelta(hours=4)).timestamp()
    os.utime(ledger_path, (stale_ts, stale_ts))

    monitors_dir = tmp_path / "reports" / "_artifacts" / "monitors"
    monitors_dir.mkdir(parents=True, exist_ok=True)
    monitors_dir.joinpath("ev_gap.json").write_text(
        json.dumps(
            {
                "name": "ev_gap",
                "status": "OK",
                "metrics": {},
                "generated_at": now.isoformat(),
            }
        ),
        encoding="utf-8",
    )

    config = RampPolicyConfig(
        lookback_days=14,
        min_fills=0,
        min_delta_bps=-10,
        min_t_stat=-10,
        ledger_max_age_minutes=60,
        monitor_max_age_minutes=120,
        seq_guard_threshold=100.0,
    )

    policy = compute_ramp_policy(
        ledger_path=ledger_path,
        artifacts_dir=tmp_path / "reports" / "_artifacts",
        monitor_artifacts_dir=monitors_dir,
        config=config,
        now=now,
    )

    assert "ledger_stale" in policy["overall"]["global_reasons"]
    ledger_age = policy["freshness"]["ledger_age_minutes"]
    assert ledger_age is not None and ledger_age > config.ledger_max_age_minutes
    for entry in policy["series"]:
        assert "ledger_stale" in entry["reasons"]
        assert entry["recommendation"] == "NO_GO"


def test_ramp_policy_panic_backoff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    now = datetime(2025, 11, 2, 18, 30, tzinfo=UTC)

    ledger_path = tmp_path / "data" / "proc" / "ledger_all.parquet"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = pl.DataFrame(
        {
            "series": ["CPI", "CLAIMS", "TENY"],
            "ev_expected_bps": [10.0, 8.0, 7.5],
            "ev_realized_bps": [11.5, 8.5, 7.9],
            "expected_fills": [50, 40, 30],
            "timestamp_et": [now - timedelta(hours=1)] * 3,
        }
    )
    ledger.write_parquet(ledger_path)

    monitors_dir = tmp_path / "reports" / "_artifacts" / "monitors"
    monitors_dir.mkdir(parents=True, exist_ok=True)
    for name in ("ev_gap", "fill_vs_alpha", "drawdown"):
        monitors_dir.joinpath(f"{name}.json").write_text(
            json.dumps(
                {
                    "name": name,
                    "status": "ALERT",
                    "metrics": {},
                    "generated_at": now.isoformat(),
                }
            ),
            encoding="utf-8",
        )

    config = RampPolicyConfig(
        lookback_days=14,
        min_fills=0,
        min_delta_bps=-10,
        min_t_stat=-10,
        panic_alert_threshold=3,
        panic_alert_window_minutes=120,
        seq_guard_threshold=100.0,
    )

    policy = compute_ramp_policy(
        ledger_path=ledger_path,
        artifacts_dir=tmp_path / "reports" / "_artifacts",
        monitor_artifacts_dir=monitors_dir,
        config=config,
        now=now,
    )

    assert policy["monitors_summary"]["panic_backoff"] is True
    assert "panic_backoff" in policy["overall"]["global_reasons"]
    for entry in policy["series"]:
        assert "panic_backoff" in entry["reasons"]
        assert entry["recommendation"] == "NO_GO"


def test_ramp_policy_sequential_alert(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    now = datetime(2025, 11, 2, 12, 0, tzinfo=UTC)

    ledger_path = tmp_path / "data" / "proc" / "ledger_all.parquet"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    deltas = [1.0, 2.2, 3.5, 4.8, 6.0, 7.5]
    ledger = pl.DataFrame(
        {
            "series": ["CPI"] * len(deltas),
            "ev_expected_bps": [10.0] * len(deltas),
            "ev_realized_bps": [10.0 + delta for delta in deltas],
            "expected_fills": [20] * len(deltas),
            "timestamp_et": [now - timedelta(minutes=idx * 5) for idx in range(len(deltas), 0, -1)],
        }
    )
    ledger.write_parquet(ledger_path)

    monitors_dir = tmp_path / "reports" / "_artifacts" / "monitors"
    monitors_dir.mkdir(parents=True, exist_ok=True)
    monitors_dir.joinpath("ev_gap.json").write_text(
        json.dumps(
            {
                "name": "ev_gap",
                "status": "OK",
                "metrics": {},
                "generated_at": now.isoformat(),
            }
        ),
        encoding="utf-8",
    )

    config = RampPolicyConfig(
        lookback_days=7,
        min_fills=0,
        min_delta_bps=-10,
        min_t_stat=-10,
        seq_guard_threshold=5.0,
        seq_guard_drift=0.5,
        seq_guard_min_sample=3,
    )

    policy = compute_ramp_policy(
        ledger_path=ledger_path,
        artifacts_dir=tmp_path / "reports" / "_artifacts",
        monitor_artifacts_dir=monitors_dir,
        config=config,
        now=now,
    )

    triggers = policy["monitors_summary"]["sequential_triggers"]
    assert triggers, "Expected sequential triggers"
    series_entry = next(entry for entry in policy["series"] if entry["series"] == "CPI")
    assert "sequential_alert" in series_entry["reasons"]
    assert series_entry["recommendation"] == "NO_GO"


def test_ramp_policy_freeze_violation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    now = datetime(2025, 11, 13, 14, 0, tzinfo=UTC)

    proc_root = tmp_path / "data" / "proc"
    calendar_dir = proc_root / "bls_cpi" / "calendar"
    calendar_dir.mkdir(parents=True, exist_ok=True)

    release_et = datetime(2025, 11, 14, 8, 30, tzinfo=ZoneInfo("America/New_York"))
    release_utc = release_et.astimezone(UTC)
    calendar_frame = pl.DataFrame(
        {
            "release_date": pl.Series([release_utc.date()], dtype=pl.Date),
            "release_datetime": pl.Series([release_utc], dtype=pl.Datetime(time_zone="UTC")),
        }
    )
    calendar_frame.write_parquet(calendar_dir / "2025-11.parquet")

    ledger_path = tmp_path / "data" / "proc" / "ledger_all.parquet"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = pl.DataFrame(
        {
            "series": ["CPI"] * 4,
            "ev_expected_bps": [10.0] * 4,
            "ev_realized_bps": [11.0, 10.8, 11.2, 10.9],
            "expected_fills": [40, 35, 30, 32],
            "timestamp_et": [now - timedelta(hours=idx + 1) for idx in range(4)],
        }
    )
    ledger.write_parquet(ledger_path)

    monitors_dir = tmp_path / "reports" / "_artifacts" / "monitors"
    monitors_dir.mkdir(parents=True, exist_ok=True)
    monitors_dir.joinpath("ev_gap.json").write_text(
        json.dumps(
            {
                "name": "ev_gap",
                "status": "OK",
                "metrics": {},
                "generated_at": now.isoformat(),
            }
        ),
        encoding="utf-8",
    )

    from kalshi_alpha.exec.policy import freeze as freeze_policy  # inline import for monkeypatch

    monkeypatch.setattr(freeze_policy, "PROC_ROOT", proc_root)

    config = RampPolicyConfig(
        lookback_days=7,
        min_fills=0,
        min_delta_bps=-10,
        min_t_stat=-10,
        seq_guard_threshold=100.0,
    )

    policy = compute_ramp_policy(
        ledger_path=ledger_path,
        artifacts_dir=tmp_path / "reports" / "_artifacts",
        monitor_artifacts_dir=monitors_dir,
        config=config,
        now=now,
    )

    freeze_series = policy["overall"]["freeze_violation_series"]
    assert freeze_series == ["CPI"]
    series_entry = next(entry for entry in policy["series"] if entry["series"] == "CPI")
    assert "freeze_window" in series_entry["reasons"]
    assert series_entry["recommendation"] == "NO_GO"

def test_ramp_policy_stale_monitors_no_go(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    now = datetime(2025, 11, 2, 15, 0, tzinfo=UTC)

    ledger_path = tmp_path / "data" / "proc" / "ledger_all.parquet"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = pl.DataFrame(
        {
            "series": ["CPI", "CLAIMS"],
            "ev_expected_bps": [10.0, 8.0],
            "ev_realized_bps": [11.0, 8.5],
            "expected_fills": [10, 12],
            "timestamp_et": [now - timedelta(hours=1), now - timedelta(hours=1)],
        }
    )
    ledger.write_parquet(ledger_path)

    monitors_dir = tmp_path / "reports" / "_artifacts" / "monitors"
    monitors_dir.mkdir(parents=True, exist_ok=True)
    monitors_dir.joinpath("ev_gap.json").write_text(
        json.dumps(
            {
                "name": "ev_gap",
                "status": "OK",
                "metrics": {},
                "generated_at": (now - timedelta(minutes=240)).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    config = RampPolicyConfig(
        lookback_days=7,
        min_fills=0,
        min_delta_bps=-10,
        min_t_stat=-10,
        ledger_max_age_minutes=120,
        monitor_max_age_minutes=60,
        seq_guard_threshold=100.0,
    )

    policy = compute_ramp_policy(
        ledger_path=ledger_path,
        artifacts_dir=tmp_path / "reports" / "_artifacts",
        monitor_artifacts_dir=monitors_dir,
        config=config,
        now=now,
    )

    freshness = policy["freshness"]
    monitors_age = freshness.get("monitors_age_minutes")
    assert monitors_age is not None and monitors_age > config.monitor_max_age_minutes
    assert "monitors_stale" in policy["overall"]["global_reasons"]
    for entry in policy["series"]:
        assert "monitors_stale" in entry["reasons"]
        assert entry["recommendation"] == "NO_GO"

def test_ramp_policy_includes_ev_honesty_bins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    now = datetime(2025, 11, 2, 12, 0, tzinfo=UTC)

    ledger_path = tmp_path / "data" / "proc" / "ledger_all.parquet"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "series": ["CPI"],
            "ev_expected_bps": [10.0],
            "ev_realized_bps": [12.0],
            "expected_fills": [50],
            "timestamp_et": [now - timedelta(hours=1)],
        }
    ).write_parquet(ledger_path)

    artifacts_dir = tmp_path / "reports" / "_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    pilot_session_path = artifacts_dir / "pilot_session.json"
    pilot_session_path.write_text(
        json.dumps(
            {
                "session_id": "pilot-cpi-test",
                "series": "CPI",
                "ev_honesty_threshold": 0.1,
                "ev_honesty_table": [
                    {
                        "market_ticker": "CPI-TEST",
                        "market_id": "M1",
                        "strike": 270.0,
                        "side": "YES",
                        "delta": 0.2,
                        "maker_ev_per_contract_original": 0.5,
                        "maker_ev_per_contract_replay": 0.3,
                        "maker_ev_per_contract_proposal": 0.45,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monitors_dir = tmp_path / "reports" / "_artifacts" / "monitors"
    monitors_dir.mkdir(parents=True, exist_ok=True)
    monitors_dir.joinpath("ev_gap.json").write_text(
        json.dumps(
            {
                "name": "ev_gap",
                "status": "OK",
                "metrics": {"mean_delta_bps": 1.0},
                "generated_at": now.isoformat(),
            }
        ),
        encoding="utf-8",
    )

    policy = compute_ramp_policy(
        ledger_path=ledger_path,
        artifacts_dir=artifacts_dir,
        monitor_artifacts_dir=monitors_dir,
        config=RampPolicyConfig(min_fills=0, min_delta_bps=-10, min_t_stat=-10),
        now=now,
    )

    series_entries = {entry["series"]: entry for entry in policy["series"]}
    bin_entries = series_entries["CPI"].get("ev_honesty_bins")
    assert bin_entries
    bin_entry = bin_entries[0]
    assert bin_entry["flagged"] is True
    assert pytest.approx(bin_entry["recommended_weight"], 0.001) == 0.5
    assert policy["overall"]["ev_honesty_flags"]["CPI"]

def test_ramp_policy_applies_manual_bin_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    now = datetime(2025, 11, 2, 12, 0, tzinfo=UTC)

    ledger_path = tmp_path / "data" / "proc" / "ledger_all.parquet"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "series": ["CPI"],
            "ev_expected_bps": [9.0],
            "ev_realized_bps": [9.5],
            "expected_fills": [20],
            "timestamp_et": [now - timedelta(hours=2)],
        }
    ).write_parquet(ledger_path)

    artifacts_dir = tmp_path / "reports" / "_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    overrides_path = tmp_path / "ramp_overrides.yaml"
    overrides_path.write_text(
        """
        series:
          CPI:
            - strike: 270
              side: YES
              weight: 0.2
              cap: 5
              reason: manual downgrade
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    monitors_dir = tmp_path / "reports" / "_artifacts" / "monitors"
    monitors_dir.mkdir(parents=True, exist_ok=True)
    monitors_dir.joinpath("ev_gap.json").write_text(
        json.dumps(
            {
                "name": "ev_gap",
                "status": "OK",
                "metrics": {},
                "generated_at": now.isoformat(),
            }
        ),
        encoding="utf-8",
    )

    policy = compute_ramp_policy(
        ledger_path=ledger_path,
        artifacts_dir=artifacts_dir,
        monitor_artifacts_dir=monitors_dir,
        config=RampPolicyConfig(min_fills=0, min_delta_bps=-10, min_t_stat=-10),
        now=now,
        bin_overrides_path=overrides_path,
    )

    series_entries = {entry["series"]: entry for entry in policy["series"]}
    bin_entries = series_entries["CPI"].get("ev_honesty_bins")
    assert bin_entries
    target = [entry for entry in bin_entries if entry.get("strike") == 270.0 and entry.get("side") == "YES"][0]
    assert pytest.approx(target["recommended_weight"], 0.001) == 0.2
    assert target.get("recommended_cap") == 5.0
    assert "manual_override" in target.get("sources", [])
    assert any("manual downgrade" in note for note in target.get("notes", []))


def test_load_ev_honesty_constraints(tmp_path: Path) -> None:
    readiness = tmp_path / "pilot_ready.json"
    readiness.write_text(
        json.dumps(
            {
                "series": [
                    {
                        "series": "CPI",
                        "ev_honesty_bins": [
                            {
                                "market_id": "M1",
                                "market_ticker": "CPI-TEST",
                                "strike": 270,
                                "side": "YES",
                                "recommended_weight": 0.5,
                                "recommended_cap": 3,
                                "sources": ["auto_ev_honesty"],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    resolver = scan_ladders._load_ev_honesty_constraints("CPI", readiness)
    assert resolver is not None and resolver.has_rules
    contracts, details = resolver.apply(
        market_id="M1",
        market_ticker="CPI-TEST",
        strike=270.0,
        side="YES",
        contracts=10,
    )
    assert contracts == 3
    assert details is not None
    summary = resolver.summary()
    assert summary["rules"] == 1
    assert summary["applied"] == 1
    assert summary.get("source_hits", {}).get("auto_ev_honesty") == 1


def test_load_ev_honesty_constraints_missing(tmp_path: Path) -> None:
    resolver = scan_ladders._load_ev_honesty_constraints("CPI", tmp_path / "missing.json")
    assert resolver is None
