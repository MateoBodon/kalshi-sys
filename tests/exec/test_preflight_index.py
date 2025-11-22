from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from kalshi_alpha.exec.preflight_index import MAX_CALIBRATION_AGE_DAYS, run_preflight

ET = ZoneInfo("America/New_York")


def _write_params(root: Path, series: str, horizon: str, generated_at: datetime) -> Path:
    path = root / series / horizon / "params.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at": generated_at.astimezone(UTC).isoformat(), "symbols": {}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _seed_all_params(root: Path, generated_at: datetime) -> None:
    _write_params(root, "INX", "close", generated_at)
    _write_params(root, "NASDAQ100", "close", generated_at)
    _write_params(root, "INXU", "noon", generated_at)
    _write_params(root, "NASDAQ100U", "noon", generated_at)


def test_missing_env_triggers_no_go(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime(2025, 11, 3, 10, 5, tzinfo=ET)
    _seed_all_params(tmp_path, now)
    for key in ("KALSHI_API_KEY_ID", "KALSHI_PRIVATE_KEY_PEM_PATH", "POLYGON_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    result = run_preflight(
        now,
        params_root=tmp_path,
        kill_switch_file=tmp_path / "kill_switch",
        polygon_ping=lambda _: True,
    )

    assert not result.go
    assert any(reason.startswith("missing_env:") for reason in result.reasons)


def test_stale_calibration_blocks_go(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime(2025, 11, 3, 12, 0, tzinfo=ET)
    stale_ts = now - timedelta(days=MAX_CALIBRATION_AGE_DAYS + 2)
    _seed_all_params(tmp_path, stale_ts)

    key_path = tmp_path / "kalshi.pem"
    key_path.write_text("dummy", encoding="utf-8")
    monkeypatch.setenv("KALSHI_API_KEY_ID", "demo-id")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PEM_PATH", str(key_path))
    monkeypatch.setenv("POLYGON_API_KEY", "demo-polygon")

    result = run_preflight(
        now,
        params_root=tmp_path,
        kill_switch_file=tmp_path / "kill_switch",
        polygon_ping=lambda _: True,
    )

    assert not result.go
    assert any(reason.startswith("calibration_stale:") for reason in result.reasons)


def test_all_checks_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime(2025, 11, 3, 14, 30, tzinfo=ET)
    _seed_all_params(tmp_path, now)

    key_path = tmp_path / "kalshi.pem"
    key_path.write_text("dummy", encoding="utf-8")
    monkeypatch.setenv("KALSHI_API_KEY_ID", "demo-id")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PEM_PATH", str(key_path))
    monkeypatch.setenv("POLYGON_API_KEY", "demo-polygon")

    result = run_preflight(
        now,
        params_root=tmp_path,
        kill_switch_file=tmp_path / "kill_switch",
        polygon_ping=lambda _: True,
    )

    assert result.go
    assert not result.reasons
