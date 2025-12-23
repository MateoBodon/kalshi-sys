from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from kalshi_alpha.exec import preflight_index
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


def _write_freshness_artifact(
    path: Path,
    *,
    polygon_ok: bool = True,
    macro_ok: bool = True,
    macro_required: bool = True,
) -> None:
    feeds = [
        {
            "id": "polygon_index.websocket",
            "label": "Polygon index websocket",
            "required": True,
            "ok": polygon_ok,
            "age_minutes": 0.001,
            "scope": "index",
        }
    ]
    required_feeds = ["polygon_index.websocket"]
    stale_feeds = [] if polygon_ok else ["polygon_index.websocket"]

    if macro_required:
        feeds.append(
            {
                "id": "macro_calendar.latest",
                "label": "Macro calendar",
                "required": True,
                "ok": macro_ok,
                "age_minutes": 45.0,
                "scope": "macro",
            }
        )
        required_feeds.append("macro_calendar.latest")
        if not macro_ok:
            stale_feeds.append("macro_calendar.latest")

    payload = {
        "name": "data_freshness",
        "status": "OK" if not stale_feeds else "ALERT",
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "metrics": {
            "required_feeds_ok": not stale_feeds,
            "required_feeds": required_feeds,
            "stale_feeds": stale_feeds,
            "feeds": feeds,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_missing_env_triggers_no_go(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime(2025, 11, 3, 10, 5, tzinfo=ET)
    _seed_all_params(tmp_path, now)
    freshness_path = tmp_path / "freshness.json"
    _write_freshness_artifact(freshness_path)
    for key in ("KALSHI_API_KEY_ID", "KALSHI_PRIVATE_KEY_PEM_PATH", "POLYGON_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    result = run_preflight(
        now,
        params_root=tmp_path,
        kill_switch_file=tmp_path / "kill_switch",
        polygon_ping=lambda _: True,
        freshness_artifact_path=freshness_path,
    )

    assert not result.go
    assert any(reason.startswith("missing_env:") for reason in result.reasons)


def test_stale_calibration_blocks_go(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime(2025, 11, 3, 12, 0, tzinfo=ET)
    stale_ts = now - timedelta(days=MAX_CALIBRATION_AGE_DAYS + 2)
    _seed_all_params(tmp_path, stale_ts)
    freshness_path = tmp_path / "freshness.json"
    _write_freshness_artifact(freshness_path)

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
        freshness_artifact_path=freshness_path,
    )

    assert not result.go
    assert any(reason.startswith("calibration_stale:") for reason in result.reasons)


def test_all_checks_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime(2025, 11, 3, 14, 30, tzinfo=ET)
    _seed_all_params(tmp_path, now)
    freshness_path = tmp_path / "freshness.json"
    _write_freshness_artifact(freshness_path)

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
        freshness_artifact_path=freshness_path,
    )

    assert result.go
    assert not result.reasons


def test_preflight_cli_emits_summary_and_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    now = datetime(2025, 11, 3, 14, 30, tzinfo=ET)
    params_root = tmp_path / "params"
    _seed_all_params(params_root, now)
    output_path = tmp_path / "go_no_go.json"
    monkeypatch.setattr(preflight_index, "GO_NO_GO_PATH", output_path)

    exit_code = preflight_index.main(
        [
            "--offline",
            "--now",
            now.isoformat(),
            "--params-root",
            str(params_root),
        ]
    )

    stdout = capsys.readouterr().out.strip()
    assert stdout
    assert "PRECHECK index:" in stdout
    assert "GO reasons=0" in stdout
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload.get("go") is True
    assert payload.get("scope") == "index"
    assert payload.get("scoped_blockers") == []
    assert payload.get("unscoped_blockers") == []
    assert exit_code == 0


def test_macro_stale_does_not_block_index_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime(2025, 11, 3, 14, 30, tzinfo=ET)
    _seed_all_params(tmp_path, now)
    freshness_path = tmp_path / "freshness.json"
    _write_freshness_artifact(freshness_path, polygon_ok=True, macro_ok=False, macro_required=True)

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
        freshness_artifact_path=freshness_path,
    )

    assert result.go
    assert "STALE_FEEDS" not in result.reasons
    assert "polygon_ws_stale" not in result.reasons
