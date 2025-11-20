from __future__ import annotations

from pathlib import Path

import pytest

from kalshi_alpha.exec import scoreboard
from kalshi_alpha.exec.pipelines import daily


def _common_daily_args() -> list[str]:
    return [
        "--mode",
        "pre_cpi",
        "--offline",
        "--driver-fixtures",
        "tests/fixtures",
        "--scanner-fixtures",
        "tests/data_fixtures",
    ]


def test_daily_skips_for_index_family(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    called = False

    def _run_mode(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        nonlocal called
        called = True

    monkeypatch.setattr(daily, "run_mode", _run_mode)

    daily.main(_common_daily_args())

    out = capsys.readouterr().out
    assert "FAMILY=index" in out
    assert not called


def test_daily_runs_when_family_macro(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def _run_mode(mode, args):  # type: ignore[no-untyped-def]
        nonlocal called
        called = True

    monkeypatch.setattr(daily, "run_mode", _run_mode)
    daily.main([*_common_daily_args(), "--family", "macro"])
    assert called


def test_scoreboard_skips_for_macro_family(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    scoreboard.main(["--family", "macro"])
    out = capsys.readouterr().out
    assert "skipping index scoreboard" in out.lower()
