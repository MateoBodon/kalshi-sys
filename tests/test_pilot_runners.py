from __future__ import annotations

from pathlib import Path

import pytest

from kalshi_alpha.exec.runners import pilot_close, pilot_hourly


def test_pilot_hourly_forwards_required_flags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: list[list[str]] = []

    def fake_main(args: list[str]) -> None:  # type: ignore[override]
        captured.append(args)

    monkeypatch.setattr("kalshi_alpha.exec.runners.scan_ladders.main", fake_main)

    kill_switch = tmp_path / "kill_hourly"
    pilot_hourly.main(
        [
            "--series",
            "INXU",
            "--fixtures-root",
            str(tmp_path),
            "--kill-switch-file",
            str(kill_switch),
            "--now",
            "2025-11-04T16:55:00+00:00",
        ]
    )

    assert captured, "expected pilot hourly runner to invoke scan_ladders"
    forwarded = captured[0]
    assert forwarded[:2] == ["--series", "INXU"]
    assert "--pilot" in forwarded
    assert "--maker-only" in forwarded
    assert ["--contracts", "1"] in [forwarded[i : i + 2] for i in range(len(forwarded) - 1)]
    assert "--paper-ledger" in forwarded  # dry broker should default to paper mode
    assert ["--kill-switch-file", str(kill_switch)] in [
        forwarded[i : i + 2] for i in range(len(forwarded) - 1)
    ]


def test_pilot_hourly_requires_ack_for_live(tmp_path: Path) -> None:
    kill_switch = tmp_path / "kill_hourly"
    with pytest.raises(ValueError):
        pilot_hourly.main(
            [
                "--series",
                "INXU",
                "--broker",
                "live",
                "--fixtures-root",
                str(tmp_path),
                "--kill-switch-file",
                str(kill_switch),
                "--now",
                "2025-11-04T16:55:00+00:00",
            ]
        )


def test_pilot_close_forwards_required_flags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: list[list[str]] = []

    def fake_main(args: list[str]) -> None:  # type: ignore[override]
        captured.append(args)

    monkeypatch.setattr("kalshi_alpha.exec.runners.scan_ladders.main", fake_main)

    kill_switch = tmp_path / "kill_close"
    pilot_close.main(
        [
            "--series",
            "INX",
            "--fixtures-root",
            str(tmp_path),
            "--kill-switch-file",
            str(kill_switch),
            "--broker",
            "live",
            "--ack",
            "--report",
            "--now",
            "2025-11-04T20:55:00+00:00",
        ]
    )

    assert captured, "expected pilot close runner to invoke scan_ladders"
    forwarded = captured[0]
    assert forwarded[:2] == ["--series", "INX"]
    assert "--pilot" in forwarded
    assert "--maker-only" in forwarded
    assert ["--contracts", "1"] in [forwarded[i : i + 2] for i in range(len(forwarded) - 1)]
    assert ["--kill-switch-file", str(kill_switch)] in [
        forwarded[i : i + 2] for i in range(len(forwarded) - 1)
    ]
    assert "--i-understand-the-risks" in forwarded
