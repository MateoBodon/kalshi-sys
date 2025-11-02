from pathlib import Path

import pytest

from kalshi_alpha.exec.runners import pilot


def test_pilot_cli_forwards_flags(monkeypatch, tmp_path: Path) -> None:
    recorded: dict[str, list[str]] = {}

    def fake_main(args: list[str]) -> None:
        recorded["args"] = args

    monkeypatch.setattr(pilot.scan_ladders, "main", fake_main)

    config_path = tmp_path / "pilot.yaml"
    config_path.write_text("pilot: {}\n", encoding="utf-8")

    exit_code = pilot.main(
        [
            "--series",
            "CPI",
            "--pilot-config",
            str(config_path),
            "--kill-switch-file",
            "ks.txt",
            "--report",
            "--paper-ledger",
        ]
    )

    assert exit_code == 0
    forwarded = recorded["args"]
    assert forwarded.count("--pilot") == 1
    assert forwarded[:7] == [
        "--series",
        "CPI",
        "--pilot",
        "--fixtures-root",
        "tests/data_fixtures",
        "--broker",
        "live",
    ]
    assert "--pilot-config" in forwarded and str(config_path) in forwarded
    assert "--kill-switch-file" in forwarded and "ks.txt" in forwarded
    assert "--i-understand-the-risks" in forwarded


def test_pilot_cli_skips_ack_for_dry(monkeypatch) -> None:
    captured: dict[str, list[str]] = {}

    def fake_main(args: list[str]) -> None:
        captured["args"] = args

    monkeypatch.setattr(pilot.scan_ladders, "main", fake_main)

    pilot.main(["--series", "CPI", "--broker", "dry"])

    forwarded = captured["args"]
    assert "--i-understand-the-risks" not in forwarded
    assert ["--broker", "dry"] in [forwarded[i : i + 2] for i in range(len(forwarded) - 1)]


def test_pilot_cli_requires_series() -> None:
    with pytest.raises(SystemExit):
        pilot.main([])
