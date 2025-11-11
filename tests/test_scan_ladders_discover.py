from __future__ import annotations

from kalshi_alpha.exec.runners import scan_ladders


def test_scan_ladders_discover_offline(capfd) -> None:
    args = [
        "--discover",
        "--offline",
        "--fixtures-root",
        "tests/data_fixtures",
        "--discover-date",
        "2025-11-10",
    ]
    scan_ladders.main(args)
    output = capfd.readouterr().out
    assert "[discover]" in output
    assert "hourly-1000" in output
    assert "INXU" in output
