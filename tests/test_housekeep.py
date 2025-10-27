from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
import os

import pytest

from kalshi_alpha.exec import housekeep


def _touch(path: Path, *, days_old: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    ts = (datetime.now(tz=UTC) - timedelta(days=days_old)).timestamp()
    os.utime(path, times=(ts, ts))


def test_housekeep_prunes_old_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    # Create older artifacts
    old_cpi = tmp_path / "reports" / "CPI" / "pre_cpi_eve_20240101"
    _touch(old_cpi, days_old=60)
    older_cpi = tmp_path / "reports" / "CPI" / "pre_cpi_eve_20231231"
    _touch(older_cpi, days_old=90)

    recent_cpi_day = tmp_path / "reports" / "CPI" / "pre_cpi_day_20240301"
    _touch(recent_cpi_day, days_old=5)

    old_claims = tmp_path / "reports" / "CLAIMS" / "claims_freeze_202401"
    _touch(old_claims, days_old=70)
    newer_claims = tmp_path / "reports" / "CLAIMS" / "claims_freeze_202402"
    _touch(newer_claims, days_old=40)

    generic_report = tmp_path / "reports" / "misc_report"
    _touch(generic_report, days_old=80)

    raw_archive = tmp_path / "data" / "raw" / "kalshi" / "2023-12-01"
    _touch(raw_archive, days_old=120)

    newest_teny = tmp_path / "reports" / "TNEY" / "teny_close_20240401"
    _touch(newest_teny, days_old=50)
    older_teny = tmp_path / "reports" / "TNEY" / "teny_close_20240101"
    _touch(older_teny, days_old=90)

    # Run housekeeping with 30-day retention
    housekeep.main(["--keep-days", "30"])

    # Old generic report removed
    assert not generic_report.exists()
    # Old archive removed
    assert not raw_archive.exists()

    # The newest CPI eve artifact should be preserved despite age; older removed
    assert old_cpi.exists()
    assert not older_cpi.exists()

    # Claims freeze newest preserved, older removed
    assert newer_claims.exists()
    assert not old_claims.exists()

    # Teny newest preserved, older removed
    assert newest_teny.exists()
    assert not older_teny.exists()

    # Recent items within retention remain untouched
    assert recent_cpi_day.exists()
