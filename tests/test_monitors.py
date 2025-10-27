from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from kalshi_alpha.exec.pipelines import daily
from kalshi_alpha.exec.reports import write_markdown_report
from kalshi_alpha.exec.runners import scan_ladders


def test_model_drift_flag(tmp_path: Path, isolated_data_roots: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    _, proc_root = isolated_data_roots
    calib_path = proc_root / "cpi_calib.parquet"
    frame = pl.DataFrame(
        {
            "record_type": ["params"],
            "crps": [0.25],
            "baseline_crps": [0.2],
            "brier": [0.6],
            "baseline_brier": [0.5],
        }
    )
    frame.write_parquet(calib_path)

    assert scan_ladders._model_drift_flag("CPI") is True

    monitors = {"model_drift": True, "tz_not_et": False, "non_monotone_ladders": 2}
    report_path = write_markdown_report(
        series="CPI",
        proposals=[],
        ledger=None,
        output_dir=tmp_path,
        monitors=monitors,
    )
    contents = report_path.read_text(encoding="utf-8")
    assert "Monitors" in contents
    assert "model_drift: True" in contents

    log = {"monitors": monitors}
    monkeypatch.setattr(daily, "PROC_ROOT", proc_root)
    timestamp = datetime(2025, 10, 25, 12, 0, tzinfo=UTC)
    daily.write_log("pre_cpi", log, timestamp)
    logs = list((proc_root / "logs").rglob("*.json"))
    assert logs, "Expected monitor log file"
    assert "model_drift" in logs[0].read_text(encoding="utf-8")
