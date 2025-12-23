from __future__ import annotations

from pathlib import Path
import json

import polars as pl

from tools import settlement_basis_audit


def test_settlement_basis_offline_fixtures(tmp_path: Path) -> None:
    report_path = tmp_path / "report.md"
    data_path = tmp_path / "data.parquet"
    json_path = tmp_path / "summary.json"
    fixtures_root = Path("tests/fixtures/settlement_basis").resolve()

    settlement_basis_audit.main(
        [
            "--day",
            "2025-11-10",
            "--series",
            "INXU",
            "--out-report",
            str(report_path),
            "--out-json",
            str(json_path),
            "--out-data",
            str(data_path),
            "--offline-fixtures",
            str(fixtures_root),
        ]
    )

    assert data_path.exists()
    frame = pl.read_parquet(data_path)
    expected_columns = {
        "day",
        "series",
        "window_label",
        "window_ts_et",
        "window_ts_utc",
        "kalshi_value",
        "kalshi_source_field",
        "kalshi_market_or_event_id",
        "polygon_value",
        "polygon_source",
        "polygon_ts_utc",
        "basis",
        "nearest_strike",
        "nearest_strike_margin",
        "strike_spacing",
        "flip_risk",
    }
    assert expected_columns.issubset(set(frame.columns))
    assert frame.filter(pl.col("flip_risk") == True).height >= 1

    assert json_path.exists()
    summary = json_path.read_text(encoding="utf-8")
    payload = json.loads(summary)
    assert payload.get("series") == "INXU"
    assert payload.get("asof_date") == "2025-11-10"
    assert "generated_at" in payload
    assert isinstance(payload.get("sample_count"), int)
    basis_quantiles = payload.get("basis_quantiles", {})
    for key in ("p01", "p05", "p50", "p95", "p99"):
        assert key in basis_quantiles
    per_window = payload.get("per_window_deltas", [])
    assert per_window
    first = per_window[0]
    for key in ("window_id", "n", "mean", "p05", "p50", "p95"):
        assert key in first
    flip_risk = payload.get("flip_risk", {})
    assert "flag" in flip_risk
    assert "rationale" in flip_risk
    assert "thresholds" in flip_risk

    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "p95=" in report_text
    assert "p99=" in report_text
