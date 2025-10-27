from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from kalshi_alpha.core.gates import run_quality_gates


def _write_parquet(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)


def test_quality_gates_pass(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    raw_root = tmp_path / "raw"
    proc_root.mkdir()
    raw_root.mkdir()

    # Metrics snapshot
    calib_frame = pl.DataFrame(
        {
            "record_type": ["params"],
            "crps": [0.18],
            "baseline_crps": [0.22],
            "brier": [0.04],
            "baseline_brier": [0.06],
        }
    )
    _write_parquet(proc_root / "cpi_calib.parquet", calib_frame)

    # Cleveland nowcast processed snapshot (fresh within 6 hours)
    now = datetime(2025, 10, 27, 12, 0, tzinfo=UTC)
    recent = now - timedelta(hours=3)
    cleveland_frame = pl.DataFrame(
        {
            "series": ["headline"],
            "label": ["Headline"],
            "as_of": [recent],
            "value": [0.21],
        }
    )
    _write_parquet(proc_root / "cleveland_nowcast/monthly/20251027T090000.parquet", cleveland_frame)

    # Treasury yields snapshot with overlapping maturities
    treasury_frame = pl.DataFrame(
        {
            "as_of": [date(2025, 10, 24), date(2025, 10, 24)],
            "maturity": ["10 YR", "DGS10"],
            "rate": [0.0435, 0.0433],
        }
    )
    _write_parquet(proc_root / "treasury_yields/daily/20251024T210000.parquet", treasury_frame)

    config_path = tmp_path / "configs/quality_gates.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        """
metrics:
  series:
    cpi:
      crps_advantage_min: 0.03
      brier_advantage_min: 0.01
data_freshness:
  - name: cleveland
    namespace: cleveland_nowcast/monthly
    timestamp_field: as_of
    max_age_hours: 6
reconciliation:
  - name: t10_vs_dgs
    namespace: treasury_yields/daily
    par_maturity: "10 YR"
    dgs_maturity: "DGS10"
    tolerance_bps: 5.0
monitors:
  tz_not_et: 0
""",
        encoding="utf-8",
    )

    result = run_quality_gates(
        config_path=config_path,
        monitors={"tz_not_et": 0},
        now=now,
        proc_root=proc_root,
        raw_root=raw_root,
    )

    assert result.go is True
    assert result.reasons == []
    assert result.details.get("metrics.cpi.crps_advantage") == pytest.approx(0.04, abs=1e-6)


def test_quality_gates_failures(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    raw_root = tmp_path / "raw"
    proc_root.mkdir()
    raw_root.mkdir()

    calib_frame = pl.DataFrame(
        {
            "record_type": ["params"],
            "crps": [0.3],
            "baseline_crps": [0.32],
            "brier": [0.06],
            "baseline_brier": [0.055],
        }
    )
    _write_parquet(proc_root / "cpi_calib.parquet", calib_frame)

    stale_time = datetime(2025, 10, 20, 12, 0, tzinfo=UTC)
    cleveland_frame = pl.DataFrame(
        {
            "series": ["headline"],
            "label": ["Headline"],
            "as_of": [stale_time.replace(tzinfo=None)],
            "value": [0.19],
        }
    )
    _write_parquet(proc_root / "cleveland_nowcast/monthly/20251020T120000.parquet", cleveland_frame)

    treasury_frame = pl.DataFrame(
        {
            "as_of": [date(2025, 10, 19), date(2025, 10, 19)],
            "maturity": ["10 YR", "DGS10"],
            "rate": [0.10, 0.04],
        }
    )
    _write_parquet(proc_root / "treasury_yields/daily/20251019T230000.parquet", treasury_frame)

    config_text = """
metrics:
  series:
    cpi:
      crps_advantage_min: 0.05
      brier_advantage_min: 0.02
data_freshness:
  - name: cleveland
    namespace: cleveland_nowcast/monthly
    timestamp_field: as_of
    max_age_hours: 12
    require_et: true
reconciliation:
  - name: t10_vs_dgs
    namespace: treasury_yields/daily
    par_maturity: "10 YR"
    dgs_maturity: "DGS10"
    tolerance_bps: 5.0
monitors:
  tz_not_et: 0
"""
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(config_text, encoding="utf-8")

    result = run_quality_gates(
        config_path=config_path,
        monitors={"tz_not_et": True},
        now=datetime(2025, 10, 27, 12, 0, tzinfo=UTC),
        proc_root=proc_root,
        raw_root=raw_root,
    )

    assert result.go is False
    joined = "|".join(result.reasons)
    assert "metrics.cpi.crps_advantage" in joined
    assert "metrics.cpi.brier_advantage" in joined
    assert "data_freshness.cleveland.timezone_missing" in joined
    assert "data_freshness.cleveland.stale" in joined
    assert "reconciliation.t10_vs_dgs.diff" in joined
    assert "monitors.tz_not_et" in joined
