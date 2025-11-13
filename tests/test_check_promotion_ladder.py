from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
from zoneinfo import ZoneInfo

from scripts import check_promotion_ladder


def _write_size_config(path: Path, current: str = "A") -> None:
    path.write_text(
        f"""
current_stage: {current}
stages:
  A:
    description: A
    per_series:
      INXU:
        max_contracts: 1
        max_bins: 1
  B:
    description: B
    per_series:
      INXU:
        max_contracts: 2
        max_bins: 2
"""
    )


def _write_pnl_file(path: Path, target_day: date, pnl: float) -> None:
    frame = pl.DataFrame(
        {
            "scope": ["day"],
            "window_date": [target_day.isoformat()],
            "window_label": [f"daily-INXU"],
            "series": ["INXU"],
            "ev_after_fees": [0.1],
            "pnl_realized": [pnl],
            "delta_ev_cents_per_lot": [0.5],
            "fill_gap_pp": [1.0],
            "contracts": [1],
            "var_exposure": [0.1],
            "var_headroom_pct": [99.0],
        }
    )
    frame.write_parquet(path)


def _write_slo(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"series": "INXU", "no_go_reasons": []}
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_check_promotion_ladder_promotes_when_green(tmp_path: Path, capsys) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    today = datetime.now(tz=ZoneInfo("America/New_York")).date()
    for offset in range(7):
        day = today - timedelta(days=6 - offset)
        _write_pnl_file(artifacts / f"pnl_window_{day.isoformat()}.parquet", day, pnl=0.2)
    slo_path = artifacts / "monitors" / "slo_selfcheck.json"
    _write_slo(slo_path)
    size_config = tmp_path / "size.yaml"
    _write_size_config(size_config, current="A")

    exit_code = check_promotion_ladder.main(
        [
            "--lookback=7",
            f"--artifacts={artifacts}",
            f"--slo={slo_path}",
            f"--size-config={size_config}",
            "--series=INXU",
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Recommended stage: B" in out


def test_check_promotion_ladder_holds_when_not_green(tmp_path: Path, capsys) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    today = datetime.now(tz=ZoneInfo("America/New_York")).date()
    for offset in range(7):
        day = today - timedelta(days=6 - offset)
        pnl = 0.2 if offset < 6 else -5.0
        _write_pnl_file(artifacts / f"pnl_window_{day.isoformat()}.parquet", day, pnl=pnl)
    slo_path = artifacts / "monitors" / "slo_selfcheck.json"
    _write_slo(slo_path)
    size_config = tmp_path / "size.yaml"
    _write_size_config(size_config, current="A")

    exit_code = check_promotion_ladder.main(
        [
            "--lookback=7",
            f"--artifacts={artifacts}",
            f"--slo={slo_path}",
            f"--size-config={size_config}",
            "--series=INXU",
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Recommended stage: A" in out
