from __future__ import annotations

import json

import polars as pl
import pytest

from kalshi_alpha.strategies.cpi import calibrate


def test_cpi_yoy_guard(offline_fixtures_root):
    history = json.loads(
        (offline_fixtures_root / "cpi" / "history.json").read_text(encoding="utf-8")
    )["history"]
    frame = calibrate(history)
    evaluations = frame.filter(pl.col("record_type") == "evaluation")
    for idx, row in enumerate(evaluations.iter_rows(named=True)):
        entry = history[idx]
        prev_yoy = entry.get("prev_yoy")
        base_effect = entry.get("base_effect")
        if prev_yoy is None or base_effect is None:
            continue
        expected = float(prev_yoy) + float(row["mom_mean"]) - float(base_effect)
        assert pytest.approx(row["yoy_prediction"], abs=1e-6) == expected
