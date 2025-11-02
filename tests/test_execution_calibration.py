from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from kalshi_alpha.core.execution import fillratio, slippage


def _prepare_ledger(tmp_path: Path) -> Path:
    now = datetime.now(tz=UTC)
    frame = pl.DataFrame(
        {
            "series": ["cpi", "CPI", "CPI", "CPI", "teny"],
            "size": [100, 30, 80, 60, 40],
            "size_partial": [20, 6, 16, 12, 8],
            "slippage_ticks": [1.0, 1.0, 3.0, 2.0, 1.5],
            "delta_p": [0.03, 0.04, 0.08, 0.07, 0.05],
            "timestamp_et": [
                now - timedelta(days=1),
                now - timedelta(days=2),
                now - timedelta(days=3),
                now - timedelta(days=4),
                now - timedelta(days=1),
            ],
        }
    )
    ledger_path = tmp_path / "data" / "proc" / "ledger_all.parquet"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(ledger_path)
    return ledger_path


def test_tune_alpha_persists_and_loads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ledger_path = _prepare_ledger(tmp_path)
    state_path = tmp_path / "data" / "proc" / "state" / "fill_alpha.json"
    monkeypatch.setattr(fillratio, "LEDGER_PATH", ledger_path)
    monkeypatch.setattr(fillratio, "STATE_PATH", state_path)

    alpha_cpi = fillratio.tune_alpha("cpi", lookback_days=7, min_observations=1)
    assert alpha_cpi == pytest.approx(0.8, rel=1e-3)

    alpha_teny = fillratio.tune_alpha("10y", lookback_days=7, min_observations=1)
    assert alpha_teny == pytest.approx(0.8, rel=1e-3)

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["series"]["CPI"]["sample_size"] == 4
    assert payload["series"]["TENY"]["sample_size"] == 1
    assert fillratio.load_alpha("CPI") == pytest.approx(0.8, rel=1e-3)
    assert fillratio.load_alpha("teny") == pytest.approx(0.8, rel=1e-3)


def test_fit_slippage_persists_and_loads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ledger_path = _prepare_ledger(tmp_path)
    state_path = tmp_path / "data" / "proc" / "state" / "slippage.json"
    monkeypatch.setattr(slippage, "LEDGER_PATH", ledger_path)
    monkeypatch.setattr(slippage, "STATE_PATH", state_path)

    calibration = slippage.fit_slippage("CPI", lookback_days=7, min_observations=1)
    assert calibration is not None
    assert calibration.family == "CPI"
    assert calibration.sample_size == 4
    assert calibration.impact_cap == pytest.approx(0.03, rel=1e-3)
    assert calibration.depth_curve == (
        (0.0, 0.0),
        (0.35, 0.01),
        (0.65, 0.02),
        (1.0, 0.03),
    )

    model = slippage.load_slippage_model("cpi")
    assert model is not None
    assert model.mode == "depth"
    assert model.impact_cap == pytest.approx(0.03, rel=1e-3)
    assert tuple(model.depth_curve) == calibration.depth_curve
