from __future__ import annotations

import json
from pathlib import Path

from kalshi_alpha.core.execution.index_models import load_alpha_curve, load_slippage_curve


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_alpha_curve_improves_error() -> None:
    alpha_path = Path("data/proc/index_exec/INXU/alpha.json")
    assert alpha_path.exists(), "Expected alpha curve to be generated"
    payload = _load_json(alpha_path)
    assert payload["model_mse"] <= payload["baseline_mse"]

    curve = load_alpha_curve("INXU")
    assert curve is not None
    low = curve.predict(depth_fraction=0.1, delta_p=0.0, tau_minutes=120.0)
    high = curve.predict(depth_fraction=0.9, delta_p=0.0, tau_minutes=120.0)
    assert high >= low


def test_slippage_curve_improves_error() -> None:
    slip_path = Path("data/proc/index_exec/INXU/slippage.json")
    assert slip_path.exists(), "Expected slippage curve to be generated"
    payload = _load_json(slip_path)
    assert payload["model_mse"] <= payload["baseline_mse"]

    curve = load_slippage_curve("INXU")
    assert curve is not None
    tight = curve.predict_ticks(depth_fraction=0.9, spread=0.02, tau_minutes=30.0)
    wide = curve.predict_ticks(depth_fraction=0.1, spread=0.10, tau_minutes=30.0)
    assert wide >= tight
