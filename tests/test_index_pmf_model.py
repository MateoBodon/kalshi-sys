from __future__ import annotations

import json
from pathlib import Path

from kalshi_alpha.models import pmf_index


def test_load_model_and_pmf(tmp_path: Path) -> None:
    payload = {
        "version": 1,
        "series": "INXU",
        "target_type": "hourly",
        "target_label": "1200",
        "sigma_curve": {"0": 0.6, "1": 0.8, "5": 1.2},
        "drift_curve": {"0": 0.0, "1": 0.05, "5": 0.1},
        "residual_std": 0.4,
        "min_std": 0.5,
        "eod_bump": {"minutes_threshold": 2, "variance": 0.04},
        "metadata": {"records": 100},
    }
    target_dir = tmp_path / "INXU" / "hourly"
    target_dir.mkdir(parents=True)
    params_path = target_dir / "1200.json"
    params_path.write_text(json.dumps(payload), encoding="utf-8")

    model = pmf_index.load_model("INXU", "hourly", "1200", root=tmp_path)
    strikes = [4990.0, 5000.0, 5010.0]
    minutes_to_target = 1
    pmf = model.pmf(strikes, minutes_to_target=minutes_to_target, current_price=5000.0)
    assert len(pmf) == len(strikes) + 1
    total = sum(bin.probability for bin in pmf)
    assert 0.99 <= total <= 1.01
    targets = pmf_index.available_targets("INXU", "hourly", root=tmp_path)
    assert targets == ["1200"]
