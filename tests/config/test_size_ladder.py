from __future__ import annotations

from pathlib import Path

from kalshi_alpha.config import size_ladder


def test_load_size_ladder_parses_limits(tmp_path: Path) -> None:
    config_path = tmp_path / "size_ladder.yaml"
    config_path.write_text(
        """
current_stage: B
stages:
  A:
    description: Micro
    per_series:
      INXU:
        max_contracts: 1
        max_bins: 2
  B:
    description: Macro
    per_series:
      INXU:
        max_contracts: 2
        max_bins: 4
"""
    )
    ladder = size_ladder.load_size_ladder(config_path)
    assert ladder.current_stage == "B"
    stage_a = ladder.stage("A")
    limits = stage_a.limits_for("inxu")
    assert limits is not None
    assert limits.max_contracts == 1
    assert limits.max_bins == 2
