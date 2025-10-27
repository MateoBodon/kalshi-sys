from __future__ import annotations

import math

import pytest

from kalshi_alpha.core.sizing.kelly import apply_caps, kelly_yes_no, truncate_kelly


def test_kelly_yes_no_signs() -> None:
    positive = kelly_yes_no(0.62, 0.55)
    negative = kelly_yes_no(0.45, 0.55)
    neutral = kelly_yes_no(0.55, 0.55)
    assert positive > 0
    assert negative < 0
    assert neutral == pytest.approx(0.0, abs=1e-12)


def test_truncate_kelly_bounds() -> None:
    assert truncate_kelly(0.4, 0.25) == 0.25
    assert truncate_kelly(-0.4, 0.25) == -0.25
    assert truncate_kelly(0.1, 0.25) == 0.1
    with pytest.raises(ValueError):
        truncate_kelly(0.1, 0.0)


def test_apply_caps_respects_smallest() -> None:
    size = 1_000.0
    result = apply_caps(size, pal=800.0, max_loss_per_strike=1_200.0, max_var=500.0)
    assert result == 500.0

    no_var = apply_caps(size, pal=800.0, max_loss_per_strike=1_200.0, max_var=None)
    assert no_var == 800.0

    assert apply_caps(100.0, pal=0.0, max_loss_per_strike=1_200.0, max_var=None) == 0.0
