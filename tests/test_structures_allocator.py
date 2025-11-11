from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kalshi_alpha.structures import (
    Allocator,
    AllocatorConfig,
    SeriesWindowSample,
    VarSnapshot,
)


def _samples(values: list[float], *, start: datetime | None = None) -> list[SeriesWindowSample]:
    base = start or datetime(2025, 11, 5, tzinfo=UTC)
    return [
        SeriesWindowSample(
            timestamp=base + timedelta(days=idx),
            ev_after_fees=value,
            fill_ratio=0.85,
            honesty=0.9,
            weight=25,
        )
        for idx, value in enumerate(values)
    ]


def test_allocator_prefers_positive_sharpe() -> None:
    config = AllocatorConfig(
        window=16,
        max_age_days=90,
        min_samples=3,
        base_capital=1_000.0,
        min_sharpe=0.1,
        vol_floor=5.0,
    )
    allocator = Allocator(config)
    allocator.register_many("INXU", _samples([40.0, 38.0, 45.0, 44.0]))
    allocator.register_many("NASDAQ100U", _samples([-5.0, -3.0, 1.0, -2.0]))
    var_state = VarSnapshot(
        portfolio_limit=5_000.0,
        portfolio_var=0.0,
        family_limits={"SPX": 2_500.0, "NDX": 2_500.0},
        family_var={"SPX": 0.0, "NDX": 0.0},
    )
    result = allocator.allocate(var_state)
    assert result.series["INXU"].capital > 0.0
    assert result.series["NASDAQ100U"].capital == 0.0
    assert "negative_signal" in result.series["NASDAQ100U"].reasons


def test_allocator_respects_family_and_portfolio_headroom() -> None:
    config = AllocatorConfig(
        window=8,
        max_age_days=30,
        min_samples=3,
        base_capital=5_000.0,
        min_sharpe=0.1,
        vol_floor=1.0,
        var_buffer=0.0,
    )
    allocator = Allocator(config)
    allocator.register_many("INXU", _samples([32.0, 28.0, 36.0, 33.0]))
    allocator.register_many("NASDAQ100U", _samples([35.0, 31.0, 38.0, 37.0]))
    var_state = VarSnapshot(
        portfolio_limit=1_000.0,
        portfolio_var=800.0,
        family_limits={"SPX": 300.0, "NDX": 600.0},
        family_var={"SPX": 280.0, "NDX": 0.0},
    )
    result = allocator.allocate(var_state)
    spx_budget = result.series["INXU"].capital
    ndx_budget = result.series["NASDAQ100U"].capital
    assert spx_budget <= 20.0
    assert ndx_budget <= 200.0
    assert "family_headroom" in result.series["INXU"].reasons


def test_allocator_history_roundtrip(tmp_path: Path) -> None:
    config = AllocatorConfig(
        window=4,
        max_age_days=10,
        min_samples=1,
        base_capital=500.0,
        min_sharpe=0.0,
        vol_floor=1.0,
    )
    allocator = Allocator(config)
    allocator.register("INXU", _samples([25.0])[0])
    path = tmp_path / "history.json"
    allocator.save_history(path)
    reloaded = Allocator.from_history(path, config=config)
    result = reloaded.allocate()
    assert "INXU" in result.series
    assert result.series["INXU"].samples == 1
