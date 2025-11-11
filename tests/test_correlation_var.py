from __future__ import annotations

from kalshi_alpha.core.pricing import OrderSide
from kalshi_alpha.risk.correlation import CorrelationAwareLimiter, CorrelationConfig


def _config() -> CorrelationConfig:
    return CorrelationConfig(
        portfolio_limit=15.0,
        z_score=2.0,
        family_limits={"SPX": 15.0, "NDX": 15.0},
        tilt_limits={"SPX": 12.0, "NDX": 12.0},
        correlations={
            "SPX": {"SPX": 1.0, "NDX": 0.8},
            "NDX": {"SPX": 0.8, "NDX": 1.0},
        },
        series_family={
            "INX": "SPX",
            "INXU": "SPX",
            "NASDAQ100": "NDX",
            "NASDAQ100U": "NDX",
        },
    )


def test_cap_contracts_limits_variance() -> None:
    limiter = CorrelationAwareLimiter(_config())
    limiter.update_surface("INX", [5000.0], [0.5])
    allowed, exposure, meta = limiter.cap_contracts(
        series="INX",
        strike=5000.0,
        side=OrderSide.YES,
        contracts=20,
        probability=0.5,
    )
    assert allowed < 20
    assert exposure is not None
    assert meta["portfolio_var"] <= limiter.config.portfolio_limit + 1e-6
    limiter.register(exposure)
    snapshot = limiter.snapshot()
    assert snapshot["portfolio_var"] <= limiter.config.portfolio_limit + 1e-6


def test_cross_series_correlation_restricts_second_order() -> None:
    limiter = CorrelationAwareLimiter(_config())
    limiter.update_surface("INX", [5000.0], [0.5])
    limiter.update_surface("NASDAQ100", [15000.0], [0.5])
    allowed_a, exposure_a, _ = limiter.cap_contracts(
        series="INX",
        strike=5000.0,
        side=OrderSide.YES,
        contracts=8,
        probability=0.5,
    )
    assert allowed_a > 0
    assert exposure_a is not None
    limiter.register(exposure_a)
    allowed_b, exposure_b, meta_b = limiter.cap_contracts(
        series="NASDAQ100",
        strike=15000.0,
        side=OrderSide.YES,
        contracts=8,
        probability=0.5,
    )
    assert allowed_b < 8
    assert exposure_b is not None


def test_inventory_tilt_pushes_size_lower() -> None:
    limiter = CorrelationAwareLimiter(_config())
    limiter.update_surface("INX", [5000.0], [0.5])
    first_allowed, first_exposure, _ = limiter.cap_contracts(
        series="INX",
        strike=5000.0,
        side=OrderSide.YES,
        contracts=6,
        probability=0.5,
    )
    assert first_allowed == 6
    assert first_exposure is not None
    limiter.register(first_exposure)
    second_allowed, second_exposure, meta = limiter.cap_contracts(
        series="INX",
        strike=5000.0,
        side=OrderSide.YES,
        contracts=6,
        probability=0.5,
    )
    assert second_allowed < 6
    assert meta["tilt"] < 1.0
    if second_exposure is not None:
        limiter.register(second_exposure)
