from __future__ import annotations

from pathlib import Path

import yaml

from kalshi_alpha.core.risk import PALPolicy, PortfolioConfig

CONFIG_ROOT = Path("configs")


def test_pal_policy_limits_are_capped() -> None:
    caps = {
        "INXU": 1000,
        "NASDAQ100U": 1000,
        "INX": 1200,
        "NASDAQ100": 1200,
    }
    policy_path = CONFIG_ROOT / "pal_policy.yaml"
    for series, cap in caps.items():
        policy = PALPolicy.from_yaml(policy_path, series=series)
        assert policy.default_max_loss <= cap
        for limit in policy.per_strike.values():
            assert limit <= cap
        sample_strike = f"KX{series}-TEST@0"
        assert policy.limit_for_strike(sample_strike) <= cap


def test_portfolio_config_conservative() -> None:
    config = PortfolioConfig.from_yaml(CONFIG_ROOT / "portfolio.yaml")
    baseline_vols = {
        "INFLATION": 1.2,
        "RATES": 1.0,
        "EMPLOYMENT": 0.9,
        "WEATHER": 1.1,
    }
    for factor, minimum in baseline_vols.items():
        assert config.factor_vols.get(factor, 0.0) >= minimum

    baseline_betas = {
        "CPI": {"INFLATION": 1.1, "RATES": 0.2},
        "TENY": {"RATES": 1.2},
        "CLAIMS": {"EMPLOYMENT": 0.8, "INFLATION": 0.3},
        "WEATHER": {"WEATHER": 1.0},
    }
    for strategy, expectations in baseline_betas.items():
        betas = config.strategy_betas.get(strategy)
        assert betas is not None
        for factor, minimum in expectations.items():
            assert betas.get(factor, 0.0) >= minimum


def test_quality_gate_thresholds_enforced() -> None:
    payload = yaml.safe_load((CONFIG_ROOT / "quality_gates.yaml").read_text(encoding="utf-8"))
    default_metrics = payload["metrics"]["default"]
    assert default_metrics["crps_advantage_min"] >= 0.01
    assert default_metrics["brier_advantage_min"] >= 0.0

    series_baselines = {
        "cpi": {"crps_advantage_min": 0.02, "brier_advantage_min": 0.0},
        "claims": {"crps_advantage_min": 0.015, "brier_advantage_min": -0.001},
        "teny": {"crps_advantage_min": 0.01, "brier_advantage_min": -0.001},
        "weather": {"crps_advantage_min": 0.005, "brier_advantage_min": -0.002},
    }
    for series, expectations in series_baselines.items():
        metrics = payload["metrics"]["series"][series]
        for key, minimum in expectations.items():
            assert metrics[key] >= minimum

    monitor_caps = {
        "tz_not_et": 0,
        "non_monotone_ladders": 0,
        "negative_ev_after_fees": 0,
    }
    for name, maximum in monitor_caps.items():
        assert payload["monitors"][name] <= maximum
