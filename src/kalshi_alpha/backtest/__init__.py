"""Backtesting utilities for index ladder strategies."""

from .index_calendar import (
    TargetSpec,
    TargetType,
    enumerate_targets,
    is_trading_day,
    targets_for_day,
    trading_days,
)

__all__ = [
    "TargetSpec",
    "TargetType",
    "enumerate_targets",
    "is_trading_day",
    "targets_for_day",
    "trading_days",
]
