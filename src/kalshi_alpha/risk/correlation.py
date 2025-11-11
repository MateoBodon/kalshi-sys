"""Correlation-aware VaR limiter with inventory tilt support for index ladders."""

from __future__ import annotations

import bisect
import math
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from kalshi_alpha.core.pricing import OrderSide

from .var_index import SERIES_FAMILY as BASE_SERIES_FAMILY

DEFAULT_CONFIG_PATH = Path("configs/index_correlation.yaml")
_EPSILON = 1e-9


def _normalize_series(value: str) -> str:
    return value.strip().upper()


def _strike_key(strike: float) -> str:
    return f"{float(strike):.4f}"


class ProbabilitySurface:
    """Accumulates strike → survival probability mappings per series."""

    def __init__(self) -> None:
        self._points: dict[str, float] = {}
        self._strikes: list[float] = []

    def update(self, strikes: Sequence[float], survival: Sequence[float]) -> None:
        if len(strikes) != len(survival):
            raise ValueError("strikes and survival must have the same length")
        ordered = sorted(zip(strikes, survival, strict=True), key=lambda item: item[0])
        updated: set[float] = set(self._strikes)
        for strike, prob in ordered:
            key = _strike_key(strike)
            self._points[key] = float(prob)
            updated.add(float(strike))
        self._strikes = sorted(updated)

    def probability_at(self, strike: float) -> float | None:
        if not self._points:
            return None
        key = _strike_key(strike)
        direct = self._points.get(key)
        if direct is not None:
            return direct
        if not self._strikes:
            return None
        idx = bisect.bisect_left(self._strikes, strike)
        if idx <= 0:
            idx = 0
        elif idx >= len(self._strikes):
            idx = len(self._strikes) - 1
        else:
            prev_strike = self._strikes[idx - 1]
            next_strike = self._strikes[idx]
            if abs(strike - prev_strike) <= abs(next_strike - strike):
                idx = idx - 1
        nearest_key = _strike_key(self._strikes[idx])
        return self._points.get(nearest_key)

    def joint_probability(self, strike_a: float, strike_b: float) -> float | None:
        """Return P(X >= max(strike_a, strike_b))."""
        if not self._points or not self._strikes:
            return None
        threshold = max(strike_a, strike_b)
        idx = bisect.bisect_left(self._strikes, threshold)
        if idx >= len(self._strikes):
            idx = len(self._strikes) - 1
        nearest_key = _strike_key(self._strikes[idx])
        return self._points.get(nearest_key)

    def snapshot(self) -> Mapping[str, float]:
        return dict(self._points)


@dataclass(frozen=True)
class CorrelationConfig:
    portfolio_limit: float
    z_score: float
    family_limits: dict[str, float]
    tilt_limits: dict[str, float]
    correlations: dict[str, dict[str, float]]
    series_family: dict[str, str]

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> CorrelationConfig:
        config_path = path or DEFAULT_CONFIG_PATH
        payload: Mapping[str, object] = {}
        if config_path.exists():
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        portfolio_limit = float(payload.get("portfolio_limit", 2500.0))
        z_score = max(float(payload.get("confidence_z", 2.0)), 1.0)
        family_section = payload.get("families") or {}
        family_limits: dict[str, float] = {}
        tilt_limits: dict[str, float] = {}
        series_family: dict[str, str] = {key: value for key, value in BASE_SERIES_FAMILY.items()}
        if isinstance(family_section, Mapping):
            for family_name, family_payload in family_section.items():
                normalized_family = _normalize_series(str(family_name))
                if not isinstance(family_payload, Mapping):
                    continue
                limit = family_payload.get("limit")
                tilt_limit = family_payload.get("tilt_limit", limit)
                if limit is not None:
                    try:
                        family_limits[normalized_family] = max(float(limit), 0.0)
                    except (TypeError, ValueError):
                        pass
                if tilt_limit is not None:
                    try:
                        tilt_limits[normalized_family] = max(float(tilt_limit), 0.0)
                    except (TypeError, ValueError):
                        pass
                for series in family_payload.get("series", []) or []:
                    series_family[_normalize_series(str(series))] = normalized_family
        correlation_section = payload.get("correlations") or {}
        correlations: dict[str, dict[str, float]] = defaultdict(dict)
        if isinstance(correlation_section, Mapping):
            for fam_a, row in correlation_section.items():
                if not isinstance(row, Mapping):
                    continue
                norm_a = _normalize_series(str(fam_a))
                for fam_b, value in row.items():
                    try:
                        correlations[norm_a][_normalize_series(str(fam_b))] = float(value)
                    except (TypeError, ValueError):
                        continue
        # Ensure diagonal defaults to 1.0
        families = set(series_family.values())
        for fam in families:
            correlations.setdefault(fam, {})
            correlations[fam].setdefault(fam, 1.0)
        return cls(
            portfolio_limit=max(portfolio_limit, 0.0),
            z_score=z_score,
            family_limits=family_limits,
            tilt_limits=tilt_limits,
            correlations={fam: dict(row) for fam, row in correlations.items()},
            series_family=series_family,
        )

    def family_for_series(self, series: str) -> str:
        return self.series_family.get(_normalize_series(series), _normalize_series(series))

    def correlation(self, series_a: str, series_b: str) -> float:
        fam_a = self.family_for_series(series_a)
        fam_b = self.family_for_series(series_b)
        return float(self.correlations.get(fam_a, {}).get(fam_b, 0.0))

    def family_limit(self, family: str) -> float | None:
        return self.family_limits.get(_normalize_series(family))

    def tilt_limit(self, family: str) -> float | None:
        return self.tilt_limits.get(_normalize_series(family))


@dataclass(frozen=True)
class InventoryExposure:
    series: str
    strike: float
    sign: int
    contracts: int
    probability: float

    def scaled(self, contracts: int) -> InventoryExposure:
        return replace(self, contracts=max(int(contracts), 0))


class CorrelationAwareLimiter:
    """Applies correlation-aware VaR caps and inventory tilt to ladder proposals."""

    def __init__(self, config: CorrelationConfig) -> None:
        self.config = config
        self._surfaces: dict[str, ProbabilitySurface] = {}
        self._exposures: list[InventoryExposure] = []
        self._family_net_sigma: defaultdict[str, float] = defaultdict(float)

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> CorrelationAwareLimiter:
        return cls(CorrelationConfig.from_yaml(path))

    # Surface / probability context -------------------------------------------------------------

    def update_surface(self, series: str, strikes: Sequence[float], survival: Sequence[float]) -> None:
        surface = self._surfaces.setdefault(_normalize_series(series), ProbabilitySurface())
        surface.update(strikes, survival)

    # Public API --------------------------------------------------------------------------------

    def cap_contracts(
        self,
        *,
        series: str,
        strike: float,
        side: OrderSide,
        contracts: int,
        probability: float | None,
    ) -> tuple[int, InventoryExposure | None, dict[str, float]]:
        """Return a contract cap enforced by correlation-aware VaR."""

        if contracts <= 0 or side not in (OrderSide.YES, OrderSide.NO):
            return 0, None, {}
        probability_value = self._normalize_probability(probability, series, strike)
        if probability_value is None:
            return contracts, None, {}
        sign = 1 if side is OrderSide.YES else -1
        exposure = InventoryExposure(
            series=_normalize_series(series),
            strike=float(strike),
            sign=sign,
            contracts=int(contracts),
            probability=probability_value,
        )
        tilt_factor = self._tilt_factor(exposure)
        capped_by_tilt = int(math.floor(exposure.contracts * tilt_factor))
        search_cap = min(exposure.contracts, max(capped_by_tilt, 0))
        if search_cap <= 0:
            return 0, None, {"tilt": float(tilt_factor)}
        allowed, portfolio_var, family_var = self._max_contracts_under_limits(exposure, search_cap)
        metadata = {
            "tilt": float(tilt_factor),
            "portfolio_var": portfolio_var,
            "portfolio_limit": self.config.portfolio_limit,
        }
        if family_var is not None:
            metadata["family_var"] = family_var
            metadata["family_limit"] = self.config.family_limit(self.config.family_for_series(series)) or 0.0
        if allowed <= 0:
            return 0, None, metadata
        return allowed, exposure.scaled(allowed), metadata

    def register(self, exposure: InventoryExposure) -> None:
        if exposure.contracts <= 0:
            return
        sigma_value = self._exposure_sigma(exposure)
        family = self.config.family_for_series(exposure.series)
        self._exposures.append(exposure)
        if sigma_value is not None:
            self._family_net_sigma[family] += exposure.sign * sigma_value

    def snapshot(self) -> dict[str, object]:
        portfolio_var, family_vars = self._var_metrics(self._exposures)
        return {
            "portfolio_var": portfolio_var,
            "portfolio_limit": self.config.portfolio_limit,
            "family_var": family_vars,
            "family_sigma": dict(self._family_net_sigma),
        }

    # Internal helpers -------------------------------------------------------------------------

    def _normalize_probability(self, probability: float | None, series: str, strike: float) -> float | None:
        if probability is not None:
            prob = float(probability)
            if 0.0 <= prob <= 1.0:
                return prob
        surface = self._surfaces.get(_normalize_series(series))
        if surface is None:
            return None
        return surface.probability_at(strike)

    def _tilt_factor(self, exposure: InventoryExposure) -> float:
        limit = self.config.tilt_limit(self.config.family_for_series(exposure.series))
        sigma = self._exposure_sigma(exposure)
        if sigma is None or limit is None or limit <= 0.0:
            return 1.0
        projected = self._family_net_sigma[self.config.family_for_series(exposure.series)] + exposure.sign * sigma
        imbalance = abs(projected) / max(limit, _EPSILON)
        if imbalance <= 0.25:
            return 1.0
        return max(0.0, 1.0 - imbalance)

    def _max_contracts_under_limits(
        self,
        exposure: InventoryExposure,
        cap: int,
    ) -> tuple[int, float, float | None]:
        if cap <= 0:
            return 0, 0.0, None
        lo = 0
        hi = cap
        best = 0
        best_portfolio = 0.0
        best_family = None
        while lo <= hi:
            mid = (lo + hi) // 2
            test_exposure = exposure.scaled(mid)
            ok, portfolio_var, family_var = self._within_limits(test_exposure)
            if ok:
                best = mid
                best_portfolio = portfolio_var
                best_family = family_var
                lo = mid + 1
            else:
                hi = mid - 1
        return best, best_portfolio, best_family

    def _within_limits(
        self,
        exposure: InventoryExposure | None,
    ) -> tuple[bool, float, float | None]:
        if exposure is None or exposure.contracts <= 0:
            exposures = list(self._exposures)
            family = None
        else:
            exposures = [*self._exposures, exposure]
            family = self.config.family_for_series(exposure.series)
        portfolio_var, family_vars = self._var_metrics(exposures)
        family_var = family_vars.get(family) if family is not None else None
        if self.config.portfolio_limit > 0.0 and portfolio_var > self.config.portfolio_limit + _EPSILON:
            return False, portfolio_var, family_var
        if family and family in self.config.family_limits:
            limit = self.config.family_limits[family]
            if limit > 0.0 and family_var is not None and family_var > limit + _EPSILON:
                return False, portfolio_var, family_var
        return True, portfolio_var, family_var

    def _var_metrics(self, exposures: Sequence[InventoryExposure]) -> tuple[float, dict[str, float]]:
        variance = self._variance_sum(exposures)
        portfolio_var = self.config.z_score * math.sqrt(max(variance, 0.0))
        family_vars: dict[str, float] = {}
        families = {_normalize_series(self.config.family_for_series(exp.series)) for exp in exposures}
        for family in families:
            if not family:
                continue
            family_exposures = [exp for exp in exposures if self.config.family_for_series(exp.series) == family]
            if not family_exposures:
                continue
            fam_variance = self._variance_sum(family_exposures)
            family_vars[family] = self.config.z_score * math.sqrt(max(fam_variance, 0.0))
        return portfolio_var, family_vars

    def _variance_sum(self, exposures: Sequence[InventoryExposure]) -> float:
        total = 0.0
        for idx, left in enumerate(exposures):
            total += self._covariance(left, left)
            for jdx in range(idx + 1, len(exposures)):
                cov = self._covariance(left, exposures[jdx])
                total += 2.0 * cov
        return max(total, 0.0)

    def _covariance(self, left: InventoryExposure, right: InventoryExposure) -> float:
        if left.contracts <= 0 or right.contracts <= 0:
            return 0.0
        if left.series == right.series:
            surface = self._surfaces.get(left.series)
            if surface is None:
                return 0.0
            joint_prob = surface.joint_probability(left.strike, right.strike)
            if joint_prob is None:
                return 0.0
            base = joint_prob - (left.probability * right.probability)
            return left.sign * right.sign * left.contracts * right.contracts * base
        sigma_left = self._exposure_sigma(left)
        sigma_right = self._exposure_sigma(right)
        if sigma_left is None or sigma_right is None:
            return 0.0
        rho = self.config.correlation(left.series, right.series)
        return left.sign * right.sign * rho * sigma_left * sigma_right

    def _exposure_sigma(self, exposure: InventoryExposure) -> float | None:
        prob = exposure.probability
        if prob is None:
            return None
        variance = prob * (1.0 - prob)
        if variance <= 0.0:
            return 0.0
        return float(exposure.contracts) * math.sqrt(max(variance, 0.0))
