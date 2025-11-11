from __future__ import annotations

from kalshi_alpha.risk import var_index


def test_family_var_limiter_enforces_limits() -> None:
    limiter = var_index.FamilyVarLimiter({"SPX": 100.0})
    assert limiter.can_accept("INXU", 60.0)
    limiter.register("INXU", 60.0)
    assert not limiter.can_accept("INX", 50.0)
    assert limiter.can_accept("NASDAQ100", 80.0)
