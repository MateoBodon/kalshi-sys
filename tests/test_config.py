from __future__ import annotations

import yaml


def test_series_registry_includes_fixture_series() -> None:
    with open("configs/series.yaml", encoding="utf-8") as handle:
        registry = yaml.safe_load(handle)
    names = registry["series"].keys()
    for required in ("CPI", "CLAIMS", "TENY", "WEATHER"):
        assert required in names
        entry = registry["series"][required]
        assert entry.get("settlement_source")
