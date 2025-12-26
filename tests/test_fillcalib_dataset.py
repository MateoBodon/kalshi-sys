from __future__ import annotations

from datetime import date
from pathlib import Path

from tools import build_fillcalib_dataset as fillcalib


def _load_payload(min_samples: int) -> dict[str, object]:
    payload, _rows = fillcalib.build_fillcalib_dataset(
        series="INXU",
        date_from=date(2025, 1, 1),
        date_to=date(2025, 1, 1),
        telemetry_root=Path("tests/fixtures/telemetry"),
        horizon_seconds=30,
        min_samples=min_samples,
        scaler=0.25,
        max_fill=0.25,
    )
    return payload


def _bucket_index(payload: dict[str, object]) -> dict[tuple[str, str], dict[str, object]]:
    series_block = payload["series"]["INXU"]
    buckets = series_block["buckets"]
    by_side = {}
    for bucket in buckets:
        key = (bucket["side"], bucket["quote_distance_to_touch_bin"])
        by_side[key] = bucket
    return by_side


def test_fillcalib_builds_curves_and_defaults() -> None:
    payload = _load_payload(min_samples=1)
    assert payload["version"] == 1
    assert payload["series"]["INXU"]["buckets"]

    buckets = _bucket_index(payload)
    yes_bucket = buckets[("YES", "0.00-0.01")]
    no_bucket = buckets[("NO", "0.05-0.10")]

    assert yes_bucket["proxy_fill_rate"] == 1.0
    assert yes_bucket["p_fill"] == 0.25
    assert no_bucket["proxy_fill_rate"] == 0.0
    assert no_bucket["p_fill"] == 0.0

    payload_insufficient = _load_payload(min_samples=2)
    for bucket in payload_insufficient["series"]["INXU"]["buckets"]:
        assert bucket["p_fill"] == 0.0
        assert bucket["reason"] == "insufficient_samples"
