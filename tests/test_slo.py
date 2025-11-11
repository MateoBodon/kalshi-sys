from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from kalshi_alpha.exec import slo


def test_collect_metrics_aggregates_files(tmp_path: Path) -> None:
    now = datetime(2025, 11, 11, 15, 30, tzinfo=UTC)
    reports_root = tmp_path / "reports"
    raw_root = tmp_path / "data" / "raw"
    summary = [
        {
            "series": "INX",
            "honesty_brier": 0.0123,
            "ev_delta_mean": -5.5,
            "fill_ratio_vs_alpha": 0.01,
            "go_reasons": ["fills 10 < 50"],
        }
    ]

    # Snapshot latencies: create two files with known ingestion/quote delta
    day_dir = raw_root / f"{now.year:04d}" / f"{now.month:02d}" / f"{now.day:02d}" / "polygon_index"
    day_dir.mkdir(parents=True, exist_ok=True)
    for offset_ms in (100, 400):
        ingest = now.strftime("%Y%m%dT%H%M%S")
        ts = (now - timedelta(milliseconds=offset_ms)).isoformat()
        path = day_dir / f"{ingest}_INX_snapshot.json"
        path.write_text(json.dumps({"timestamp": ts}), encoding="utf-8")
        now += timedelta(seconds=1)

    # Report with ops_seconds_to_cancel + Portfolio VaR
    report_dir = reports_root / "INX"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_dir.joinpath("2025-11-11.md").write_text(
        "\n".join(
            [
                "- ops_seconds_to_cancel: 300.0",
                "- Portfolio VaR: 12.5 USD",
            ]
        ),
        encoding="utf-8",
    )

    metrics = slo.collect_metrics(
        summary,
        reports_root=reports_root,
        raw_root=raw_root,
        lookback_days=3,
        now=datetime(2025, 11, 11, 18, 0, tzinfo=UTC),
        var_limits={"SPX": 100.0},
    )
    entry = metrics["INX"]
    assert entry.freshness_p95_ms is not None and entry.freshness_p95_ms >= 100.0
    assert entry.time_at_risk_p95_s == 300.0
    assert entry.var_headroom_usd == 87.5
    assert entry.fill_gap_pp == 1.0
    assert entry.ev_gap_bps == -5.5
    assert entry.no_go_reasons == ["fills 10 < 50"]
