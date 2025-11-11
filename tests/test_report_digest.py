from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from report import digest


def _write_ledger(path: Path) -> None:
    frame = pl.DataFrame(
        {
            "series": ["INX", "NASDAQ100"],
            "ev_after_fees": [12.5, 8.1],
            "pnl_simulated": [11.2, 6.4],
            "ev_realized_bps": [140.0, 95.0],
            "ev_expected_bps": [120.0, 90.0],
            "fill_ratio_observed": [0.62, 0.58],
            "alpha_target": [0.6, 0.57],
            "slippage_ticks": [0.5, 0.4],
            "timestamp_et": [
                datetime(2025, 11, 10, 14, 5, tzinfo=UTC),
                datetime(2025, 11, 10, 15, 15, tzinfo=UTC),
            ],
        }
    )
    frame.write_parquet(path)


def test_digest_writes_markdown_and_plot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger_path = tmp_path / "ledger.parquet"
    _write_ledger(ledger_path)
    reports_dir = tmp_path / "reports"
    (reports_dir / "INX").mkdir(parents=True, exist_ok=True)
    (reports_dir / "NASDAQ100").mkdir(parents=True, exist_ok=True)
    (reports_dir / "_artifacts" / "monitors").mkdir(parents=True, exist_ok=True)
    (reports_dir / "_artifacts" / "monitors" / "freshness.json").write_text(
        """
        {
          "name": "freshness",
          "status": "OK",
          "generated_at": "2025-11-10T12:00:00Z",
          "metrics": {}
        }
        """.strip(),
        encoding="utf-8",
    )
    report_body = "\n".join(
        [
            "# Report",
            "- Portfolio VaR: 5.0 USD",
            "## Monitors",
            "- non_monotone: 0",
        ]
    )
    for series in ("INX", "NASDAQ100"):
        (reports_dir / series / "2025-11-10.md").write_text(report_body, encoding="utf-8")
    replay_dir = reports_dir / "_artifacts" / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    replay_summary = {
        "epsilon": 0.15,
        "window_type_max": {"close": 0.12},
        "windows": [
            {
                "window_label": "INX 2025-11-10 16:00",
                "window_type": "close",
                "max_abs_delta": 0.12,
                "threshold_breach": False,
            }
        ],
        "worst_bins": [
            {"market_ticker": "KXINX-TEST", "strike": 5000.0, "delta_per_contract": -0.12},
        ],
    }
    (replay_dir / "replay_summary_2025-11-10.json").write_text(json.dumps(replay_summary), encoding="utf-8")

    output_dir = tmp_path / "digests"
    args = [
        "--date",
        "2025-11-10",
        "--ledger",
        str(ledger_path),
        "--reports",
        str(reports_dir),
        "--output",
        str(output_dir),
    ]
    digest.main(args)

    markdown = output_dir / "digest_2025-11-10.md"
    plot = output_dir / "digest_2025-11-10.png"
    assert markdown.exists(), "digest markdown missing"
    assert plot.exists(), "digest plot missing"
    contents = markdown.read_text(encoding="utf-8")
    assert "Daily Digest" in contents
    assert "INX" in contents
    assert "NASDAQ100" in contents
    assert "Replay Parity" in contents
    assert "INX 2025-11-10 16:00" in contents
    assert "Monitor Status" in contents
