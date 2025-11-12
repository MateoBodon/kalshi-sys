from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from report import digest


def _write_ledger(path: Path) -> None:
    frame = pl.DataFrame(
        {
            "series": ["SPX", "NDX"],
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


@pytest.mark.parametrize("engine", ["polars", "pandas"])
def test_digest_writes_markdown_and_plot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    engine: str,
) -> None:
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

    output_dir = tmp_path / "digests"
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    args = [
        "--date",
        "2025-11-10",
        "--ledger",
        str(ledger_path),
        "--reports",
        str(reports_dir),
        "--raw",
        str(raw_dir),
        "--output",
        str(output_dir),
        "--engine",
        engine,
        "--skip-slo",
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
    assert "Monitor Status" in contents
