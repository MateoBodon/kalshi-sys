from __future__ import annotations

import polars as pl

from kalshi_alpha.drivers import treasury_yields


def test_treasury_reconciliation(offline_fixtures_root):
    fixtures = offline_fixtures_root / "treasury_yields"
    yields = treasury_yields.fetch_daily_yields(offline=True, fixtures_dir=fixtures)
    dgs = pl.read_csv(fixtures / "dgs10.csv").with_columns(
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False)
    )
    dgs_map = {row["date"]: float(row["rate"]) for row in dgs.iter_rows(named=True)}

    tolerance_bp = 5.0
    matched = False
    for entry in yields:
        if entry.maturity.upper() == "DGS10":
            as_of = entry.as_of.date()
            if as_of in dgs_map:
                matched = True
                diff = abs(entry.rate - dgs_map[as_of]) * 100
                assert diff <= tolerance_bp
    assert matched, "Expected overlapping dates for reconciliation"
