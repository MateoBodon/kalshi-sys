from pathlib import Path

import polars as pl

from scripts.build_index_panel_polygon import build_panel
from kalshi_alpha.exec.scanners.utils import pmf_to_survival
from kalshi_alpha.strategies.index.model_polygon import fit_from_panel, predict_pmf


FIXTURE_ROOT = Path("tests/data_fixtures/index_panel_fast/raw/polygon/index")


def _panel():
    return build_panel(
        symbols=("I:SPX", "I:NDX"),
        input_root=FIXTURE_ROOT,
        output_path=None,
    )


def test_fit_and_predict_noon():
    panel = _panel()
    params = fit_from_panel(panel, horizon="noon")
    sample = panel.filter(pl.col("symbol") == "I:SPX").sort("timestamp_et").head(1)
    price_now = float(sample.item(0, "price"))
    minutes = float(sample.item(0, "minutes_to_noon"))
    strikes = [price_now - 25, price_now, price_now + 25]
    pmf = predict_pmf(
        {"symbol": "I:SPX", "price": price_now, "minutes_to_target": minutes},
        params,
        strikes,
    )
    total_prob = sum(bin_prob.probability for bin_prob in pmf)
    assert abs(total_prob - 1.0) < 1e-6
    assert all(bin_prob.probability >= 0 for bin_prob in pmf)


def test_probabilities_shift_with_price():
    panel = _panel()
    params = fit_from_panel(panel, horizon="close")
    sample = panel.filter(pl.col("symbol") == "I:NDX").sort("timestamp_et").head(1)
    base_price = float(sample.item(0, "price"))
    minutes = float(sample.item(0, "minutes_to_close"))
    strikes = [base_price - 50, base_price, base_price + 50]

    pmf_low = predict_pmf(
        {"symbol": "I:NDX", "price": base_price - 25, "minutes_to_target": minutes},
        params,
        strikes,
    )
    pmf_high = predict_pmf(
        {"symbol": "I:NDX", "price": base_price + 25, "minutes_to_target": minutes},
        params,
        strikes,
    )
    survival_low = pmf_to_survival(pmf_low, strikes)
    survival_high = pmf_to_survival(pmf_high, strikes)
    # Higher starting price should increase survival at the top strike.
    assert survival_high[-1] > survival_low[-1]
