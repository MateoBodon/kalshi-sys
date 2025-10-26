from __future__ import annotations

import json
from pathlib import Path

from kalshi_alpha.core.backtest import crps_from_pmf
from kalshi_alpha.core.pricing import LadderBinProbability
from kalshi_alpha.drivers.nws_cli import parse_multi_station_report
from kalshi_alpha.strategies import base
from kalshi_alpha.strategies.claims import ClaimsInputs
from kalshi_alpha.strategies.claims import calibrate as claims_calibrate
from kalshi_alpha.strategies.claims import pmf as claims_pmf
from kalshi_alpha.strategies.cpi import CPIInputs
from kalshi_alpha.strategies.cpi import calibrate as cpi_calibrate
from kalshi_alpha.strategies.cpi import nowcast as cpi_nowcast
from kalshi_alpha.strategies.teny import TenYInputs
from kalshi_alpha.strategies.teny import calibrate as teny_calibrate
from kalshi_alpha.strategies.teny import pmf as teny_pmf
from kalshi_alpha.strategies.weather import WeatherInputs
from kalshi_alpha.strategies.weather import calibrate as weather_calibrate
from kalshi_alpha.strategies.weather import pmf as weather_pmf

FIXTURE_ROOT = Path(__file__).parent / "fixtures"


def _brier_score(pmf: list[LadderBinProbability], actual: float) -> float:
    probabilities = [bin_prob.probability for bin_prob in pmf]
    indicators = [_bin_indicator(bin_prob, actual) for bin_prob in pmf]
    return sum(
        (prob - indicator) ** 2 for prob, indicator in zip(probabilities, indicators, strict=True)
    )


def _bin_indicator(bin_prob: LadderBinProbability, value: float) -> float:
    lower = float("-inf") if bin_prob.lower is None else bin_prob.lower
    upper = float("inf") if bin_prob.upper is None else bin_prob.upper
    return 1.0 if lower <= value < upper else 0.0


def test_cpi_nowcast_crps_beats_baseline() -> None:
    data = json.loads((FIXTURE_ROOT / "cpi" / "history.json").read_text(encoding="utf-8"))
    history = data["history"]
    cpi_calibrate(history)
    baseline_scores: list[float] = []
    model_scores: list[float] = []
    previous_actual: float | None = None
    for entry in history:
        inputs = CPIInputs(
            cleveland_nowcast=entry["cleveland_nowcast"],
            latest_release_mom=previous_actual,
            aaa_delta=entry["aaa_delta"],
        )
        distribution = cpi_nowcast(inputs)
        pmf = base.grid_distribution_to_pmf(distribution)
        if previous_actual is not None:
            model_scores.append(crps_from_pmf(pmf, entry["actual"]))
            baseline_distribution = base.pmf_from_gaussian(
                [round(previous_actual + 0.05 * i, 2) for i in range(-5, 6)],
                mean=previous_actual,
                std=0.25,
            )
            baseline_scores.append(crps_from_pmf(baseline_distribution, entry["actual"]))
        previous_actual = entry["actual"]
    assert model_scores and baseline_scores
    assert sum(model_scores) / len(model_scores) < sum(baseline_scores) / len(baseline_scores)


def test_claims_model_beats_naive_brier() -> None:
    data = json.loads((FIXTURE_ROOT / "claims" / "history.json").read_text(encoding="utf-8"))
    history = data["history"]
    claims_calibrate(history)
    strikes = [200_000, 205_000, 210_000, 215_000, 220_000, 225_000]
    observed: list[int] = []
    model_scores: list[float] = []
    baseline_scores: list[float] = []
    for idx, entry in enumerate(history):
        observed.append(entry["claims"])
        if idx == 0:
            continue
        inputs = ClaimsInputs(
            history=observed[:-1],
            holiday_next=entry["holiday"],
            freeze_active=False,
        )
        pmf = claims_pmf(strikes, inputs=inputs)
        model_scores.append(_brier_score(pmf, entry["claims"]))
        baseline_inputs = ClaimsInputs(latest_initial_claims=observed[-2])
        baseline_pmf = claims_pmf(strikes, inputs=baseline_inputs)
        baseline_scores.append(_brier_score(baseline_pmf, entry["claims"]))
    assert model_scores and baseline_scores
    assert sum(model_scores) / len(model_scores) < sum(baseline_scores) / len(baseline_scores)


def _brier_score_from_probabilities(
    pmf: list[LadderBinProbability],
    probabilities: list[float],
    actual: float,
) -> float:
    indicators = [_bin_indicator(bin_prob, actual) for bin_prob in pmf]
    return sum(
        (prob - indicator) ** 2 for prob, indicator in zip(probabilities, indicators, strict=True)
    )


def test_teny_crps_factor_model() -> None:
    data = json.loads((FIXTURE_ROOT / "teny" / "history.json").read_text(encoding="utf-8"))
    history = data["history"]
    strikes = [4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9]
    teny_calibrate(history)
    closes: list[float] = []
    model_scores: list[float] = []
    baseline_scores: list[float] = []
    for idx, entry in enumerate(history):
        closes.append(entry["actual_close"])
        if idx == 0:
            continue
        prior_close = history[idx - 1]["actual_close"]
        inputs = TenYInputs(
            prior_close=prior_close,
            macro_shock=entry["macro_shock"],
            trailing_history=[record["actual_close"] for record in history[:idx]],
        )
        pmf = teny_pmf(strikes, inputs=inputs)
        model_scores.append(crps_from_pmf(pmf, entry["actual_close"]))
        baseline_distribution = {
            round(prior_close - 0.1, 2): 0.25,
            round(prior_close, 2): 0.5,
            round(prior_close + 0.1, 2): 0.25,
        }
        baseline_pmf = base.grid_distribution_to_pmf(baseline_distribution)
        baseline_scores.append(crps_from_pmf(baseline_pmf, entry["actual_close"]))
    assert model_scores and baseline_scores
    assert sum(model_scores) / len(model_scores) < sum(baseline_scores) / len(baseline_scores)


def test_weather_bias_spread_distribution_and_parser() -> None:
    data = json.loads((FIXTURE_ROOT / "weather" / "history.json").read_text(encoding="utf-8"))
    history = data["history"]
    strikes = [float(value) for value in range(40, 101, 5)]
    weather_calibrate(history)
    model_scores: list[float] = []
    baseline_scores: list[float] = []
    for entry in history:
        inputs = WeatherInputs(
            forecast_high=entry["forecast_high"],
            bias=entry["bias"],
            spread=entry["spread"],
            station=entry["station"],
        )
        pmf = weather_pmf(strikes, inputs=inputs)
        model_scores.append(crps_from_pmf(pmf, entry["actual_high"]))
        baseline_distribution = {
            round(entry["forecast_high"] - 6, 2): 0.2,
            round(entry["forecast_high"], 2): 0.6,
            round(entry["forecast_high"] + 6, 2): 0.2,
        }
        baseline_pmf = base.grid_distribution_to_pmf(baseline_distribution)
        baseline_scores.append(crps_from_pmf(baseline_pmf, entry["actual_high"]))
    assert sum(model_scores) / len(model_scores) < sum(baseline_scores) / len(baseline_scores)

    records = parse_multi_station_report(FIXTURE_ROOT / "weather" / "nws_dcr_multi.txt")
    assert len(records) == 3
    assert {record.station_id for record in records} == {"KBOS", "KATL", "KSEA"}
