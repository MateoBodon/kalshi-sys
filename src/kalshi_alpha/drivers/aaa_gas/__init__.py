"""AAA gasoline price driver stub (fixture backed)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GasPriceSummary:
    national_average: float
    state_averages: dict[str, float]


def load_summary(*, offline_path: Path | None = None) -> GasPriceSummary:
    if offline_path is None:
        raise RuntimeError("AAA gas integration requires offline fixtures for now")
    data = json.loads(offline_path.read_text(encoding="utf-8"))
    return GasPriceSummary(
        national_average=float(data["national_average"]),
        state_averages={state: float(value) for state, value in data["state_averages"].items()},
    )
