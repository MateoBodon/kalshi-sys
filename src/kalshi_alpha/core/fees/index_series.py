"""Index series fee curve loader."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INDEX_FEE_PATHS: tuple[Path, ...] = (
    ROOT / "configs" / "fees.json",
    ROOT / "configs" / "fees" / "series.json",
)
DEFAULT_INDEX_FEE_PATH = DEFAULT_INDEX_FEE_PATHS[0]
LEGACY_INDEX_FEE_PATH = ROOT / "data" / "reference" / "index_fee_curves.json"


@dataclass(frozen=True)
class IndexFeeCurve:
    """Parametric fee curve for a specific index ladder series."""

    series: str
    coefficient: Decimal


def load_index_fee_curves(path: Path | None = None) -> Mapping[str, IndexFeeCurve]:
    """Load all index fee curves from the reference JSON file."""

    if path is not None:
        resolved = Path(path).resolve()
    else:
        resolved = None
        for candidate in DEFAULT_INDEX_FEE_PATHS:
            if candidate.exists():
                resolved = candidate.resolve()
                break
        if resolved is None and LEGACY_INDEX_FEE_PATH.exists():
            resolved = LEGACY_INDEX_FEE_PATH.resolve()
        if resolved is None:
            raise FileNotFoundError(
                "Index fee configuration not found in configs/fees.json, "
                "configs/fees/series.json, or data/reference/index_fee_curves.json",
            )
    return _load_index_fee_curves_cached(str(resolved))


def get_index_fee_curve(series: str, path: Path | None = None) -> IndexFeeCurve | None:
    """Return the fee curve for *series*, or ``None`` if not configured."""

    curves = load_index_fee_curves(path)
    return curves.get(series.upper())


@lru_cache(maxsize=4)
def _load_index_fee_curves_cached(resolved_path: str) -> dict[str, IndexFeeCurve]:
    path = Path(resolved_path)
    if not path.exists():
        raise FileNotFoundError(f"Index fee configuration not found at {resolved_path}")
    data = _load_with_extends(path)
    series_section = data.get("series", {})
    if not isinstance(series_section, dict):
        raise ValueError("index fee configuration must include a 'series' mapping")
    curves: dict[str, IndexFeeCurve] = {}
    for key, value in series_section.items():
        try:
            coefficient = Decimal(str(value["coefficient"]))
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"missing coefficient for index series {key}") from exc
        curves[key.upper()] = IndexFeeCurve(series=key.upper(), coefficient=coefficient)
    if not curves:
        raise ValueError("index fee configuration contained no series entries")
    return curves


def _load_with_extends(path: Path, *, _visited: frozenset[Path] | None = None) -> dict[str, object]:
    visited = set(_visited or ())
    resolved = path.resolve()
    if resolved in visited:
        chain = " -> ".join(p.as_posix() for p in (*visited, resolved))
        raise ValueError(f"Detected circular fee config extends chain: {chain}")
    visited.add(resolved)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    extends = payload.get("extends")
    if isinstance(extends, str):
        base_path = (resolved.parent / extends).resolve()
        base_payload = _load_with_extends(base_path, _visited=frozenset(visited))
        merged = dict(base_payload)
        merged.update({k: v for k, v in payload.items() if k != "extends"})
        return merged
    return payload


__all__ = ["IndexFeeCurve", "load_index_fee_curves", "get_index_fee_curve"]
