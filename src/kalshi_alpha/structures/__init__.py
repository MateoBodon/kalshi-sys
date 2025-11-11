"""Structure-level utilities (allocators, range builders, hedges)."""

from .allocator import (
    AllocationResult,
    Allocator,
    AllocatorConfig,
    SeriesBudget,
    SeriesWindowSample,
    VarSnapshot,
    correlation_var_snapshot,
    load_scoreboard_history,
)
from .range_ab import RangeABStructure, StructureLeg, build_range_structures

__all__ = [
    "AllocationResult",
    "Allocator",
    "AllocatorConfig",
    "SeriesBudget",
    "SeriesWindowSample",
    "VarSnapshot",
    "correlation_var_snapshot",
    "load_scoreboard_history",
    "RangeABStructure",
    "StructureLeg",
    "build_range_structures",
]
