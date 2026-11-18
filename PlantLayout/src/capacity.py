"""Plant-count allocation and capacity-screening utilities."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from .plant_data import PlantTaxon


@dataclass(frozen=True)
class PackingScreen:
    """Necessary, but not sufficient, area estimate for a plant list."""

    required_area_m2: float
    available_area_m2: float

    @property
    def passes(self) -> bool:
        return self.required_area_m2 <= self.available_area_m2


@dataclass(frozen=True)
class CapacitySearchResult:
    """Recorded result of a bounded capacity search."""

    capacity: int
    checks: tuple[tuple[int, bool], ...]


def spacing_map(
    taxa: Mapping[str, PlantTaxon],
    *,
    fallbacks_cm: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Resolve lower planting distances to metres using explicit fallbacks."""
    fallbacks_cm = fallbacks_cm or {}
    resolved: dict[str, float] = {}
    for name, taxon in taxa.items():
        values = taxon.numeric_values("Pflanzabstand von")
        if values:
            resolved[name] = min(values) / 100
        elif name in fallbacks_cm:
            fallback = fallbacks_cm[name]
            if fallback <= 0:
                raise ValueError(f"fallback spacing for {name!r} must be positive")
            resolved[name] = fallback / 100
        else:
            raise ValueError(f"missing spacing and fallback for {name!r}")
    return resolved


def largest_remainder_counts(
    proportions: Mapping[str, float],
    total: int,
) -> dict[str, int]:
    """Allocate ``total`` integer plants with minimum proportion deviation."""
    if total < 0:
        raise ValueError("total must not be negative")
    if not proportions:
        if total:
            raise ValueError("cannot allocate a positive total without taxa")
        return {}
    if any(value < 0 for value in proportions.values()):
        raise ValueError("proportions must not be negative")
    proportion_sum = sum(proportions.values())
    if not math.isclose(proportion_sum, 1.0, abs_tol=1e-9):
        raise ValueError("proportions must sum to 1")

    exact = {name: value * total for name, value in proportions.items()}
    counts = {name: math.floor(value) for name, value in exact.items()}
    remaining = total - sum(counts.values())
    order = sorted(
        proportions,
        key=lambda name: (exact[name] - counts[name], name),
        reverse=True,
    )
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def counts_for_total(
    supplied_counts: Mapping[str, int],
    total: int,
) -> dict[str, int]:
    """Preserve supplied frequencies using the largest-remainder method."""
    if not supplied_counts or any(value < 0 for value in supplied_counts.values()):
        raise ValueError("supplied counts must be non-negative and non-empty")
    supplied_total = sum(supplied_counts.values())
    if supplied_total <= 0:
        raise ValueError("supplied counts must have a positive total")
    proportions = {
        name: count / supplied_total for name, count in supplied_counts.items()
    }
    return largest_remainder_counts(proportions, total)


def disc_area_screen(
    taxa: Mapping[str, PlantTaxon],
    counts: Mapping[str, int],
    *,
    spacing_trait: str = "Pflanzabstand von",
    spacing_unit_to_m: float = 0.01,
    available_area_m2: float,
) -> PackingScreen:
    """Estimate required area from non-overlapping discs.

    Missing spacing is an error because this screen cannot infer a fallback.
    """
    if available_area_m2 < 0:
        raise ValueError("available_area_m2 must not be negative")
    required = 0.0
    for name, count in counts.items():
        if count < 0:
            raise ValueError("plant counts must not be negative")
        try:
            spacing_values = taxa[name].numeric_values(spacing_trait)
        except KeyError as exc:
            raise ValueError(f"unknown taxon: {name}") from exc
        if not spacing_values:
            raise ValueError(f"missing spacing for {name!r}")
        spacing_m = min(spacing_values) * spacing_unit_to_m
        required += count * np.pi * (spacing_m / 2) ** 2
    return PackingScreen(required, available_area_m2)


def search_capacity(
    *,
    maximum: int,
    feasible: Callable[[int], bool],
    verify_neighbours: bool = True,
) -> CapacitySearchResult:
    """Find a feasible capacity by bounded search and local verification.

    The binary phase assumes feasibility is monotone. The returned checks
    allow callers to report this assumption and inspect adjacent counts.
    """
    if maximum < 0:
        raise ValueError("maximum must not be negative")
    checks: list[tuple[int, bool]] = []

    def check(total: int) -> bool:
        result = feasible(total)
        checks.append((total, result))
        return result

    if not check(0):
        raise ValueError("zero plants must be feasible")
    if check(maximum):
        capacity = maximum
    else:
        low, high = 0, maximum
        while high - low > 1:
            middle = (low + high) // 2
            if check(middle):
                low = middle
            else:
                high = middle
        capacity = low

    if verify_neighbours:
        for candidate in (capacity - 1, capacity + 1):
            if 0 <= candidate <= maximum and candidate not in dict(checks):
                check(candidate)
    return CapacitySearchResult(capacity, tuple(checks))
