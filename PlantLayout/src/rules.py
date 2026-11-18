"""Hard constraints and the four proposal soft-rule penalties."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage

from .geometry import GridGeometry
from .plant_data import PlantTaxon

MONTHS = {
    "Januar": 1,
    "Februar": 2,
    "März": 3,
    "April": 4,
    "Mai": 5,
    "Juni": 6,
    "Juli": 7,
    "August": 8,
    "September": 9,
    "Oktober": 10,
    "November": 11,
    "Dezember": 12,
}
SOCIAL_INTERVALS = {
    1: (1, 5),
    2: (3, 10),
    3: (10, 20),
}
SOCIAL_PATTERN = re.compile(r"Stufe\s+([123])")
_SOCIAL_CACHE: dict[int, tuple[tuple[int, int], ...]] = {}
_HEIGHT_CACHE: dict[int, float] = {}
_BLOOM_CACHE: dict[int, frozenset[int]] = {}


@dataclass(frozen=True)
class Layout:
    """Positions and taxon labels for one candidate planting plan."""

    positions_m: np.ndarray
    taxa: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.positions_m.ndim != 2 or self.positions_m.shape[1] != 2:
            raise ValueError("positions_m must have shape (n, 2)")
        if len(self.taxa) != len(self.positions_m):
            raise ValueError("positions_m and taxa must have equal length")


@dataclass(frozen=True)
class RuleWeights:
    grouping: float = 1.0
    height_structure: float = 1.0
    local_flowering: float = 1.0
    structural_plants: float = 1.0


@dataclass(frozen=True)
class EnergyBreakdown:
    grouping: float
    height_structure: float
    local_flowering: float
    structural_plants: float

    @property
    def total(self) -> float:
        return (
            self.grouping
            + self.height_structure
            + self.local_flowering
            + self.structural_plants
        )

    def weighted(self, weights: RuleWeights) -> float:
        return (
            self.grouping * weights.grouping
            + self.height_structure * weights.height_structure
            + self.local_flowering * weights.local_flowering
            + self.structural_plants * weights.structural_plants
        )


def hard_constraint_violations(
    layout: Layout,
    geometry: GridGeometry,
    spacing_m: Mapping[str, float],
) -> int:
    """Count edge, mask, and pair-spacing violations."""
    cell_coordinates = np.rint(layout.positions_m / geometry.cell_size_m).astype(int)
    rows, columns = cell_coordinates[:, 1], cell_coordinates[:, 0]
    in_bounds = (
        (rows >= 0)
        & (rows < geometry.mask.shape[0])
        & (columns >= 0)
        & (columns < geometry.mask.shape[1])
    )
    edge_violations = int(np.count_nonzero(~in_bounds))
    valid_rows = rows[in_bounds]
    valid_columns = columns[in_bounds]
    valid_indices = np.flatnonzero(in_bounds)
    plantable = np.zeros(len(layout.taxa), dtype=bool)
    plantable[valid_indices] = geometry.mask[valid_rows, valid_columns]
    clearances = np.zeros(len(layout.taxa))
    clearances[valid_indices] = geometry.clearance_m[valid_rows, valid_columns]
    radii = np.array([spacing_m[name] / 2 for name in layout.taxa])
    edge_violations += int(np.count_nonzero(~plantable | (clearances < radii)))

    if len(layout.taxa) < 2:
        return edge_violations
    distances = np.linalg.norm(
        layout.positions_m[:, None] - layout.positions_m[None, :], axis=2
    )
    minimums = (radii[:, None] + radii[None, :])
    upper_triangle = np.triu(np.ones(distances.shape, dtype=bool), k=1)
    return edge_violations + int(
        np.count_nonzero(upper_triangle & (distances < minimums))
    )


def grouping_penalty(
    layout: Layout,
    taxa_data: Mapping[str, PlantTaxon],
    spacing_m: Mapping[str, float],
) -> float:
    """Penalize same-taxon connected-group sizes outside sociability ranges."""
    penalties: list[float] = []
    for taxon in set(layout.taxa):
        indices = [i for i, name in enumerate(layout.taxa) if name == taxon]
        intervals = _sociability_intervals(taxa_data[taxon])
        if not intervals or len(indices) < 2:
            continue
        radius = 1.5 * spacing_m[taxon]
        adjacency = np.zeros((len(indices), len(indices)), dtype=bool)
        for i, index in enumerate(indices):
            for j, other in enumerate(indices[:i]):
                adjacency[i, j] = adjacency[j, i] = (
                    np.linalg.norm(layout.positions_m[index] - layout.positions_m[other])
                    <= radius
                )
        components, _ = ndimage.label(adjacency | np.eye(len(indices), dtype=bool))
        sizes = np.bincount(components.ravel())[1:]
        for size in sizes:
            penalties.append(_interval_deviation(float(size), intervals))
    return float(np.mean(penalties)) if penalties else 0.0


def height_structure_penalty(
    layout: Layout,
    taxa_data: Mapping[str, PlantTaxon],
    neighbourhood_m: float = 1.0,
) -> float:
    """Penalize locally uniform height where at least two heights are known."""
    heights = np.array([_height(taxa_data[name]) for name in layout.taxa])
    distances = np.linalg.norm(
        layout.positions_m[:, None] - layout.positions_m[None, :], axis=2
    )
    penalties: list[float] = []
    for index in range(len(layout.taxa)):
        if not np.isfinite(heights[index]):
            continue
        neighbours = distances[index] <= neighbourhood_m
        local = heights[neighbours & np.isfinite(heights)]
        if len(local) >= 2:
            coefficient = float(np.std(local) / np.mean(local)) if np.mean(local) else 0
            penalties.append(max(0.0, 1.0 - coefficient / 0.5))
    return float(np.mean(penalties)) if penalties else 0.0


def local_flowering_penalty(
    layout: Layout,
    taxa_data: Mapping[str, PlantTaxon],
    *,
    blocks_per_axis: int | None = None,
) -> float:
    """Measure the mean fraction of season months without local flowering."""
    if not len(layout.taxa):
        return 0.0
    axis = blocks_per_axis or max(1, round(math.sqrt(len(layout.taxa) / 10)))
    minimum = np.min(layout.positions_m, axis=0)
    maximum = np.max(layout.positions_m, axis=0)
    span = np.maximum(maximum - minimum, 1e-9)
    block_ids = np.floor((layout.positions_m - minimum) / span * axis).astype(int)
    block_ids = np.clip(block_ids, 0, axis - 1)
    penalties: list[float] = []
    for row in range(axis):
        for column in range(axis):
            selected = np.all(block_ids == (column, row), axis=1)
            if not np.any(selected):
                continue
            months = set().union(
                *(_bloom_months(taxa_data[name]) for name, keep in zip(layout.taxa, selected) if keep)
            )
            penalties.append((12 - len(months)) / 12)
    return float(np.mean(penalties)) if penalties else 0.0


def structural_plants_penalty(
    layout: Layout,
    structural_taxa: set[str],
    area_m2: float,
) -> float:
    """Compare structural nearest-neighbour spacing with a random expectation."""
    indices = [i for i, name in enumerate(layout.taxa) if name in structural_taxa]
    if len(indices) < 2:
        return 0.0
    positions = layout.positions_m[indices]
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
    distances[distances == 0] = np.inf
    observed = float(np.mean(np.min(distances, axis=1)))
    expected = 0.5 * math.sqrt(area_m2 / len(indices))
    return max(0.0, 1.0 - observed / expected) if expected else 0.0


def energy(
    layout: Layout,
    geometry: GridGeometry,
    taxa_data: Mapping[str, PlantTaxon],
    spacing_m: Mapping[str, float],
    structural_taxa: set[str],
) -> EnergyBreakdown:
    """Return all four soft penalties; hard violations are reported separately."""
    if hard_constraint_violations(layout, geometry, spacing_m):
        raise ValueError("cannot score a layout that violates hard constraints")
    return EnergyBreakdown(
        grouping_penalty(layout, taxa_data, spacing_m),
        height_structure_penalty(layout, taxa_data),
        local_flowering_penalty(layout, taxa_data),
        structural_plants_penalty(
            layout, structural_taxa, float(np.count_nonzero(geometry.mask)) * geometry.cell_size_m**2
        ),
    )


def _sociability_intervals(taxon: PlantTaxon) -> tuple[tuple[int, int], ...]:
    cached = _SOCIAL_CACHE.get(id(taxon))
    if cached is not None:
        return cached
    intervals: list[tuple[int, int]] = []
    for value in taxon.values("Geselligkeitsstufe"):
        match = SOCIAL_PATTERN.search(value)
        if match:
            interval = SOCIAL_INTERVALS[int(match.group(1))]
            if interval not in intervals:
                intervals.append(interval)
    result = tuple(intervals)
    _SOCIAL_CACHE[id(taxon)] = result
    return result


def _interval_deviation(size: float, intervals: Sequence[tuple[int, int]]) -> float:
    if any(low <= size <= high for low, high in intervals):
        return 0.0
    distance = min(min(abs(size - low), abs(size - high)) for low, high in intervals)
    scale = max(high for _, high in intervals)
    return min(1.0, distance / scale)


def _height(taxon: PlantTaxon) -> float:
    cached = _HEIGHT_CACHE.get(id(taxon))
    if cached is not None:
        return cached
    values = taxon.numeric_values("Blüten-Hoehe_v") + taxon.numeric_values("Blüten_Hoehe_b")
    result = float(np.mean(values)) if values else float("nan")
    _HEIGHT_CACHE[id(taxon)] = result
    return result


def _bloom_months(taxon: PlantTaxon) -> frozenset[int]:
    cached = _BLOOM_CACHE.get(id(taxon))
    if cached is not None:
        return cached
    starts = [MONTHS[value] for value in taxon.values("Blütezeit von") if value in MONTHS]
    ends = [MONTHS[value] for value in taxon.values("Blütezeit bis") if value in MONTHS]
    if not starts or not ends:
        result = frozenset()
        _BLOOM_CACHE[id(taxon)] = result
        return result
    start, end = min(starts), max(ends)
    result = frozenset(range(start, end + 1))
    _BLOOM_CACHE[id(taxon)] = result
    return result
