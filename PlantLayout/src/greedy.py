"""Greedy construction of feasible planting layouts."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .geometry import GridGeometry
from .plant_data import PlantTaxon
from .rules import Layout, RuleWeights, energy, hard_constraint_violations


@dataclass(frozen=True)
class GreedyResult:
    """Output and accounting for one greedy insertion run."""

    layout: Layout
    feasible: bool
    energy: float
    evaluations: int
    failed_taxon: str | None = None


def greedy_insert(
    geometry: GridGeometry,
    taxa_data: Mapping[str, PlantTaxon],
    counts: Mapping[str, int],
    spacing_m: Mapping[str, float],
    structural_taxa: set[str],
    *,
    seed: int,
    weights: RuleWeights = RuleWeights(),
    order: str = "constrained",
) -> GreedyResult:
    """Insert plants into the lowest-energy admissible grid cells.

    ``order="constrained"`` ranks taxa by the number of admissible cells,
    spacing, then sociability restriction. ``order="random"`` is the H3
    comparison baseline.
    """
    if order not in {"constrained", "random"}:
        raise ValueError("order must be 'constrained' or 'random'")
    if any(count < 0 for count in counts.values()):
        raise ValueError("counts must not be negative")
    rng = random.Random(seed)
    remaining = [name for name, count in counts.items() for _ in range(count)]
    if order == "random":
        rng.shuffle(remaining)
    else:
        remaining.sort(
            key=lambda name: (
                int(np.count_nonzero(geometry.admissible_cells(spacing_m[name] / 2))),
                -spacing_m[name],
                _sociability_rank(taxa_data[name]),
                rng.random(),
            )
        )

    positions: list[tuple[float, float]] = []
    placed_taxa: list[str] = []
    evaluations = 0
    for taxon in remaining:
        candidates = _candidate_positions(
            geometry, positions, placed_taxa, taxon, spacing_m
        )
        if not candidates:
            layout = Layout(np.asarray(positions, dtype=float).reshape(-1, 2), tuple(placed_taxa))
            return GreedyResult(layout, False, float("inf"), evaluations, taxon)

        best_energy = float("inf")
        best_positions: list[tuple[float, float]] = []
        for candidate in candidates:
            trial = Layout(
                np.asarray([*positions, candidate], dtype=float),
                tuple([*placed_taxa, taxon]),
            )
            if hard_constraint_violations(trial, geometry, spacing_m):
                continue
            breakdown = energy(
                trial, geometry, taxa_data, spacing_m, structural_taxa
            )
            candidate_energy = breakdown.weighted(weights)
            evaluations += 1
            if candidate_energy < best_energy - 1e-12:
                best_energy = candidate_energy
                best_positions = [candidate]
            elif np.isclose(candidate_energy, best_energy):
                best_positions.append(candidate)
        if not best_positions:
            layout = Layout(np.asarray(positions, dtype=float).reshape(-1, 2), tuple(placed_taxa))
            return GreedyResult(layout, False, float("inf"), evaluations, taxon)
        position = rng.choice(best_positions)
        positions.append(position)
        placed_taxa.append(taxon)

    final_layout = Layout(np.asarray(positions, dtype=float).reshape(-1, 2), tuple(placed_taxa))
    final_energy = energy(
        final_layout, geometry, taxa_data, spacing_m, structural_taxa
    ).weighted(weights)
    return GreedyResult(final_layout, True, final_energy, evaluations)


def _candidate_positions(
    geometry: GridGeometry,
    positions: list[tuple[float, float]],
    placed_taxa: list[str],
    taxon: str,
    spacing_m: Mapping[str, float],
) -> list[tuple[float, float]]:
    mask = geometry.admissible_cells(spacing_m[taxon] / 2)
    coordinates = np.argwhere(mask)
    candidates: list[tuple[float, float]] = []
    for row, column in coordinates:
        position = (column * geometry.cell_size_m, row * geometry.cell_size_m)
        valid = all(
            np.linalg.norm(np.asarray(position) - np.asarray(existing))
            >= (spacing_m[taxon] + spacing_m[other]) / 2
            for existing, other in zip(positions, placed_taxa)
        )
        if valid:
            candidates.append(position)
    return candidates


def _sociability_rank(taxon: PlantTaxon) -> int:
    values = taxon.values("Geselligkeitsstufe")
    roman = {"I": 1, "II": 2, "III": 3}
    return min(
        (
            roman.get(value.split("Stufe ", 1)[1].split(" ", 1)[0], 4)
            for value in values
        ),
        default=4,
    )
