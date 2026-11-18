"""Simulated Annealing improvement of a feasible greedy layout."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .geometry import GridGeometry
from .evaluator import EnergyEvaluator
from .plant_data import PlantTaxon
from .rules import Layout, RuleWeights, hard_constraint_violations


@dataclass(frozen=True)
class AnnealingResult:
    """Best plan and run statistics from one annealing run."""

    layout: Layout
    initial_energy: float
    best_energy: float
    evaluations: int
    accepted_moves: int


def simulated_annealing(
    start: Layout,
    geometry: GridGeometry,
    taxa_data: Mapping[str, PlantTaxon],
    spacing_m: Mapping[str, float],
    structural_taxa: set[str],
    *,
    seed: int,
    iterations: int = 20_000,
    initial_temperature: float = 10.0,
    cooling_rate: float = 0.9995,
    weights: RuleWeights = RuleWeights(),
) -> AnnealingResult:
    """Improve a feasible layout using shift, swap, and group moves."""
    if iterations < 0 or initial_temperature <= 0 or not 0 < cooling_rate <= 1:
        raise ValueError("invalid annealing parameters")
    if hard_constraint_violations(start, geometry, spacing_m):
        raise ValueError("start layout violates hard constraints")
    rng = random.Random(seed)
    current = start
    evaluator = EnergyEvaluator(
        geometry, taxa_data, spacing_m, structural_taxa, weights
    )
    current_energy = evaluator.score(current)
    best = current
    best_energy = current_energy
    accepted = 0
    evaluations = 1
    temperature = initial_temperature

    for _ in range(iterations):
        proposal = _propose(
            current, geometry, spacing_m, rng
        )
        if proposal is not None and not hard_constraint_violations(
            proposal, geometry, spacing_m
        ):
            proposal_energy = evaluator.score(proposal)
            evaluations += 1
            delta = proposal_energy - current_energy
            if delta <= 0 or rng.random() < math.exp(-delta / temperature):
                current, current_energy = proposal, proposal_energy
                accepted += 1
                if current_energy < best_energy:
                    best, best_energy = current, current_energy
        temperature *= cooling_rate

    return AnnealingResult(
        best, evaluator.score(start), best_energy, evaluations, accepted
    )


def energy_delta(
    before: Layout,
    after: Layout,
    geometry: GridGeometry,
    taxa_data: Mapping[str, PlantTaxon],
    spacing_m: Mapping[str, float],
    structural_taxa: set[str],
    weights: RuleWeights = RuleWeights(),
) -> float:
    """Return the exact energy difference used to verify incremental scoring."""
    return EnergyEvaluator(
        geometry, taxa_data, spacing_m, structural_taxa, weights
    ).delta(before, after)


def _propose(
    layout: Layout,
    geometry: GridGeometry,
    spacing_m: Mapping[str, float],
    rng: random.Random,
) -> Layout | None:
    if not layout.taxa:
        return None
    move = rng.choice(("shift", "swap", "group"))
    if move == "shift":
        index = rng.randrange(len(layout.taxa))
        row, column = rng.choice(
            np.argwhere(geometry.admissible_cells(spacing_m[layout.taxa[index]] / 2))
        )
        positions = layout.positions_m.copy()
        positions[index] = (column * geometry.cell_size_m, row * geometry.cell_size_m)
        return Layout(positions, layout.taxa)
    if move == "swap":
        different = [
            (i, j)
            for i in range(len(layout.taxa))
            for j in range(i)
            if layout.taxa[i] != layout.taxa[j]
        ]
        if not different:
            return None
        first, second = rng.choice(different)
        taxa = list(layout.taxa)
        taxa[first], taxa[second] = taxa[second], taxa[first]
        return Layout(layout.positions_m.copy(), tuple(taxa))

    taxon = rng.choice(layout.taxa)
    indices = [i for i, name in enumerate(layout.taxa) if name == taxon]
    if len(indices) < 2:
        return None
    delta = np.asarray(rng.choice((-1, 1)), dtype=float) * geometry.cell_size_m
    axis = rng.randrange(2)
    offset = np.zeros(2)
    offset[axis] = delta
    positions = layout.positions_m.copy()
    positions[indices] += offset
    return Layout(positions, layout.taxa)

