"""Reusable exact energy evaluator for optimization loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .geometry import GridGeometry
from .plant_data import PlantTaxon
from .rules import Layout, RuleWeights, energy


@dataclass
class EnergyEvaluator:
    """Score layouts consistently and verify proposed energy deltas.

    The implementation deliberately uses the full rule model as the reference
    until local rule caches are introduced. Keeping this API separate prevents
    the optimizer from duplicating scoring logic during that transition.
    """

    geometry: GridGeometry
    taxa_data: Mapping[str, PlantTaxon]
    spacing_m: Mapping[str, float]
    structural_taxa: set[str]
    weights: RuleWeights = RuleWeights()

    def score(self, layout: Layout) -> float:
        return energy(
            layout,
            self.geometry,
            self.taxa_data,
            self.spacing_m,
            self.structural_taxa,
        ).weighted(self.weights)

    def delta(self, before: Layout, after: Layout) -> float:
        """Return the exact proposal delta for verification and acceptance."""
        return self.score(after) - self.score(before)
