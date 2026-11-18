"""Reproducible experiment runners and metrics for RQ3."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from .annealing import simulated_annealing
from .geometry import GeometryName, GridGeometry, build_geometry
from .greedy import GreedyResult, greedy_insert
from .plant_data import PlantTaxon
from .rules import Layout, RuleWeights, local_flowering_penalty


@dataclass(frozen=True)
class ExperimentRecord:
    """One seed/method/geometry result in serializable form."""

    geometry: str
    seed: int
    method: str
    feasible: bool
    plants_placed: int
    initial_energy: float
    final_energy: float
    improvement: float
    proportion_deviation: float
    local_flowering_penalty: float
    runtime_s: float
    evaluations: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def run_experiments(
    geometries: Mapping[str, GridGeometry],
    taxa_data: Mapping[str, PlantTaxon],
    counts: Mapping[str, int],
    spacing_m: Mapping[str, float],
    structural_taxa: set[str],
    *,
    seeds: Sequence[int],
    weights: RuleWeights = RuleWeights(),
    annealing_iterations: int = 20_000,
) -> list[ExperimentRecord]:
    """Run matched constrained, random-order, and annealed comparisons."""
    records: list[ExperimentRecord] = []
    for geometry_name, geometry in geometries.items():
        for seed in seeds:
            constrained = _timed_greedy(
                geometry, taxa_data, counts, spacing_m, structural_taxa,
                seed=seed, order="constrained", weights=weights,
            )
            records.append(_record(
                geometry_name, seed, "greedy_constrained", constrained,
                counts, taxa_data, runtime_s=constrained[1],
            ))
            random_order = _timed_greedy(
                geometry, taxa_data, counts, spacing_m, structural_taxa,
                seed=seed, order="random", weights=weights,
            )
            records.append(_record(
                geometry_name, seed, "greedy_random", random_order,
                counts, taxa_data, runtime_s=random_order[1],
            ))
            if constrained[0].feasible:
                started = time.perf_counter()
                annealed = simulated_annealing(
                    constrained[0].layout, geometry, taxa_data, spacing_m,
                    structural_taxa, seed=seed, iterations=annealing_iterations,
                    weights=weights,
                )
                runtime = time.perf_counter() - started + constrained[1]
                records.append(ExperimentRecord(
                    geometry_name, seed, "greedy_sa", True,
                    len(annealed.layout.taxa), annealed.initial_energy,
                    annealed.best_energy,
                    _improvement(annealed.initial_energy, annealed.best_energy),
                    _proportion_deviation(
                        annealed.layout, counts
                    ),
                    local_flowering_penalty(annealed.layout, taxa_data),
                    runtime, annealed.evaluations + constrained[0].evaluations,
                ))
    return records


def default_geometries(cell_size_m: float = 0.2) -> dict[str, GridGeometry]:
    """Build the three proposal geometries with the configured cell size."""
    return {
        name.value: build_geometry(name, cell_size_m=cell_size_m)
        for name in GeometryName
    }


def _timed_greedy(*args, **kwargs) -> tuple[GreedyResult, float]:
    started = time.perf_counter()
    result = greedy_insert(*args, **kwargs)
    return result, time.perf_counter() - started


def _record(
    geometry_name: str,
    seed: int,
    method: str,
    timed: tuple[GreedyResult, float],
    counts: Mapping[str, int],
    taxa_data: Mapping[str, PlantTaxon],
    *,
    runtime_s: float,
) -> ExperimentRecord:
    result = timed[0]
    layout = result.layout
    penalty = (
        local_flowering_penalty(layout, taxa_data)
        if result.feasible and layout.taxa
        else float("nan")
    )
    return ExperimentRecord(
        geometry_name, seed, method, result.feasible, len(layout.taxa),
        float("nan") if not result.feasible else result.energy,
        float("nan") if not result.feasible else result.energy,
        0.0, _proportion_deviation(layout, counts), penalty,
        runtime_s, result.evaluations,
    )


def _improvement(initial: float, final: float) -> float:
    return (initial - final) / initial if initial else 0.0


def _proportion_deviation(
    layout: Layout, target_counts: Mapping[str, int]
) -> float:
    if not layout.taxa:
        return 1.0
    total_target = sum(target_counts.values())
    total_actual = len(layout.taxa)
    actual = {
        name: layout.taxa.count(name) / total_actual for name in target_counts
    }
    return float(sum(
        abs(actual[name] - target_counts[name] / total_target)
        for name in target_counts
    ) / 2)
