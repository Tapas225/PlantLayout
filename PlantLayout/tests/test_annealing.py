from pathlib import Path

import numpy as np
import pytest

from src.annealing import energy_delta, simulated_annealing
from src.geometry import build_geometry
from src.greedy import greedy_insert
from src.plant_data import load_dataset
from src.rules import Layout

ROOT = Path(__file__).parents[1]


@pytest.fixture
def context():
    dataset = load_dataset(
        ROOT / "data/raw/Plant_traits_list_C.csv",
        ROOT / "data/raw/optimization_report_C.txt",
    )
    geometry = build_geometry("rectangle")
    counts = {"Cardamine trifolia": 2, "Lathyrus pratensis": 1}
    spacing = {"Cardamine trifolia": 0.2, "Lathyrus pratensis": 0.8}
    greedy = greedy_insert(
        geometry, dataset.taxa, counts, spacing, set(), seed=2
    )
    return dataset, geometry, spacing, greedy.layout


def test_energy_delta_matches_full_energy_difference(context) -> None:
    dataset, geometry, spacing, layout = context
    moved = Layout(layout.positions_m.copy(), layout.taxa)
    moved.positions_m[0] += np.array([0.2, 0.0])

    delta = energy_delta(
        layout, moved, geometry, dataset.taxa, spacing, set()
    )

    from src.rules import energy

    expected = energy(
        moved, geometry, dataset.taxa, spacing, set()
    ).total - energy(
        layout, geometry, dataset.taxa, spacing, set()
    ).total
    assert delta == pytest.approx(expected)


def test_annealing_retains_feasible_best_layout(context) -> None:
    dataset, geometry, spacing, layout = context

    result = simulated_annealing(
        layout,
        geometry,
        dataset.taxa,
        spacing,
        set(),
        seed=4,
        iterations=50,
    )

    assert result.best_energy <= result.initial_energy
    assert result.evaluations >= 1
    assert result.accepted_moves >= 0
