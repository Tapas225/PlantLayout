from pathlib import Path

import numpy as np
import pytest

from src.evaluator import EnergyEvaluator
from src.geometry import build_geometry
from src.greedy import greedy_insert
from src.plant_data import load_dataset
from src.rules import Layout

ROOT = Path(__file__).parents[1]


def test_evaluator_delta_matches_two_scores() -> None:
    dataset = load_dataset(
        ROOT / "data/raw/Plant_traits_list_C.csv",
        ROOT / "data/raw/optimization_report_C.txt",
    )
    geometry = build_geometry("rectangle")
    spacing = {"Cardamine trifolia": 0.2, "Lathyrus pratensis": 0.8}
    start = greedy_insert(
        geometry, dataset.taxa,
        {"Cardamine trifolia": 2, "Lathyrus pratensis": 1},
        spacing, set(), seed=1,
    ).layout
    after = Layout(start.positions_m.copy(), start.taxa)
    after.positions_m[0] += np.array([0.2, 0.0])
    evaluator = EnergyEvaluator(geometry, dataset.taxa, spacing, set())

    assert evaluator.delta(start, after) == pytest.approx(
        evaluator.score(after) - evaluator.score(start)
    )
