from pathlib import Path

import numpy as np
import pytest

from src.geometry import build_geometry
from src.plant_data import load_dataset
from src.rules import (
    Layout,
    grouping_penalty,
    hard_constraint_violations,
    local_flowering_penalty,
)

ROOT = Path(__file__).parents[1]


@pytest.fixture
def dataset():
    return load_dataset(
        ROOT / "data/raw/Plant_traits_list_C.csv",
        ROOT / "data/raw/optimization_report_C.txt",
    )


def test_hard_constraints_reject_nonplantable_and_too_close_positions(dataset) -> None:
    geometry = build_geometry("rectangle")
    taxa = ("Cardamine trifolia", "Cardamine trifolia")
    layout = Layout(np.array([[0.2, 0.2], [0.2, 0.3]]), taxa)
    spacing = {"Cardamine trifolia": 0.2}

    assert hard_constraint_violations(layout, geometry, spacing) == 1


def test_grouping_preserves_multiple_sociability_intervals(dataset) -> None:
    layout = Layout(
        np.array([[1.0, 1.0], [1.2, 1.0], [1.4, 1.0]]),
        ("Ajuga reptans 'Stölzle'",) * 3,
    )

    penalty = grouping_penalty(
        layout,
        dataset.taxa,
        {"Ajuga reptans 'Stölzle'": 0.25},
    )

    assert penalty == pytest.approx(0.0)


def test_local_flowering_penalty_is_zero_for_full_local_coverage(dataset) -> None:
    layout = Layout(
        np.array([[1.0, 1.0], [1.2, 1.0], [1.4, 1.0]]),
        ("Cardamine trifolia", "Veronica chamaedrys", "Centaurea jacea"),
    )

    penalty = local_flowering_penalty(layout, dataset.taxa, blocks_per_axis=1)

    assert penalty == pytest.approx(0.5)
