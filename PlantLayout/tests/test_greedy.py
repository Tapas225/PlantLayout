from pathlib import Path

import numpy as np

from src.geometry import build_geometry
from src.greedy import greedy_insert
from src.plant_data import load_dataset

ROOT = Path(__file__).parents[1]


def test_constrained_greedy_places_requested_small_layout() -> None:
    dataset = load_dataset(
        ROOT / "data/raw/Plant_traits_list_C.csv",
        ROOT / "data/raw/optimization_report_C.txt",
    )
    geometry = build_geometry("rectangle")
    counts = {"Cardamine trifolia": 2, "Lathyrus pratensis": 1}
    spacing = {"Cardamine trifolia": 0.2, "Lathyrus pratensis": 0.8}

    result = greedy_insert(
        geometry,
        dataset.taxa,
        counts,
        spacing,
        set(),
        seed=3,
    )

    assert result.feasible
    assert len(result.layout.taxa) == 3
    assert result.evaluations > 0
    assert np.isfinite(result.energy)


def test_greedy_marks_failed_placement_explicitly() -> None:
    dataset = load_dataset(
        ROOT / "data/raw/Plant_traits_list_C.csv",
        ROOT / "data/raw/optimization_report_C.txt",
    )
    geometry = build_geometry("rectangle", width_m=0.2, height_m=0.2)

    result = greedy_insert(
        geometry,
        dataset.taxa,
        {"Lathyrus pratensis": 2},
        {"Lathyrus pratensis": 0.8},
        set(),
        seed=0,
    )

    assert not result.feasible
    assert result.failed_taxon == "Lathyrus pratensis"
