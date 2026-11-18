from pathlib import Path

from src.experiments import default_geometries, run_experiments
from src.plant_data import load_dataset

ROOT = Path(__file__).parents[1]


def test_experiment_runner_records_matched_methods() -> None:
    dataset = load_dataset(
        ROOT / "data/raw/Plant_traits_list_C.csv",
        ROOT / "data/raw/optimization_report_C.txt",
    )
    geometries = {"rectangle": default_geometries()["rectangle"]}
    counts = {"Cardamine trifolia": 2, "Lathyrus pratensis": 1}
    spacing = {"Cardamine trifolia": 0.2, "Lathyrus pratensis": 0.8}

    records = run_experiments(
        geometries, dataset.taxa, counts, spacing, set(),
        seeds=[0, 1], annealing_iterations=5,
    )

    assert len(records) == 6
    assert {record.method for record in records} == {
        "greedy_constrained", "greedy_random", "greedy_sa"
    }
    assert all(record.geometry == "rectangle" for record in records)
    assert all(record.seed in {0, 1} for record in records)
