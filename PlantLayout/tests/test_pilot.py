from pathlib import Path

from src.pilot import prepare_exclusion_pilot
from src.plant_data import load_dataset

ROOT = Path(__file__).parents[1]


def test_exclusion_pilot_is_explicit_and_preserves_remaining_counts() -> None:
    dataset = load_dataset(
        ROOT / "data/raw/Plant_traits_list_C.csv",
        ROOT / "data/raw/optimization_report_C.txt",
    )
    excluded = {"Epilobium angustifolium 'Album'"}

    taxa, counts = prepare_exclusion_pilot(
        dataset.taxa, dataset.quantities, excluded_taxa=excluded
    )

    assert "Epilobium angustifolium 'Album'" not in taxa
    assert "Epilobium angustifolium 'Album'" not in counts
    assert sum(counts.values()) == 97
