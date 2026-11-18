from pathlib import Path

import pytest

from src.plant_data import load_dataset, load_traits_csv

ROOT = Path(__file__).parents[1]
TRAITS = ROOT / "data/raw/Plant_traits_list_C.csv"
REPORT = ROOT / "data/raw/optimization_report_C.txt"


def test_example_dataset_matches_report() -> None:
    dataset = load_dataset(TRAITS, REPORT)

    assert len(dataset.taxa) == 17
    assert dataset.total_quantity == 100
    assert dataset.quantities["Lathyrus pratensis"] == 16


def test_duplicate_sociability_intervals_are_preserved() -> None:
    taxa = load_traits_csv(TRAITS)

    values = taxa["Ajuga reptans 'Stölzle'"].values("Geselligkeitsstufe")

    assert len(values) == 2
    assert values[0].startswith("Stufe III")
    assert values[1].startswith("Stufe II")


def test_missing_spacing_is_not_silently_filled() -> None:
    taxa = load_traits_csv(TRAITS)

    assert taxa["Cichorium intybus var. sativum"].numeric_values(
        "Pflanzabstand von"
    ) == ()
    assert taxa["Epilobium angustifolium 'Album'"].values(
        "Pflanzabstand von"
    ) == ()
