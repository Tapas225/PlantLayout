from pathlib import Path

import pytest

from src.capacity import (
    counts_for_total,
    disc_area_screen,
    largest_remainder_counts,
    search_capacity,
)
from src.plant_data import load_dataset

ROOT = Path(__file__).parents[1]


def test_largest_remainder_allocates_exact_total() -> None:
    counts = largest_remainder_counts({"a": 0.5, "b": 0.3, "c": 0.2}, 7)

    assert counts == {"a": 4, "b": 2, "c": 1}
    assert sum(counts.values()) == 7


def test_counts_for_total_preserves_example_frequencies() -> None:
    dataset = load_dataset(
        ROOT / "data/raw/Plant_traits_list_C.csv",
        ROOT / "data/raw/optimization_report_C.txt",
    )

    counts = counts_for_total(dataset.quantities, 500)

    assert sum(counts.values()) == 500
    assert counts["Lathyrus pratensis"] == 80
    assert counts["Cichorium intybus var. sativum"] == 135


def test_disc_screen_uses_lower_spacing_and_reports_area() -> None:
    dataset = load_dataset(
        ROOT / "data/raw/Plant_traits_list_C.csv",
        ROOT / "data/raw/optimization_report_C.txt",
    )
    counts = {"Lathyrus pratensis": 1}

    screen = disc_area_screen(
        dataset.taxa, counts, available_area_m2=2.0
    )

    assert screen.required_area_m2 == pytest.approx(0.5026548)
    assert screen.passes


def test_capacity_search_records_neighbouring_checks() -> None:
    result = search_capacity(
        maximum=10,
        feasible=lambda total: total <= 6,
    )

    assert result.capacity == 6
    assert dict(result.checks)[5] is True
    assert dict(result.checks)[7] is False
