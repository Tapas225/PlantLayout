from pathlib import Path

import pytest

from src.capacity import spacing_map
from src.plant_data import load_traits_csv

ROOT = Path(__file__).parents[1]


def test_confirmed_fallback_is_applied_and_unresolved_taxon_is_blocked() -> None:
    taxa = load_traits_csv(ROOT / "data/raw/Plant_traits_list_C.csv")

    with pytest.raises(ValueError, match="Epilobium"):
        spacing_map(
            taxa,
            fallbacks_cm={"Cichorium intybus var. sativum": 50},
        )


def test_fallback_is_converted_from_centimetres_to_metres() -> None:
    taxa = load_traits_csv(ROOT / "data/raw/Plant_traits_list_C.csv")
    selected = {
        "Cichorium intybus var. sativum": taxa["Cichorium intybus var. sativum"]
    }

    resolved = spacing_map(
        selected,
        fallbacks_cm={"Cichorium intybus var. sativum": 50},
    )

    assert resolved == {"Cichorium intybus var. sativum": 0.5}
