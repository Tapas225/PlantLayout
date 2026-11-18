"""Loading and validation for the long-format plant data."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REQUIRED_COLUMNS = {
    "id",
    "scientific_name",
    "name",
    "allowed_type",
    "value_str",
    "description",
}
REPORT_PLANT_LINE = re.compile(
    r"^(?P<name>.+?)\s+\|\s+[^|]+\|\s+(?P<count>\d+)\s+Stk\.\s*$"
)


@dataclass(frozen=True)
class TraitRow:
    """One source row, retaining the original value and metadata."""

    record_id: str
    scientific_name: str
    name: str
    allowed_type: str
    value: str
    description: str


@dataclass(frozen=True)
class PlantTaxon:
    """All source traits for one scientific taxon."""

    scientific_name: str
    rows: tuple[TraitRow, ...]

    def values(self, trait_name: str) -> tuple[str, ...]:
        """Return all values for a trait, including duplicates and intervals."""
        return tuple(row.value for row in self.rows if row.name == trait_name)

    def numeric_values(self, trait_name: str) -> tuple[float, ...]:
        """Return numeric trait values, raising for malformed non-empty data."""
        values = self.values(trait_name)
        parsed: list[float] = []
        for value in values:
            match = re.fullmatch(r"\s*(-?\d+(?:[.,]\d+)?)\s*", value)
            if not match:
                raise ValueError(
                    f"{self.scientific_name!r} has non-numeric {trait_name!r}: "
                    f"{value!r}"
                )
            parsed.append(float(match.group(1).replace(",", ".")))
        return tuple(parsed)


@dataclass(frozen=True)
class PlantDataset:
    """Validated plant traits and report quantities."""

    taxa: dict[str, PlantTaxon]
    quantities: dict[str, int]

    @property
    def total_quantity(self) -> int:
        return sum(self.quantities.values())


def load_traits_csv(path: str | Path) -> dict[str, PlantTaxon]:
    """Load long-format traits while preserving source row order and values."""
    source = Path(path)
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"{source} is missing columns: {sorted(missing)}")

        grouped: dict[str, list[TraitRow]] = {}
        for line_number, row in enumerate(reader, start=2):
            name = (row["scientific_name"] or "").strip()
            trait_name = (row["name"] or "").strip()
            if not name or not trait_name:
                raise ValueError(f"{source}:{line_number} has an empty taxon or trait")
            grouped.setdefault(name, []).append(
                TraitRow(
                    record_id=(row["id"] or "").strip(),
                    scientific_name=name,
                    name=trait_name,
                    allowed_type=(row["allowed_type"] or "").strip(),
                    value=(row["value_str"] or "").strip(),
                    description=(row["description"] or "").strip(),
                )
            )

    return {
        name: PlantTaxon(name, tuple(rows)) for name, rows in grouped.items()
    }


def load_report_quantities(path: str | Path) -> dict[str, int]:
    """Read taxon quantities from the report's detailed plant list."""
    source = Path(path)
    quantities: dict[str, int] = {}
    in_plant_list = False
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip() == "DETAILLIERTE PFLANZLISTE:":
            in_plant_list = True
            continue
        if not in_plant_list:
            continue
        match = REPORT_PLANT_LINE.match(line)
        if match:
            name = match.group("name").strip()
            if name in quantities:
                raise ValueError(f"{source} contains duplicate taxon: {name}")
            quantities[name] = int(match.group("count"))
    if not quantities:
        raise ValueError(f"{source} contains no taxon summary rows")
    return quantities


def load_dataset(traits_path: str | Path, report_path: str | Path) -> PlantDataset:
    """Load both sources and verify that the report taxa exist in the CSV."""
    taxa = load_traits_csv(traits_path)
    quantities = load_report_quantities(report_path)
    missing = sorted(set(quantities) - set(taxa))
    if missing:
        raise ValueError(f"Report taxa missing from traits CSV: {missing}")
    return PlantDataset(taxa=taxa, quantities=quantities)


def missing_numeric_trait(
    taxa: Iterable[PlantTaxon], trait_name: str
) -> tuple[str, ...]:
    """Return taxa without a value for a required numeric trait."""
    return tuple(
        taxon.scientific_name for taxon in taxa if not taxon.numeric_values(trait_name)
    )
