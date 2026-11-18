"""Explicit preparation of reduced pilot datasets."""

from __future__ import annotations

from collections.abc import Mapping, Set

from .plant_data import PlantTaxon


def prepare_exclusion_pilot(
    taxa: Mapping[str, PlantTaxon],
    counts: Mapping[str, int],
    *,
    excluded_taxa: Set[str],
) -> tuple[dict[str, PlantTaxon], dict[str, int]]:
    """Return inputs for a labeled pilot that excludes named unresolved taxa."""
    missing = sorted(set(excluded_taxa) - set(taxa))
    if missing:
        raise ValueError(f"cannot exclude unknown taxa: {missing}")
    if missing := sorted(set(excluded_taxa) - set(counts)):
        raise ValueError(f"excluded taxa have no quantities: {missing}")
    return (
        {name: taxon for name, taxon in taxa.items() if name not in excluded_taxa},
        {name: count for name, count in counts.items() if name not in excluded_taxa},
    )
