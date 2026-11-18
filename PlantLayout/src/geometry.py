"""Grid geometries and hard plantable-area clearance calculations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from scipy.ndimage import distance_transform_edt


class GeometryName(StrEnum):
    """Geometries used by the proposal's initial experiments."""

    RECTANGLE = "rectangle"
    PATH = "path"
    HOLE = "hole"


@dataclass(frozen=True)
class GridGeometry:
    """A regular grid with a plantable mask and boundary clearance map."""

    mask: np.ndarray
    cell_size_m: float
    clearance_m: np.ndarray

    def __post_init__(self) -> None:
        if self.mask.ndim != 2 or self.mask.dtype != np.bool_:
            raise ValueError("mask must be a two-dimensional boolean array")
        if self.cell_size_m <= 0:
            raise ValueError("cell_size_m must be positive")
        if self.clearance_m.shape != self.mask.shape:
            raise ValueError("clearance_m must have the same shape as mask")
        if np.any(self.clearance_m[~self.mask] != 0):
            raise ValueError("non-plantable cells must have zero clearance")

    @property
    def shape(self) -> tuple[int, int]:
        return self.mask.shape

    @property
    def plantable_cells(self) -> int:
        return int(np.count_nonzero(self.mask))

    def admissible_cells(self, radius_m: float) -> np.ndarray:
        """Return coordinates whose cell centre has enough boundary clearance."""
        if radius_m < 0:
            raise ValueError("radius_m must not be negative")
        return self.mask & (self.clearance_m >= radius_m)


def build_geometry(
    name: GeometryName | str,
    *,
    cell_size_m: float = 0.2,
    width_m: float = 5.0,
    height_m: float = 10.0,
) -> GridGeometry:
    """Build a deterministic 5 m by 10 m geometry at the requested resolution.

    The path and hole use the same outer bounding box as the rectangle. This
    holds the external geometry constant while measuring the effect of lost
    plantable cells.
    """
    try:
        geometry_name = GeometryName(name)
    except ValueError as exc:
        raise ValueError(f"unknown geometry: {name!r}") from exc

    columns = _dimension_to_cells(width_m, cell_size_m, "width_m")
    rows = _dimension_to_cells(height_m, cell_size_m, "height_m")
    mask = np.ones((rows, columns), dtype=bool)

    if geometry_name is GeometryName.PATH:
        path_width = max(1, round(0.4 / cell_size_m))
        start = (columns - path_width) // 2
        mask[:, start : start + path_width] = False
    elif geometry_name is GeometryName.HOLE:
        hole_width = max(1, round(2.0 / cell_size_m))
        hole_height = max(1, round(2.0 / cell_size_m))
        column_start = (columns - hole_width) // 2
        row_start = (rows - hole_height) // 2
        mask[row_start : row_start + hole_height, column_start : column_start + hole_width] = False

    padded_mask = np.pad(mask, 1, mode="constant", constant_values=False)
    clearance = distance_transform_edt(
        padded_mask, sampling=cell_size_m
    )[1:-1, 1:-1]
    clearance[~mask] = 0.0
    return GridGeometry(mask=mask, cell_size_m=cell_size_m, clearance_m=clearance)


def _dimension_to_cells(value_m: float, cell_size_m: float, name: str) -> int:
    if value_m <= 0:
        raise ValueError(f"{name} must be positive")
    cells = round(value_m / cell_size_m)
    if not np.isclose(cells * cell_size_m, value_m):
        raise ValueError(f"{name} must be a multiple of cell_size_m")
    return cells
