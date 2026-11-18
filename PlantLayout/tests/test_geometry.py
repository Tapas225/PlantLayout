import numpy as np
import pytest

from src.geometry import GeometryName, build_geometry


def test_rectangle_has_expected_grid_and_padded_boundary_clearance() -> None:
    geometry = build_geometry(GeometryName.RECTANGLE)

    assert geometry.shape == (50, 25)
    assert geometry.plantable_cells == 1250
    assert geometry.clearance_m[0, 0] == pytest.approx(0.2)
    assert geometry.clearance_m[25, 12] == pytest.approx(2.6)


def test_path_and_hole_keep_outer_box_but_remove_cells() -> None:
    rectangle = build_geometry("rectangle")
    path = build_geometry("path")
    hole = build_geometry("hole")

    assert path.shape == rectangle.shape == hole.shape
    assert path.plantable_cells < rectangle.plantable_cells
    assert hole.plantable_cells < rectangle.plantable_cells


def test_admissible_cells_use_radius_and_never_include_masks() -> None:
    geometry = build_geometry("path")

    admissible = geometry.admissible_cells(0.3)

    assert admissible.dtype == np.bool_
    assert not np.any(admissible & ~geometry.mask)
    assert np.count_nonzero(admissible) < geometry.plantable_cells


def test_invalid_geometry_and_dimensions_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown geometry"):
        build_geometry("triangle")
    with pytest.raises(ValueError, match="multiple"):
        build_geometry("rectangle", width_m=1.1)
