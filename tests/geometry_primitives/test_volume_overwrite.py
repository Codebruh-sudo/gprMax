"""A replaced volume must not leave magnetic ownership in adjacent cells."""

import numpy as np
import pytest

from gprMax.cython import geometry_primitives as primitives
from gprMax.cython.yee_cell_build import build_electric_components, build_magnetic_components
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.materials import Material, create_built_in_materials

pytestmark = pytest.mark.unit
SHAPES = ("box", "sphere", "ellipsoid", "cylinder", "cone", "triangle", "sector")
DL = 0.001


def _grid():
    grid = FDTDGrid()
    grid.nx, grid.ny, grid.nz = 20, 21, 22
    create_built_in_materials(grid)
    for name, er, mr in (("old", 4, 7), ("replacement", 2, 3)):
        material = Material(len(grid.materials), name)
        material.er = er
        material.mr = mr
        grid.materials.append(material)
    grid.initialise_geometry_arrays()
    grid.tags = np.full(tuple(grid.size), 3, np.uint16)
    return grid


def _args(grid, material, averaging, tags=True):
    args = (
        material,
        material,
        material,
        material,
        averaging,
        False,
        False,
        False,
        grid.solid,
        grid.rigidE,
        grid.rigidH,
        grid.ID,
    )
    return (*args, grid.tags, 5 if material == 3 else 7) if tags else args


def _shape(grid, name, material, averaging):
    args = _args(grid, material, averaging)
    if name == "box":
        return primitives.build_box(8, 12, 8, 12, 8, 12, *args)
    if name == "sphere":
        return primitives.build_sphere(10, 10, 10, 0.0032, DL, DL, DL, *args)
    if name == "ellipsoid":
        return primitives.build_ellipsoid(10, 10, 10, 0.0032, 0.0022, 0.0042, DL, DL, DL, *args)
    if name == "cylinder":
        return primitives.build_cylinder(0.007, 0.010, 0.010, 0.013, 0.010, 0.010, 0.0022, DL, DL, DL, *args)
    if name == "cone":
        return primitives.build_cone(0.007, 0.010, 0.010, 0.013, 0.010, 0.010, 0.0012, 0.0032, DL, DL, DL, *args)
    if name == "triangle":
        return primitives.build_triangle(
            0.007, 0.007, 0.008, 0.013, 0.007, 0.008, 0.010, 0.014, 0.008, "z", 0.004, DL, DL, DL, *args
        )
    return primitives.build_cylindrical_sector(
        0.010, 0.010, 0.008, 0, 1.5 * np.pi, 0.0032, "z", 0.004, DL, DL, DL, *args
    )


def _components(grid):
    build_electric_components(grid.solid, grid.rigidE, grid.ID, grid)
    build_magnetic_components(grid.solid, grid.rigidH, grid.ID, grid)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("old_averaging", [False, True])
@pytest.mark.parametrize("new_averaging", [False, True])
def test_same_volume_replacement_matches_fresh_geometry(shape, old_averaging, new_averaging):
    reference, overwritten = _grid(), _grid()
    _shape(reference, shape, 4, new_averaging)
    _shape(overwritten, shape, 3, old_averaging)
    _shape(overwritten, shape, 4, new_averaging)
    _components(reference)
    _components(overwritten)
    for name in ("solid", "rigidE", "rigidH", "ID", "tags"):
        np.testing.assert_array_equal(getattr(overwritten, name), getattr(reference, name), err_msg=name)
    assert not np.any(overwritten.ID == 3), "No component may retain the overwritten material"


@pytest.mark.parametrize("shape", SHAPES)
def test_volume_rigid_h_claims_belong_only_to_its_occupied_cells(shape):
    grid = _grid()
    _shape(grid, shape, 3, False)
    occupied = grid.solid == 3
    assert occupied.any()
    np.testing.assert_array_equal(grid.rigidH, np.broadcast_to(occupied, grid.rigidH.shape))
    # This is an ownership change only: the represented magnetic positions
    # remain each voxel's two own-axis faces, not its tangential corners.
    for axis in range(3):
        expected = np.zeros(grid.ID.shape[1:], bool)
        for offset in (0, 1):
            spatial = [slice(0, size) for size in grid.size]
            spatial[axis] = slice(offset, int(grid.size[axis]) + offset)
            expected[tuple(spatial)] |= occupied
        np.testing.assert_array_equal(grid.ID[axis + 3] == 3, expected)


@pytest.mark.parametrize("axis", range(3), ids=("x", "y", "z"))
@pytest.mark.parametrize("side", [-1, 1])
def test_partial_replacement_preserves_retained_neighbours_claim(axis, side):
    reference, overwritten = _grid(), _grid()
    retained = np.array([10, 10, 10])
    removed = retained.copy()
    removed[axis] += side
    for grid in (reference, overwritten):
        primitives.build_voxel(*retained, *_args(grid, 3, False, tags=False))
    primitives.build_voxel(*removed, *_args(overwritten, 3, False, tags=False))
    for grid in (reference, overwritten):
        primitives.build_voxel(*removed, *_args(grid, 4, True, tags=False))
        _components(grid)
    for name in ("solid", "rigidE", "rigidH", "ID"):
        np.testing.assert_array_equal(getattr(overwritten, name), getattr(reference, name), err_msg=name)
    face = retained.copy()
    face[axis] += int(side > 0)
    assert overwritten.ID[(axis + 3, *face)] == 3


@pytest.mark.parametrize("axis", range(3), ids=("x", "y", "z"))
def test_explicit_magnetic_edge_keeps_independent_neighbour_claim(axis):
    grid = _grid()
    position = np.array([10, 10, 10])
    (primitives.build_magnetic_edge_x, primitives.build_magnetic_edge_y, primitives.build_magnetic_edge_z)[axis](
        *position, 3, grid.rigidH, grid.ID
    )
    assert np.count_nonzero(grid.rigidH) == 2
    primitives.build_voxel(*position, *_args(grid, 4, True, tags=False))
    _components(grid)
    assert grid.ID[(axis + 3, *position)] == 3
