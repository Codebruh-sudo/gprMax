# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

"""Point ownership using real MPI extents, without creating communicators.

These tests cover the partition/bounds contract, not halo communication or
field updates. Points include the terminal node on every global axis, as
required for the same source/receiver coordinates accepted by a serial grid.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from gprMax.grid.mpi_grid import MPIGrid

pytestmark = pytest.mark.unit

PARTITIONS = [
    pytest.param((7, 5, 3), (1, 1, 1), id="unsplit"),
    pytest.param((8, 6, 4), (2, 3, 2), id="equal-extents"),
    pytest.param((7, 5, 3), (3, 2, 2), id="unequal-extents"),
]
for axis in range(3):
    partition = [1, 1, 1]
    partition[axis] = 3
    PARTITIONS.append(pytest.param((7, 5, 3), tuple(partition), id=f"split-{'xyz'[axis]}-only"))
    for thickness in (1, 2):
        size = [7, 5, 3]
        size[axis] = thickness
        partition = [2, 2, 2]
        partition[axis] = 1
        PARTITIONS.append(
            pytest.param(
                tuple(size), tuple(partition), id=f"thin-{'xyz'[axis]}-{thickness}-unsplit"
            )
        )
        # MPIGrid.build permits up to global_size + 1 ranks on an axis.
        # The final rank then owns no cells but still owns the terminal node.
        # This checks ownership only, not whether such a thin decomposition
        # supports a particular field solver or PML configuration.
        partition[axis] = thickness + 1
        PARTITIONS.append(
            pytest.param(
                tuple(size), tuple(partition), id=f"thin-{'xyz'[axis]}-{thickness}-terminal-rank"
            )
        )


def _partition(global_size, partition):
    ranks = []
    for coords in np.ndindex(*partition):
        grid = MPIGrid.__new__(MPIGrid)
        grid.global_size = np.asarray(global_size, dtype=np.int32)
        grid.mpi_tasks = np.asarray(partition, dtype=np.int32)
        grid.comm = SimpleNamespace(coords=list(coords))
        # Extent calculation only needs neighbour presence, not rank IDs.
        grid.neighbours = np.full((3, 2), -1, dtype=np.int32)
        for axis in range(3):
            if coords[axis] > 0:
                grid.neighbours[axis, 0] = 0
            if coords[axis] < partition[axis] - 1:
                grid.neighbours[axis, 1] = 0
        grid.calculate_local_extents()
        ranks.append(grid)
    return ranks


@pytest.mark.parametrize("global_size,partition", PARTITIONS)
def test_every_global_node_has_one_owner_including_faces_edges_and_corners(global_size, partition):
    ranks = _partition(global_size, partition)
    missing, duplicated, wrong_terminal_owner = [], [], []
    for point in np.ndindex(*(np.asarray(global_size) + 1)):
        global_point = np.asarray(point, dtype=np.int32)
        owners = [
            grid
            for grid in ranks
            if grid.within_bounds(grid.global_to_local_coordinate(global_point))
        ]
        if not owners:
            missing.append(point)
        elif len(owners) > 1:
            duplicated.append((point, [grid.coords for grid in owners]))
        else:
            np.testing.assert_array_equal(
                owners[0].coords,
                owners[0].get_grid_coord_from_coordinate(global_point),
                err_msg=f"Point ownership and source/receiver routing disagree at {point}",
            )
            for axis in range(3):
                if (
                    point[axis] == global_size[axis]
                    and owners[0].coords[axis] != partition[axis] - 1
                ):
                    wrong_terminal_owner.append((point, owners[0].coords))
    assert not missing, f"{len(missing)} unowned global nodes, including {missing[:8]}"
    assert (
        not duplicated
    ), f"{len(duplicated)} multiply owned global nodes, including {duplicated[:8]}"
    assert (
        not wrong_terminal_owner
    ), f"Global terminal nodes assigned before the last rank: {wrong_terminal_owner[:8]}"


@pytest.mark.parametrize("global_size,partition", PARTITIONS)
def test_internal_negative_and_positive_halos_are_not_owned(global_size, partition):
    ranks = _partition(global_size, partition)
    for grid in ranks:
        for axis in range(3):
            for direction in (0, 1):
                if grid.neighbours[axis, direction] < 0:
                    continue
                local_point = grid.negative_halo_offset.astype(np.int32)
                local_point[axis] = 0 if direction == 0 else grid.size[axis]
                global_point = grid.local_to_global_coordinate(local_point)
                assert not grid.within_bounds(local_point), (
                    grid.coords,
                    axis,
                    direction,
                    global_point,
                )


@pytest.mark.parametrize("axis", range(3), ids=list("xyz"))
@pytest.mark.parametrize("side", ("negative", "positive"))
def test_points_outside_global_bounds_raise_the_axis_on_every_rank(axis, side):
    global_size = (7, 5, 3)
    global_point = np.asarray((3, 2, 1), dtype=np.int32)
    global_point[axis] = -1 if side == "negative" else global_size[axis] + 1
    for grid in _partition(global_size, (3, 2, 2)):
        with pytest.raises(ValueError, match=f"^{'xyz'[axis]}$"):
            grid.within_bounds(grid.global_to_local_coordinate(global_point))
