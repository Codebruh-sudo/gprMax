"""Public construction order is independent of labels, runtime pages and ranks."""

import copy

import h5py
import numpy as np
import pytest

from gprMax.fields_outputs import write_hd5_data
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.user_objects.cmds_multiuse import Rx

pytestmark = pytest.mark.unit


def test_output_sorts_ordinals_not_runtime_pages(make_view_grid, make_rx, tmp_path):
    grid = make_view_grid(nx=8, ny=8, nz=8)
    receivers = []
    for index, name in enumerate(("zulu", "alpha", "same", "same")):
        rx = make_rx(ID=name, position=(2, 2, 2), outputs=("Ex",))
        rx.build_index, rx.name_kind = index, "user"
        rx.outputs["Ex"][:] = index + 10
        receivers.append(rx)
    private = make_rx(ID="private", outputs=("Ex",))
    private.internal = True
    # Simulate arbitrary rank gather/migration order, including coincident IDs.
    grid.rxs = [receivers[2], private, receivers[0], receivers[3], receivers[1]]
    original = list(grid.rxs)
    path = tmp_path / "ordered.h5"
    with h5py.File(path, "w") as output:
        write_hd5_data(output, grid)
    assert grid.rxs == original
    with h5py.File(path) as output:
        assert output.attrs["ReceiverOrder"] == "construction"
        assert output.attrs["ReceiverOrderSchemaVersion"] == 1
        assert output.attrs["nrx"] == 4
        for index, receiver in enumerate(receivers):
            group = output[f"rxs/rx{index+1}"]
            assert group.attrs["BuildIndex"] == index
            assert group.attrs["Name"] == receiver.ID
            np.testing.assert_array_equal(group["Ex"], index + 10)


@pytest.mark.parametrize("indices", [(0, 0), (0, None), (0, 2), (-1, 0)])
def test_writer_rejects_invalid_global_indices(make_view_grid, make_rx, tmp_path, indices):
    grid = make_view_grid(nx=8, ny=8, nz=8)
    grid.rxs = [make_rx(ID="same", outputs=("Ex",)) for _ in indices]
    for rx, index in zip(grid.rxs, indices):
        rx.build_index = index
    with h5py.File(tmp_path / "invalid.h5", "w") as output:
        with pytest.raises(ValueError, match="build indices"):
            write_hd5_data(output, grid)


def test_ordinals_allocated_on_every_rank_before_ownership(monkeypatch):
    grids = [FDTDGrid(), FDTDGrid()]
    owner = [False, True, False, True]
    for rank, grid in enumerate(grids):
        for number, belongs_to_one in enumerate(owner):
            command = Rx(p1=(number, 0, 0), id=f"name{3-number}")

            class Inputs:
                def resolve_inf_point(self, point):
                    return point

                def check_src_rx_point(self, point, message):
                    return belongs_to_one == bool(rank), np.array(point)

                def round_to_grid_static_point(self, point):
                    return point

            monkeypatch.setattr(command, "_create_uip", lambda grid: Inputs())
            monkeypatch.setattr(
                command, "_create_receiver", lambda grid, coord: type("Receiver", (), {"outputs": {}, "coord": coord})()
            )
            command.build(grid)
    assert grids[0].receiver_definitions == grids[1].receiver_definitions
    assert [rx.build_index for rx in grids[0].rxs] == [0, 2]
    assert [rx.build_index for rx in grids[1].rxs] == [1, 3]
    # MPI transmits the object; insertion on a new owner must not allocate again.
    migrated = copy.deepcopy(grids[0].rxs[1])
    grids[1].add_receiver(migrated)
    assert migrated.build_index == 2
    assert len(grids[1].receiver_definitions) == 4


@pytest.mark.parametrize("count", (0, 1))
def test_writer_rejects_lost_tail_of_gathered_receivers(make_view_grid, make_rx, tmp_path, count):
    grid = make_view_grid(nx=8, ny=8, nz=8)
    grid.receiver_definitions = [("one", ["Ex"], None), ("two", ["Ex"], None)]
    grid.rxs = [make_rx(ID="one", outputs=("Ex",)) for _ in range(count)]
    for rx in grid.rxs:
        rx.build_index = 0
    with h5py.File(tmp_path / "incomplete.h5", "w") as output:
        with pytest.raises(ValueError, match="registered declarations"):
            write_hd5_data(output, grid)
