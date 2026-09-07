"""Deterministic message-matching tests, without MPI transport or field solves.

The queue deliberately puts unrelated traffic before the expected message.
Real MPI scan/restart parity is covered in test_mpi_release_regressions.py.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from mpi4py import MPI

from gprMax import config
from gprMax.grid.axes import Dim, Dir
from gprMax.grid.mpi_grid import MPIGrid
from gprMax.receivers import Rx
from gprMax.sources import HertzianDipole, MagneticDipole

pytestmark = pytest.mark.unit


class QueuedComm:
    size = 3

    def __init__(self, rank, messages=(), incoming=0):
        self.rank = rank
        self.coords = [rank, 0, 0]
        self.messages = list(messages)
        self.incoming = incoming
        self.sent = []
        self.received = []

    def Get_rank(self):
        return self.rank

    def Get_cart_rank(self, coord):
        return coord[0]

    def isend(self, value, dest, tag=0):
        self.sent.append((dest, tag, MPI.pickle.dumps(value)))
        return MPI.REQUEST_NULL

    def Reduce(self, *args, **kwargs):
        pass

    def Scatter(self, sendbuf, recvbuf):
        recvbuf[0][0] = self.incoming

    def recv(self, buf=None, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG):
        for index, (sender, message_tag, payload) in enumerate(self.messages):
            if source not in (MPI.ANY_SOURCE, sender) or tag not in (MPI.ANY_TAG, message_tag):
                continue
            self.received.append((sender, message_tag))
            self.messages.pop(index)
            # A halo selected here must fail decoding, just as it did in the
            # real three-rank run; a mocked receive returning an object would
            # hide the protocol error regardless of the supplied tag.
            return MPI.pickle.loads(payload)
        raise AssertionError("No queued message matches the receive")


def make_grid(comm, monkeypatch):
    monkeypatch.setattr(
        config, "sim_config", SimpleNamespace(current_model=1, model_start=0, model_end=4)
    )
    grid = MPIGrid.__new__(MPIGrid)
    grid.comm = comm
    grid.global_size = np.array((60, 12, 12), dtype=np.int32)
    grid.mpi_tasks = np.array((3, 1, 1), dtype=np.int32)
    grid.neighbours = np.full((3, 2), -1, dtype=np.int32)
    if comm.rank > 0:
        grid.neighbours[Dim.X, Dir.NEG] = comm.rank - 1
    if comm.rank < 2:
        grid.neighbours[Dim.X, Dir.POS] = comm.rank + 1
    grid.calculate_local_extents()
    grid.srcsteps = np.array((2, 0, 0), dtype=np.int32)
    grid.rxsteps = grid.srcsteps.copy()
    for name in (
        "voltagesources",
        "hertziandipoles",
        "magneticdipoles",
        "transmissionlines",
        "rxs",
    ):
        setattr(grid, name, [])
    return grid


@pytest.mark.parametrize("object_type", (Rx, HertzianDipole, MagneticDipole))
def test_migration_ignores_halo_before_objects_from_multiple_ranks(monkeypatch, object_type):
    halo_tag = getattr(MPIGrid, "HALO_MESSAGE_TAG", 0)
    migration_tag = getattr(MPIGrid, "MIGRATION_MESSAGE_TAG", 1)
    assert halo_tag != migration_tag
    objects = []
    for x, origin in ((20, 18), (38, 40)):
        item = object_type()
        item.ID = f"object-{x}"
        item.coord = np.array((x, 6, 6), dtype=np.int32)
        item.coordorigin = np.array((origin, 6, 6), dtype=np.int32)
        objects.append(item)
    halo = (2, halo_tag, np.zeros(169, dtype=np.float64).tobytes())
    comm = QueuedComm(
        1,
        [
            halo,
            (0, migration_tag, MPI.pickle.dumps([objects[0]])),
            (2, migration_tag, MPI.pickle.dumps([objects[1]])),
        ],
        incoming=2,
    )
    grid = make_grid(comm, monkeypatch)

    grid.update_sources_and_recievers()

    assert comm.messages == [halo]
    assert comm.received == [(0, migration_tag), (2, migration_tag)]
    received = grid.rxs + grid.hertziandipoles + grid.magneticdipoles
    assert len(received) == 2
    for original, actual in zip(objects, received):
        assert actual.ID == original.ID
        np.testing.assert_array_equal(grid.local_to_global_coordinate(actual.coord), original.coord)
        np.testing.assert_array_equal(
            grid.local_to_global_coordinate(actual.coordorigin), original.coordorigin
        )


def test_outgoing_migration_uses_object_tag_and_preserves_origin(monkeypatch):
    comm = QueuedComm(0)
    grid = make_grid(comm, monkeypatch)
    receiver = Rx()
    receiver.ID = "moving"
    receiver.coord = np.array((18, 6, 6), dtype=np.int32)
    receiver.coordorigin = receiver.coord.copy()
    grid.rxs.append(receiver)

    grid.update_sources_and_recievers()

    assert len(comm.sent) == 1
    destination, tag, payload = comm.sent[0]
    assert destination == 1
    assert tag == getattr(MPIGrid, "MIGRATION_MESSAGE_TAG", 1)
    (sent,) = MPI.pickle.loads(payload)
    np.testing.assert_array_equal(sent.coord, (20, 6, 6))
    np.testing.assert_array_equal(sent.coordorigin, (18, 6, 6))
    assert grid.rxs == []


@pytest.mark.parametrize("dimension", (Dim.X, Dim.Y, Dim.Z))
@pytest.mark.parametrize("direction", (Dir.NEG, Dir.POS))
def test_halo_send_and_receive_use_only_the_halo_tag(dimension, direction):
    calls = []

    def send(buffer, destination, tag=0):
        calls.append(("send", destination, tag))
        return MPI.REQUEST_NULL

    def receive(buffer, source, tag=MPI.ANY_TAG):
        calls.append(("receive", source, tag))
        return MPI.REQUEST_NULL

    grid = MPIGrid.__new__(MPIGrid)
    grid.comm = SimpleNamespace(Isend=send, Irecv=receive)
    grid.neighbours = np.full((3, 2), -1, dtype=np.int32)
    grid.neighbours[dimension, direction] = 2
    grid.send_halo_map = np.full((3, 2), MPI.DOUBLE, dtype=object)
    grid.recv_halo_map = grid.send_halo_map.copy()
    grid.send_requests = []
    grid.recv_requests = []

    grid._halo_swap(np.zeros((2, 2, 2)), dimension, direction)

    tag = getattr(MPIGrid, "HALO_MESSAGE_TAG", 0)
    assert calls == [("send", 2, tag), ("receive", 2, tag)]
    assert len(grid.send_requests) == len(grid.recv_requests) == 1
