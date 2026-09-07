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

from types import SimpleNamespace

import numpy as np
import pytest

from gprMax.ports import RationalNetworkPortOutput, VoltageSourcePortMonitor
from gprMax.sources import VoltageSource
from gprMax.user_objects.cmds_multiuse import _reserve_port_output_id


def test_mpi_port_id_reservation_keeps_coordinator_owner():
    grid = SimpleNamespace(mpi_port_output_ids=[], mpi_port_output_owners={})
    first = SimpleNamespace()
    second = SimpleNamespace()

    assert _reserve_port_output_id(grid, None, first) == "port1"
    assert _reserve_port_output_id(grid, "feed", second) == "feed"
    assert grid.mpi_port_output_owners == {"port1": first, "feed": second}


def test_voltage_port_rebind_uses_gathered_global_source_and_receiver():
    # The old source can still have rank-local coordinates. Only gathered
    # source/receiver coordinates may be compared with one another.
    old_source = SimpleNamespace(port_id="feed", polarisation="z", coord=np.asarray((0, 9, 10)))
    old_receiver = SimpleNamespace()
    monitor = VoltageSourcePortMonitor("feed", old_source, old_receiver, 10)
    source = SimpleNamespace(port_id="feed", polarisation="z", coord=np.asarray((8, 9, 10)))
    receiver = SimpleNamespace(
        internal=True,
        port_id="feed",
        coord=np.asarray((8, 9, 10)),
    )
    grid = SimpleNamespace(
        rxs=[receiver],
        voltagesources=[source],
        dx=0.1,
        dy=0.2,
        dz=0.3,
    )

    monitor.rebind_after_mpi_gather(grid)

    assert monitor.source is source
    assert monitor.receiver is receiver
    np.testing.assert_allclose(monitor.source_position, (0.8, 1.8, 3.0))


def _coincident_voltage_ports(port_ids):
    sources, receivers, monitors = [], [], []
    for port_id in port_ids:
        source = VoltageSource()
        source.port_id = port_id
        source.polarisation = "z"
        source.coord = np.asarray((8, 9, 10))
        receiver = SimpleNamespace(internal=True, port_id=port_id, coord=source.coord.copy())
        # Separate objects model the independently gathered monitor payload.
        old_source = SimpleNamespace(port_id=port_id, polarisation="z", coord=source.coord.copy())
        monitors.append(VoltageSourcePortMonitor(port_id, old_source, object(), 10))
        sources.append(source)
        receivers.append(receiver)
    grid = SimpleNamespace(rxs=receivers, voltagesources=sources, dx=0.1, dy=0.2, dz=0.3)
    return grid, monitors


@pytest.mark.parametrize("port_ids", [("feed_a", "feed_b"), ("port1", "port2")])
@pytest.mark.parametrize("reverse_sources", [False, True])
def test_coincident_voltage_ports_rebind_by_identity(port_ids, reverse_sources):
    grid, monitors = _coincident_voltage_ports(port_ids)
    expected_sources = dict(zip(port_ids, grid.voltagesources))
    expected_receivers = dict(zip(port_ids, grid.rxs))
    if reverse_sources:
        grid.voltagesources.reverse()
    for monitor in monitors:
        monitor.rebind_after_mpi_gather(grid)
        assert monitor.source is expected_sources[monitor.output_id]
        assert monitor.receiver is expected_receivers[monitor.output_id]
        assert grid.voltagesources[monitor.source_index - 1] is monitor.source


@pytest.mark.parametrize("kind", ["source", "receiver"])
@pytest.mark.parametrize("count", [0, 2])
def test_voltage_port_rebind_rejects_missing_or_duplicate_identity(kind, count):
    grid, (monitor,) = _coincident_voltage_ports(("feed",))
    objects = grid.voltagesources if kind == "source" else grid.rxs
    if count == 0:
        # A geometrically matching object with a different ID is not a fallback.
        objects[0].port_id = "other"
    else:
        objects.append(objects[0])
    with pytest.raises(RuntimeError, match="could not uniquely rebind"):
        monitor.rebind_after_mpi_gather(grid)


@pytest.mark.parametrize("mismatch", ["polarisation", "position"])
def test_voltage_port_rebind_rejects_inconsistent_edge(mismatch):
    grid, (monitor,) = _coincident_voltage_ports(("feed",))
    old_source, old_receiver = monitor.source, monitor.receiver
    if mismatch == "polarisation":
        grid.voltagesources[0].polarisation = "x"
    else:
        grid.rxs[0].coord[0] += 1
    with pytest.raises(RuntimeError, match="inconsistent MPI source/receiver edge"):
        monitor.rebind_after_mpi_gather(grid)
    assert monitor.source is old_source
    assert monitor.receiver is old_receiver


def test_network_port_rebind_uses_gathered_global_terminal():
    monitor = RationalNetworkPortOutput.__new__(RationalNetworkPortOutput)
    monitor.output_id = "load"
    terminal = SimpleNamespace(ID="load", coord=np.asarray((4, 5, 6)), output=None)
    grid = SimpleNamespace(networkterminals=[terminal], dx=0.1, dy=0.2, dz=0.3)

    monitor.rebind_after_mpi_gather(grid)

    assert monitor.terminal is terminal
    assert monitor.source is terminal
    assert terminal.output is monitor
    np.testing.assert_allclose(monitor.source_position, (0.4, 1.0, 1.8))
