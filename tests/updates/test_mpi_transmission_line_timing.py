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

from gprMax.fields_outputs import Ix, Iy, Iz
from gprMax.solvers import Solver
from gprMax.sources import TransmissionLine
from gprMax.updates.cpu_updates import CPUUpdates
from gprMax.updates.mpi_updates import MPIUpdates


class _RecordingSource:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls
        self.invocations = []

    def update_magnetic(self, *args):
        self.calls.append(self.name)
        self.invocations.append(("serial", args))

    def update_magnetic_mpi(self, *args):
        self.calls.append(self.name)
        self.invocations.append(("mpi", args))


@pytest.mark.parametrize("updates_type", (CPUUpdates, MPIUpdates))
def test_cpu_mpi_separate_magnetic_writers_from_transmission_line_sampling(updates_type):
    calls = []
    grid = SimpleNamespace(
        transmissionlines=[_RecordingSource("transmission_line", calls)],
        magneticdipoles=[_RecordingSource("magnetic_dipole", calls)],
        magneticfrillsources=[_RecordingSource("magnetic_frill", calls)],
        updatecoeffsH=None,
        ID=None,
        Hx=None,
        Hy=None,
        Hz=None,
    )
    updates = updates_type.__new__(updates_type)
    updates.grid = grid

    updates.update_magnetic_sources(3)
    assert calls == ["magnetic_dipole", "magnetic_frill"]

    updates.update_magnetic_edge_devices(3)
    assert calls == ["magnetic_dipole", "magnetic_frill", "transmission_line"]
    expected_args = (3, None, None, None, None, None, grid)
    assert grid.magneticdipoles[0].invocations == [("serial", expected_args)]
    frill_method = "mpi" if updates_type is MPIUpdates else "serial"
    assert grid.magneticfrillsources[0].invocations == [(frill_method, expected_args)]
    assert grid.transmissionlines[0].invocations == [("serial", expected_args)]


@pytest.mark.parametrize("updates_type", (CPUUpdates, MPIUpdates))
@pytest.mark.parametrize("polarisation, contour", (("x", Ix), ("y", Iy), ("z", Iz)))
@pytest.mark.unit
def test_edge_device_samples_current_h_without_writing_fields(updates_config, updates_type, polarisation, contour):
    """The shared dispatch reads H and advances the line at the same step."""
    iteration = 3
    dt = 1e-12
    line = TransmissionLine(iterations=30, dt=dt)
    line.polarisation = polarisation
    line.xcoord = line.ycoord = line.zcoord = 2
    line.resistance = 50
    line.start = iteration * dt
    line.stop = iteration * dt
    line.waveformvalues_halfdt = np.zeros(31)
    line.waveformvalues_halfdt[iteration] = 1
    fields = [np.arange(64, dtype=float).reshape((4,) * 3) * scale for scale in (1, 2, 3)]
    before = [field.copy() for field in fields]
    grid = SimpleNamespace(
        transmissionlines=[line],
        magneticdipoles=[],
        magneticfrillsources=[],
        updatecoeffsH=None,
        ID=None,
        Hx=fields[0],
        Hy=fields[1],
        Hz=fields[2],
        dt=dt,
        dx=0.001,
        dy=0.002,
        dz=0.003,
    )
    setattr(grid, f"calculate_I{polarisation}", lambda i, j, k: contour(i, j, k, *fields, grid))
    expected_current = contour(2, 2, 2, *fields, grid)
    assert expected_current != 0
    updates = updates_type.__new__(updates_type)
    updates.grid = grid

    updates.update_magnetic_sources(iteration)
    np.testing.assert_array_equal(line.current, 0)
    updates.update_magnetic_edge_devices(iteration)

    assert line.current[line.antpos] == pytest.approx(expected_current)
    assert line.current[line.srcpos - 1] > 0
    for actual, original in zip(fields, before):
        np.testing.assert_array_equal(actual, original)


def test_solver_samples_mpi_transmission_lines_after_magnetic_halo():
    calls = []
    updates = MPIUpdates.__new__(MPIUpdates)
    updates.grid = SimpleNamespace(iterations=1)

    def record(name):
        return lambda *args, **kwargs: calls.append(name)

    for name in (
        "time_start",
        "store_outputs",
        "store_snapshots",
        "observe_ntff_electric",
        "update_magnetic",
        "update_magnetic_pml",
        "update_magnetic_sources",
        "update_eigenmode_sources_magnetic",
        "update_plane_waves_magnetic",
        "observe_eigenmode_ports",
        "halo_swap_magnetic",
        "update_magnetic_edge_devices",
        "observe_ntff_magnetic",
        "update_electric_a",
        "update_symmetry_boundaries_electric",
        "update_electric_pml",
        "update_electric_sources",
        "update_eigenmode_sources_electric",
        "update_plane_waves_electric",
        "update_symmetry_boundaries_electric_b",
        "update_electric_b",
        "update_network_terminals",
        "halo_swap_electric",
        "finalise",
        "cleanup",
    ):
        setattr(updates, name, record(name))
    updates.calculate_solve_time = lambda: 0.0

    Solver(updates).solve(range(1))

    assert calls.index("update_magnetic_sources") < calls.index("halo_swap_magnetic")
    assert calls.index("update_eigenmode_sources_magnetic") < calls.index("halo_swap_magnetic")
    assert calls.index("update_plane_waves_magnetic") < calls.index("halo_swap_magnetic")
    assert calls.index("halo_swap_magnetic") < calls.index("update_magnetic_edge_devices")
    assert calls.count("update_magnetic_edge_devices") == 1
    assert calls.index("update_magnetic_edge_devices") < calls.index("observe_ntff_magnetic")
    assert calls.index("update_magnetic_edge_devices") < calls.index("update_electric_a")
    assert calls.index("update_electric_a") < calls.index("update_symmetry_boundaries_electric")
    assert calls.index("update_symmetry_boundaries_electric") < calls.index("update_electric_pml")
    assert calls.index("update_plane_waves_electric") < calls.index("update_symmetry_boundaries_electric_b")
    assert calls.index("update_symmetry_boundaries_electric_b") < calls.index("update_electric_b")


def test_mpi_finalise_completes_last_halo_exchange():
    calls = []
    updates = MPIUpdates.__new__(MPIUpdates)
    updates.grid = SimpleNamespace(
        complete_halo_swaps=lambda: calls.append("complete_halo_swaps"),
        ntff_monitors=[],
    )

    updates.finalise()

    assert calls == ["complete_halo_swaps"]
