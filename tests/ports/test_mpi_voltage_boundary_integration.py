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

"""Real MPI regressions for voltage ports at and away from internal rank faces."""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


def _mpi_launcher():
    environment_launcher = Path(sys.executable).parent / "mpiexec"
    if environment_launcher.is_file():
        return str(environment_launcher)
    return shutil.which("mpiexec")


def _run(command, directory):
    environment = os.environ.copy()
    # These subprocesses are complete MPI applications, so allow mpi4py to
    # finalise MPI normally. An inherited MPI4PY_RC_FINALIZE=0 makes Open MPI
    # report an otherwise successful run as an abnormal rank exit.
    environment.pop("MPI4PY_RC_FINALIZE", None)
    environment.update(
        {
            "FI_PROVIDER": "shm",
            "OMP_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    completed = subprocess.run(
        command,
        cwd=directory,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.integration
@pytest.mark.skipif(
    _mpi_launcher() is None or importlib.util.find_spec("mpi4py") is None,
    reason="requires mpi4py and an MPI launcher",
)
@pytest.mark.parametrize("polarisation", ["x", "y", "z"])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("named", [False, True], ids=["automatic-ids", "named-ids"])
@pytest.mark.parametrize("on_face", [False, True], ids=["rank-interior", "rank-face"])
def test_coincident_voltage_ports_match_serial_across_reused_geometry(
    tmp_path, polarisation, precision, named, on_face
):
    """Check existing ordered-source semantics, not a physical two-port model.

    The pair is owned by the coordinator in the interior case, and by the
    other rank on its lower internal face in the boundary case. Voltage ports
    stay fixed at their material edge during geometry reuse. Distinct drives
    and resistances expose swapped identities although both sample one edge.
    """
    split_axis = ("xyz".index(polarisation) + 1) % 3
    point = [0.006] * 3
    point[split_axis] = 0.006 if on_face else 0.005
    partition = [1] * 3
    partition[split_axis] = 2
    position = " ".join(map(str, point))
    port_ids = ("feed_a", "feed_b") if named else ("port1", "port2")
    suffixes = [f" 0 1e-9 {port_id} 10" if named else "" for port_id in port_ids]
    model = tmp_path / "coincident.in"
    model.write_text(
        "\n".join(
            (
                "#domain: 0.012 0.012 0.012",
                "#dx_dy_dz: 0.001 0.001 0.001",
                "#time_window: 160",
                "#pml_cells: 0",
                "#omp_threads: 1",
                "#waveform: ricker 0.7 2e10 first",
                "#waveform: ricker -0.4 1.5e10 second",
                f"#voltage_source: {polarisation} {position} 50 first{suffixes[0]}",
                f"#voltage_source: {polarisation} {position} 75 second{suffixes[1]}",
                "#rx: 0.008 0.008 0.008",
                "",
            )
        ),
        encoding="utf-8",
    )
    common = (
        sys.executable,
        "-m",
        "gprMax",
        str(model),
        "--hide-progress-bars",
        "-cpu_precision",
        precision,
        "-n",
        "3",
        "--geometry-fixed",
    )
    serial_output, mpi_output = tmp_path / "serial", tmp_path / "mpi"
    _run((*common, "-o", str(serial_output)), tmp_path)
    _run(
        (
            _mpi_launcher(),
            "-n",
            "2",
            *common,
            "--mpi",
            *map(str, partition),
            "-o",
            str(mpi_output),
        ),
        tmp_path,
    )
    tolerance = 3e-5 if precision == "single" else 2e-12
    for run in range(1, 4):
        with h5py.File(f"{serial_output}{run}.h5") as serial, h5py.File(
            f"{mpi_output}{run}.h5"
        ) as mpi:
            assert set(serial["ports"]) == set(mpi["ports"]) == set(port_ids)
            for port_id, resistance in zip(port_ids, (50, 75)):
                reference, actual = serial[f"ports/{port_id}"], mpi[f"ports/{port_id}"]
                assert reference.attrs["ReferenceImpedance"] == resistance
                for attribute in (
                    "Name",
                    "Position",
                    "GridPosition",
                    "Polarisation",
                    "SourceIndex",
                    "ReferenceImpedance",
                    "WaveformID",
                ):
                    np.testing.assert_array_equal(
                        actual.attrs[attribute], reference.attrs[attribute]
                    )
                np.testing.assert_allclose(actual.attrs["Position"], point, rtol=1e-7)
                assert set(actual) == set(reference)
                assert np.any(reference["valid_S11"][...])
                assert np.max(np.abs(reference["Vtotal"][...])) > 0
                for dataset in reference:
                    expected, values = reference[dataset][...], actual[dataset][...]
                    # Invalid spectral bins deliberately contain NaNs. Compare
                    # masks exactly and use each quantity's own finite scale.
                    np.testing.assert_array_equal(np.isfinite(values), np.isfinite(expected))
                    if np.issubdtype(expected.dtype, np.integer):
                        np.testing.assert_array_equal(values, expected)
                    else:
                        scale = np.max(np.abs(expected[np.isfinite(expected)]), initial=0.0)
                        np.testing.assert_allclose(
                            values,
                            expected,
                            rtol=tolerance,
                            atol=tolerance * max(float(scale), 1e-30),
                            equal_nan=True,
                            err_msg=f"run {run}, {port_id}/{dataset}",
                        )
            assert not np.array_equal(
                mpi[f"ports/{port_ids[0]}/Vgenerator"][...],
                mpi[f"ports/{port_ids[1]}/Vgenerator"][...],
            )
            for component in (kind + axis for kind in "EH" for axis in "xyz"):
                expected = serial[f"rxs/rx1/{component}"][...]
                assert np.isfinite(expected).all()
                np.testing.assert_allclose(
                    mpi[f"rxs/rx1/{component}"][...],
                    expected,
                    rtol=tolerance,
                    atol=tolerance * max(float(np.max(np.abs(expected))), 1e-30),
                )
            assert np.linalg.norm(serial[f"rxs/rx1/E{polarisation}"][...]) > 0


@pytest.mark.integration
@pytest.mark.skipif(_mpi_launcher() is None, reason="mpiexec is not installed")
def test_hard_voltage_port_on_internal_rank_boundary_matches_serial(tmp_path):
    """The positive rank must use its halo, not mistake its face for x=0."""

    model = tmp_path / "hard_port.in"
    model.write_text(
        "\n".join(
            (
                "#title: hard voltage port at MPI split",
                "#domain: 0.04 0.02 0.02",
                "#dx_dy_dz: 0.002 0.002 0.002",
                "#time_window: 1e-9",
                "#pml_cells: 2",
                "#waveform: ricker 1 1e9 pulse",
                # The 2x1x1 decomposition splits x at 0.02 m. For a z-directed
                # hard source, x is transverse to the Ampere current loop.
                "#voltage_source: z 0.02 0.01 0.01 0 pulse 0 1e-9 feed nyquist 50",
                "",
            )
        ),
        encoding="utf-8",
    )

    common = (sys.executable, "-m", "gprMax", str(model), "--hide-progress-bars")
    serial_output = tmp_path / "serial"
    mpi_output = tmp_path / "mpi"
    _run((*common, "-o", str(serial_output)), tmp_path)
    _run(
        (
            _mpi_launcher(),
            "-n",
            "2",
            *common,
            "--mpi",
            "2",
            "1",
            "1",
            "-o",
            str(mpi_output),
        ),
        tmp_path,
    )

    with h5py.File(serial_output.with_suffix(".h5")) as serial, h5py.File(
        mpi_output.with_suffix(".h5")
    ) as mpi:
        for dataset in ("S11", "Zin", "Yin", "Vtotal", "Iloop"):
            reference = serial[f"ports/feed/{dataset}"][...]
            np.testing.assert_allclose(
                mpi[f"ports/feed/{dataset}"][...],
                reference,
                rtol=2e-5,
                atol=2e-6 * max(float(np.nanmax(np.abs(reference), initial=0.0)), 1e-18),
                equal_nan=True,
            )
