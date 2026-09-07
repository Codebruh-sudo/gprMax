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

"""Global terminal-plane points have one MPI owner; positive rank halos do not."""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = Path(sys.executable).parent / "mpiexec"
MPIEXEC = str(LAUNCHER) if LAUNCHER.is_file() else shutil.which("mpiexec")
ITERATIONS = 100
COMPONENTS = tuple(kind + axis for kind in "EH" for axis in "xyz")
CASES = tuple(
    (axis, side, polarisation)
    for axis in range(3)
    for side in ("lower", "upper")
    for polarisation in "xyz"
    if polarisation != "xyz"[axis]
)
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        MPIEXEC is None or importlib.util.find_spec("mpi4py") is None,
        reason="requires mpi4py and an MPI launcher",
    ),
]


def _run(output, precision, partition=None, scan_axis=None, scan_direction=1):
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), str(output), "--precision", precision]
    if scan_axis is not None:
        command.extend(("--scan-axis", str(scan_axis), "--scan-direction", str(scan_direction)))
    if partition is not None:
        command = [
            MPIEXEC,
            "-n",
            str(np.prod(partition)),
            *command,
            "--partition",
            *map(str, partition),
        ]
    environment = os.environ.copy()
    environment.pop("MPI4PY_RC_FINALIZE", None)
    environment.pop("FI_PROVIDER", None)
    environment.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    environment.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    environment.update(
        PYTHONPATH=str(ROOT) + os.pathsep + environment.get("PYTHONPATH", ""),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
    )
    result = subprocess.run(
        command, cwd=output.parent, env=environment, capture_output=True, text=True, timeout=180
    )
    output.with_suffix(".log").write_text(result.stdout + result.stderr, encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Traceback" not in result.stdout + result.stderr
    return json.loads(output.with_suffix(".json").read_text())


def _assert_owners(captures, source_points, receiver_points):
    assert len(source_points) == len(receiver_points)
    assert captures
    assert all(len(rank) == len(source_points) for rank in captures)
    failures = []
    for model, (sources, receivers) in enumerate(zip(source_points, receiver_points)):
        for kind, expected in (("sources", sources), ("receivers", receivers)):
            actual = [item for rank in captures for item in rank[model][kind]]
            for name, position in expected.items():
                owners = [item for item in actual if item["name"] == name]
                if len(owners) != 1:
                    failures.append(f"model {model + 1}, {kind}/{name}: {len(owners)} owners")
                elif owners[0]["global"] != list(position):
                    failures.append(
                        f"model {model + 1}, {kind}/{name}: wrong global point {owners[0]['global']}"
                    )
            if len(actual) != len(expected):
                failures.append(
                    f"model {model + 1}, {kind}: expected {len(expected)}, found {len(actual)}"
                )
    assert not failures, "\n".join(failures)


def _compare_outputs(reference_path, actual_path, precision, component):
    with h5py.File(reference_path) as reference, h5py.File(actual_path) as actual:
        expected_receivers = {str(rx.attrs["Name"]): rx for rx in reference["rxs"].values()}
        actual_receivers = {str(rx.attrs["Name"]): rx for rx in actual["rxs"].values()}
        assert actual_receivers.keys() == expected_receivers.keys()
        assert len(actual["srcs"]) == len(reference["srcs"])
        tolerance = 3e-5 if precision == "single" else 2e-12
        for name, receiver in expected_receivers.items():
            observed = actual_receivers[name]
            for attribute in ("Position", "GridPosition"):
                np.testing.assert_array_equal(observed.attrs[attribute], receiver.attrs[attribute])
            for field in COMPONENTS:
                expected, values = receiver[field][...], observed[field][...]
                assert expected.shape == values.shape == (ITERATIONS,)
                assert np.isfinite(expected).all() and np.isfinite(values).all()
                # A component can vanish by symmetry. Its absolute tolerance
                # uses only same-unit E or H components of this receiver.
                scale = max(float(np.max(np.abs(receiver[field[0] + axis][...]))) for axis in "xyz")
                np.testing.assert_allclose(
                    values, expected, rtol=tolerance, atol=tolerance * max(scale, 1e-30)
                )
        assert np.linalg.norm(expected_receivers["remote"][component][...]) > 1e-8


def _points(axis, side):
    source = [10, 10, 10]
    source[axis] = 0 if side == "lower" else 20
    remote = source.copy()
    remote[axis] += 6 if side == "lower" else -6
    return source, remote


@pytest.mark.parametrize(
    "partition",
    [(2, 1, 1), (1, 2, 1), (1, 1, 2), (2, 2, 1)],
    ids=("split-x", "split-y", "split-z", "split-xy"),
)
@pytest.mark.parametrize("precision", ["single", "double"])
def test_pmc_terminal_sources_and_receivers_have_unique_owners(tmp_path, partition, precision):
    serial, distributed = tmp_path / "serial", tmp_path / "mpi"
    _run(serial, precision)
    captures = _run(distributed, precision, partition)
    source_points, receiver_points = [], []
    for axis, side, _ in CASES:
        source, remote = _points(axis, side)
        source_points.append({"drive": source, "seam": [10, 10, 10]})
        receiver_points.append({"face": source, "remote": remote, "seam": [10, 10, 10]})
    _assert_owners(captures, source_points, receiver_points)
    for index, (_, _, polarisation) in enumerate(CASES, start=1):
        _compare_outputs(
            f"{serial}{index}.h5", f"{distributed}{index}.h5", precision, "E" + polarisation
        )


@pytest.mark.parametrize("axis", range(3), ids=("scan-x", "scan-y", "scan-z"))
@pytest.mark.parametrize("direction", (1, -1), ids=("arrive", "depart"))
def test_geometry_fixed_sources_and_receivers_reach_upper_pmc_plane(tmp_path, axis, direction):
    serial, distributed = tmp_path / "serial", tmp_path / "mpi"
    partition = [1, 1, 1]
    partition[axis] = 2
    _run(serial, "double", scan_axis=axis, scan_direction=direction)
    captures = _run(distributed, "double", partition, scan_axis=axis, scan_direction=direction)
    source_points, receiver_points = [], []
    coordinates = (8, 14, 20) if direction == 1 else (20, 14, 8)
    for coordinate in coordinates:
        point = [10, 10, 10]
        point[axis] = coordinate
        remote = point.copy()
        remote[axis] -= 2
        source_points.append({"drive": point})
        receiver_points.append({"face": point, "remote": remote})
    _assert_owners(captures, source_points, receiver_points)
    component = "E" + "xyz"[(axis + 1) % 3]
    for index in range(1, 4):
        _compare_outputs(f"{serial}{index}.h5", f"{distributed}{index}.h5", "double", component)
        # Each stepped serial result is also checked against a rebuilt scene.
        _compare_outputs(f"{serial}{index}.h5", f"{serial}_fresh{index}.h5", "double", component)


def _scene(axis, side, polarisation, scan_coordinate=None):
    import gprMax

    source, remote = _points(axis, side)
    if scan_coordinate is not None:
        source[axis] = scan_coordinate
        remote = source.copy()
        remote[axis] -= 2

    def metres(point):
        return tuple(float(value) * 0.001 for value in point)

    scene = gprMax.Scene()
    for obj in (
        gprMax.Domain(p1=(0.020,) * 3),
        gprMax.Discretisation(p1=(0.001,) * 3),
        gprMax.PMLThickness(thickness=0),
        gprMax.OMPThreads(1),
        gprMax.TimeWindow(iterations=ITERATIONS),
        gprMax.SymmetryBoundary(face="xyz"[axis] + ("0" if side == "lower" else "max"), type="pmc"),
        gprMax.Waveform(wave_type="ricker", amp=1, freq=1.5e10, id="drive"),
        gprMax.HertzianDipole(p1=metres(source), polarisation=polarisation, waveform_id="drive"),
        gprMax.Rx(p1=metres(source), id="face"),
        gprMax.Rx(p1=metres(remote), id="remote"),
    ):
        scene.add(obj)
    if scan_coordinate is None:
        # This point lies on every split plane (and at a four-rank seam in
        # split-xy). A transparent zero drive tests ownership without loading.
        scene.add(gprMax.Waveform(wave_type="ricker", amp=0, freq=1.5e10, id="seam"))
        scene.add(
            gprMax.HertzianDipole(p1=(0.010,) * 3, polarisation=polarisation, waveform_id="seam")
        )
        scene.add(gprMax.Rx(p1=(0.010,) * 3, id="seam"))
    return scene


def _worker(args):
    import gprMax
    from gprMax.model import Model

    comm = None
    if args.partition is not None:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    captures = []
    original_build = Model.build

    def capture(model):
        result = original_build(model)
        grid = model.G

        def record(item, name):
            coordinate = (
                grid.local_to_global_coordinate(item.coord) if comm is not None else item.coord
            )
            return {
                "name": name,
                "global": list(map(int, coordinate)),
                "local": list(map(int, item.coord)),
            }

        captures.append(
            {
                "sources": [record(source, source.waveformID) for source in grid.hertziandipoles],
                "receivers": [record(receiver, receiver.ID) for receiver in grid.rxs],
            }
        )
        return result

    Model.build = capture
    options = dict(
        outputfile=Path(args.output),
        mpi=tuple(args.partition) if args.partition is not None else None,
        cpu_precision=args.precision,
        hide_progress_bars=True,
        log_level=40,
    )
    if args.scan_axis is None:
        scenes = [_scene(*case) for case in CASES]
        gprMax.run(scenes=scenes, n=len(scenes), **options)
    else:
        axis = args.scan_axis
        polarisation = "xyz"[(axis + 1) % 3]
        coordinates = (8, 14, 20) if args.scan_direction == 1 else (20, 14, 8)
        scene = _scene(axis, "upper", polarisation, scan_coordinate=coordinates[0])
        step = [0.0, 0.0, 0.0]
        step[axis] = 0.006 * args.scan_direction
        scene.add(gprMax.SrcSteps(p1=tuple(step)))
        scene.add(gprMax.RxSteps(p1=tuple(step)))
        gprMax.run(scenes=[scene], n=3, geometry_fixed=True, **options)
        if comm is None:
            for index, coordinate in enumerate(coordinates, start=1):
                fresh = dict(options, outputfile=Path(f"{args.output}_fresh{index}"))
                gprMax.run(
                    scenes=[_scene(axis, "upper", polarisation, scan_coordinate=coordinate)],
                    **fresh,
                )
    gathered = comm.gather(captures, root=0) if comm is not None else [captures]
    if comm is None or comm.rank == 0:
        Path(args.output).with_suffix(".json").write_text(
            json.dumps(gathered, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--precision", choices=("single", "double"), required=True)
    parser.add_argument("--partition", nargs=3, type=int)
    parser.add_argument("--scan-axis", choices=range(3), type=int)
    parser.add_argument("--scan-direction", choices=(1, -1), type=int, default=1)
    args = parser.parse_args()
    try:
        _worker(args)
    except BaseException:
        if args.partition is not None:
            from mpi4py import MPI

            traceback.print_exc()
            MPI.COMM_WORLD.Abort(1)
        raise
