"""Live terminal sources and last staggered components on real backends."""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

import gprMax

ROOT = Path(__file__).resolve().parents[2]
MPIEXEC = str(Path(sys.executable).with_name("mpiexec"))
if not Path(MPIEXEC).is_file():
    MPIEXEC = shutil.which("mpiexec")
COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
CASES = [(component, terminal) for component in COMPONENTS for terminal in (False, True)]
pytestmark = pytest.mark.integration


def _scene(component, terminal, invalid=False):
    axis = "xyz".index(component[1])
    boundary_axis = (axis + 1) % 3 if (component[0] == "E") == terminal else axis
    point = np.array((6, 6, 6))
    point[boundary_axis] = 12 if terminal or invalid else 11
    scene = gprMax.Scene()
    for obj in (
        gprMax.Domain(p1=(0.012,) * 3),
        gprMax.Discretisation(p1=(0.001,) * 3),
        gprMax.PMLThickness(thickness=0),
        gprMax.OMPThreads(1),
        gprMax.TimeWindow(iterations=100),
        gprMax.Waveform(wave_type="gaussian", amp=1, freq=2e10, id="pulse"),
        gprMax.Rx(p1=(0.005, 0.007, 0.008), id="remote"),
    ):
        scene.add(obj)
    if terminal:
        scene.add(gprMax.SymmetryBoundary(face="xyz"[boundary_axis] + "max", type="pmc"))
    constructor = gprMax.HertzianDipole if component[0] == "E" else gprMax.MagneticDipole
    scene.add(constructor(p1=tuple(point * 0.001), polarisation=component[1], waveform_id="pulse"))
    return scene


def _solve(output, precision, **backend):
    gprMax.run(
        scenes=[_scene(*case) for case in CASES],
        n=len(CASES),
        outputfile=output,
        cpu_precision=precision,
        hide_progress_bars=True,
        log_level=40,
        **backend
    )


def _read(output):
    fields = []
    for index in range(1, len(CASES) + 1):
        with h5py.File(output.with_name(output.name + str(index)).with_suffix(".h5")) as f:
            fields.append({component: f["rxs/rx1/" + component][...] for component in COMPONENTS})
    return fields


def _compare(actual, expected, precision):
    tolerance = 4e-5 if precision == "single" else 2e-12
    for case, values, reference in zip(CASES, actual, expected):
        for kind in "EH":
            scale = max(np.max(np.abs(reference[kind + axis])) for axis in "xyz")
            assert scale > 1e-10, (case, kind, "source must excite interior fields")
            for axis in "xyz":
                key = kind + axis
                assert values[key].shape == reference[key].shape == (100,)
                assert np.isfinite(values[key]).all()
                np.testing.assert_allclose(
                    values[key],
                    reference[key],
                    rtol=tolerance,
                    atol=tolerance * scale,
                    err_msg=str((case, key)),
                )


@pytest.fixture(scope="module")
def cpu_reference(tmp_path_factory):
    directory = tmp_path_factory.mktemp("bounds_cpu")
    cache = {}

    def reference(precision):
        if precision not in cache:
            output = directory / precision
            _solve(output, precision)
            cache[precision] = _read(output)
        return cache[precision]

    return reference


@pytest.mark.parametrize("precision", ("single", "double"))
def test_cpu_terminal_and_last_staggered_sources_are_live(cpu_reference, precision):
    values = cpu_reference(precision)
    _compare(values, values, precision)


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ("cuda", "opencl"))
@pytest.mark.parametrize("precision", ("single", "double"))
def test_accelerator_boundary_sources_match_cpu(
    tmp_path, request, cpu_reference, backend, precision
):
    device = request.getfixturevalue("gpu_device" if backend == "cuda" else "opencl_device")
    options = {"gpu" if backend == "cuda" else "opencl": [device], "gpu_precision": precision}
    output = tmp_path / "accelerator"
    _solve(output, precision, **options)
    _compare(_read(output), cpu_reference(precision), precision)


def _mpi_run(output, precision, partition, invalid=None):
    command = [
        MPIEXEC,
        "-n",
        str(np.prod(partition)),
        sys.executable,
        str(Path(__file__).resolve()),
        str(output),
        "--precision",
        precision,
        "--partition",
        *map(str, partition),
    ]
    if invalid:
        command += ["--invalid", invalid]
    environment = os.environ.copy()
    environment.pop("MPI4PY_RC_FINALIZE", None)
    environment.pop("FI_PROVIDER", None)
    environment.update(
        PYTHONPATH=str(ROOT), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=180)
    output.with_suffix(".log").write_text(result.stdout + result.stderr, encoding="utf-8")
    return result


@pytest.mark.skipif(
    MPIEXEC is None or importlib.util.find_spec("mpi4py") is None, reason="MPI unavailable"
)
@pytest.mark.parametrize("partition", ((2, 1, 1), (1, 2, 1), (1, 1, 2), (2, 2, 1)))
@pytest.mark.parametrize("precision", ("single", "double"))
def test_mpi_boundary_sources_match_cpu(tmp_path, cpu_reference, partition, precision):
    output = tmp_path / "mpi"
    result = _mpi_run(output, precision, partition)
    assert result.returncode == 0, result.stdout + result.stderr
    _compare(_read(output), cpu_reference(precision), precision)


@pytest.mark.skipif(
    MPIEXEC is None or importlib.util.find_spec("mpi4py") is None, reason="MPI unavailable"
)
@pytest.mark.parametrize("component", ("Ex", "Hy"))
def test_mpi_rejects_padded_global_component(tmp_path, component):
    result = _mpi_run(tmp_path / "invalid", "double", (2, 2, 1), invalid=component)
    assert result.returncode != 0
    assert "physical Yee component" in result.stdout + result.stderr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--partition", type=int, nargs=3, required=True)
    parser.add_argument("--invalid")
    args = parser.parse_args()
    if args.invalid:
        gprMax.run(
            scenes=[_scene(args.invalid, False, invalid=True)],
            outputfile=args.output,
            mpi=tuple(args.partition),
            cpu_precision=args.precision,
            geometry_only=True,
            hide_progress_bars=True,
            log_level=40,
        )
    else:
        _solve(args.output, args.precision, mpi=tuple(args.partition))
