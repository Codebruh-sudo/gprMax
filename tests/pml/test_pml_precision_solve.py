"""Real CPU/device/MPI solves with non-binary-exact PML grid spacings."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

import gprMax

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = Path(sys.executable).parent / "mpiexec"
MPIEXEC = str(LAUNCHER) if LAUNCHER.is_file() else shutil.which("mpiexec")
COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def _scene(formulation, order):
    dl = np.array((0.001, 0.0016, 0.0013))
    domain = tuple(dl * (24, 20, 18))
    scene = gprMax.Scene()
    for obj in (
        gprMax.Discretisation(p1=tuple(dl)),
        gprMax.Domain(p1=domain),
        gprMax.PMLThickness(thickness=3),
        gprMax.PMLFormulation(formulation=formulation),
        gprMax.TimeWindow(iterations=180),
        gprMax.OMPThreads(n=1),
        gprMax.Material(er=2.5, se=0.025, mr=1.3, sm=0.002, id="background"),
        gprMax.Box(p1=(0, 0, 0), p2=domain, material_id="background"),
        gprMax.Waveform(wave_type="ricker", amp=0.001, freq=8e9, id="pulse"),
        gprMax.HertzianDipole(p1=tuple(dl * (11, 9, 8)), polarisation="z", waveform_id="pulse"),
        # A second orientation gives all six receiver components a signal;
        # a lone z dipole leaves Hz at roundoff level in this uniform medium.
        gprMax.HertzianDipole(p1=tuple(dl * (9, 7, 9)), polarisation="x", waveform_id="pulse"),
        gprMax.Rx(p1=tuple(dl * (15, 12, 10))),
    ):
        scene.add(obj)
    for pole in range(order):
        scene.add(
            gprMax.PMLCFS(
                alphascalingprofile="constant",
                alphascalingdirection="forward",
                alphamin=0.2 if pole else 0.001,
                alphamax=0.2 if pole else 0.001,
                kappascalingprofile="linear",
                kappascalingdirection="forward",
                kappamin=1,
                kappamax=1.2 + 0.3 * pole,
                sigmascalingprofile="linear",
                sigmascalingdirection="forward",
                sigmamin=0,
                sigmamax=0.1 * (pole + 1),
            )
        )
    return scene


def _run(output, formulation, order, precision, **options):
    gprMax.run(
        scenes=[_scene(formulation, order)],
        outputfile=output,
        cpu_precision=precision,
        gpu_precision=precision,
        hide_progress_bars=True,
        log_level=40,
        **options,
    )


def _compare(reference, actual, precision, record_property):
    with h5py.File(reference.with_suffix(".h5")) as a, h5py.File(actual.with_suffix(".h5")) as b:
        for component in COMPONENTS:
            expected, result = a[f"rxs/rx1/{component}"][...], b[f"rxs/rx1/{component}"][...]
            dtype = np.float32 if precision == "single" else np.float64
            assert expected.dtype == result.dtype == dtype
            assert np.isfinite(expected).all() and np.isfinite(result).all()
            scale = np.max(np.abs(expected))
            assert scale > 0
            error = np.max(np.abs(result - expected)) / scale
            record_property(f"{component}_peak_relative_error", float(error))
            # Scale by the trace peak, not individual values near zero crossings.
            assert error < (3e-5 if precision == "single" else 2e-12), (component, error)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("backend", ("cuda", "opencl"))
@pytest.mark.parametrize("precision", ("single", "double"))
@pytest.mark.parametrize("formulation", ("HORIPML", "MRIPML"))
@pytest.mark.parametrize("order", (1, 2))
def test_pml_cpu_matches_device(
    tmp_path, request, record_property, backend, precision, formulation, order
):
    device = request.getfixturevalue("gpu_device" if backend == "cuda" else "opencl_device")
    options = {"gpu" if backend == "cuda" else "opencl": [device]}
    reference, actual = tmp_path / "cpu", tmp_path / backend
    _run(reference, formulation, order, precision)
    _run(actual, formulation, order, precision, **options)
    _compare(reference, actual, precision, record_property)


@pytest.mark.skipif(
    MPIEXEC is None or not h5py.get_config().mpi, reason="requires MPI and parallel HDF5"
)
@pytest.mark.parametrize("partition", ((2, 1, 1), (1, 2, 1), (1, 1, 2)))
@pytest.mark.parametrize("precision", ("single", "double"))
@pytest.mark.parametrize("formulation", ("HORIPML", "MRIPML"))
def test_pml_cpu_matches_real_mpi(tmp_path, record_property, partition, precision, formulation):
    reference, actual = tmp_path / "cpu", tmp_path / "mpi"
    _run(reference, formulation, 2, precision)
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""),
        MPI4PY_RC_INITIALIZE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OMPI_MCA_rmaps_base_oversubscribe="1",
    )
    result = subprocess.run(
        [
            MPIEXEC,
            "-n",
            "2",
            sys.executable,
            str(Path(__file__).resolve()),
            str(actual),
            "--formulation",
            formulation,
            "--precision",
            precision,
            "--partition",
            *map(str, partition),
        ],
        env=env,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    _compare(reference, actual, precision, record_property)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--formulation", choices=("HORIPML", "MRIPML"), required=True)
    parser.add_argument("--precision", choices=("single", "double"), required=True)
    parser.add_argument("--partition", nargs=3, type=int, required=True)
    args = parser.parse_args()
    _run(args.output, args.formulation, 2, args.precision, mpi=tuple(args.partition))
