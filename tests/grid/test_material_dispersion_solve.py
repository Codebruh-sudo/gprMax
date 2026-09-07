"""Real propagation and setup rejection for magnetic/dielectric media."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from scipy.constants import c
from scipy.optimize import brentq

import gprMax
from gprMax.dispersion import spatial_resolution
from gprMax.materials import Material

pytestmark = pytest.mark.integration


def _scene(er=1, mr=4, dl=0.005, se=0, sm=0):
    scene = gprMax.Scene()
    for obj in (
        gprMax.Discretisation(p1=(dl, dl, dl)),
        gprMax.Domain(p1=(3.0, 0.02, 0.02)),
        gprMax.PMLThickness(thickness=(30, 0, 0, 30, 0, 0)),
        gprMax.SymmetryBoundary(face="y0", type="pmc"),
        gprMax.SymmetryBoundary(face="ymax", type="pmc"),
        gprMax.TimeWindow(time=4e-9),
        gprMax.OMPThreads(n=1),
        gprMax.Material(er=er, se=se, mr=mr, sm=sm, id="medium"),
        gprMax.Box(p1=(0, 0, 0), p2=(3.0, 0.02, 0.02), material_id="medium"),
        gprMax.Waveform(wave_type="ricker", amp=1, freq=1e9, id="pulse"),
        gprMax.Rx(p1=(0.95, 0.01, 0.01), id="near"),
        gprMax.Rx(p1=(0.975, 0.01, 0.01), id="far"),
    ):
        scene.add(obj)
    # A uniform electric-current sheet, PMC y faces and PEC z faces support
    # a TEM wave. Use a 3-D grid: explicit symmetry is not available in 2-D.
    for y in range(round(0.02 / dl) + 1):
        for z in range(round(0.02 / dl)):
            scene.add(
                gprMax.HertzianDipole(
                    p1=(0.8, y * dl, z * dl), polarisation="z", waveform_id="pulse"
                )
            )
    return scene


def _run(path, er=1, mr=4, dl=0.005, se=0, sm=0, **options):
    return gprMax.run(
        scenes=[_scene(er, mr, dl, se, sm)],
        outputfile=path,
        hide_progress_bars=True,
        log_level=40,
        cpu_precision="double",
        **options,
    )


def _check_phase(path, er, mr, dl):
    with h5py.File(path.with_suffix(".h5")) as data:
        receivers = {group.attrs["Name"]: group["Ez"][...] for group in data["rxs"].values()}
        near, far = receivers["near"], receivers["far"]
        dt = data.attrs["dt"]
    frequencies = np.asarray([0.8e9, 1e9, 1.2e9])
    kernel = np.exp(-2j * np.pi * frequencies[:, None] * np.arange(near.size) * dt)
    ratio = (kernel @ far) / (kernel @ near)
    observed = -np.angle(ratio) / 0.025
    continuum = 2 * np.pi * frequencies * np.sqrt(er * mr) / c
    numerical = []
    for frequency in frequencies:
        omega = 2 * np.pi * frequency
        velocity = c / np.sqrt(er * mr)
        rhs = (np.sin(omega * dt / 2) / (velocity * dt)) ** 2
        numerical.append(brentq(lambda k: (np.sin(k * dl / 2) / dl) ** 2 - rhs, 0, np.pi / dl))
    np.testing.assert_allclose(observed, numerical, rtol=2e-6, atol=0)
    np.testing.assert_allclose(np.abs(ratio), 1, rtol=2e-6, atol=0)
    material = Material(3, "medium")
    material.er, material.mr = er, mr
    grid = SimpleNamespace(materials=[material], dt=dt, dx=dl, dy=dl, dz=dl)
    diagnostic = spatial_resolution(grid, frequencies[-1], "3D")
    measured_velocity_error = 100 * (continuum[-1] / observed[-1] - 1)
    assert diagnostic["deltavp"] == pytest.approx(measured_velocity_error, abs=1e-5)
    return float(np.max(np.abs(observed / continuum - 1)))


@pytest.mark.parametrize("er,mr", ((1, 1), (4, 1), (1, 4)))
def test_lossless_phase_matches_yee_and_converges(tmp_path, er, mr):
    errors = []
    for index, dl in enumerate((0.005, 0.0025)):
        path = tmp_path / f"mesh{index}"
        _run(path, er, mr, dl)
        errors.append(_check_phase(path, er, mr, dl))
    assert errors[1] < 0.4 * errors[0]


def test_undersampled_magnetic_model_is_rejected_before_solving(tmp_path):
    with pytest.raises(ValueError, match="Insufficient spatial resolution.*medium"):
        _run(tmp_path / "unresolved", mr=100)
    assert not (tmp_path / "unresolved.h5").exists()


@pytest.mark.parametrize("magnetic_loss", (False, True))
def test_lossy_complex_propagation_converges_to_continuum(tmp_path, magnetic_loss):
    from scipy.constants import epsilon_0, mu_0

    # Match the electric and magnetic loss tangents for two independent cases.
    se = 0 if magnetic_loss else 0.02
    sm = 0.02 * mu_0 / (4 * epsilon_0) if magnetic_loss else 0
    frequencies = np.asarray([0.8e9, 1e9, 1.2e9])
    omega = 2 * np.pi * frequencies
    tangent = 0.02 / (omega * 4 * epsilon_0)
    factor = omega / c * np.sqrt(2)
    beta = factor * np.sqrt(np.sqrt(1 + tangent**2) + 1)
    alpha = factor * np.sqrt(np.sqrt(1 + tangent**2) - 1)
    continuum = beta - 1j * alpha
    errors = []
    for index, dl in enumerate((0.005, 0.0025)):
        path = tmp_path / f"loss{index}"
        _run(path, er=4, mr=1, dl=dl, se=se, sm=sm)
        with h5py.File(path.with_suffix(".h5")) as data:
            traces = {v.attrs["Name"]: v["Ez"][...] for v in data["rxs"].values()}
            dt = data.attrs["dt"]
        kernel = np.exp(-1j * omega[:, None] * np.arange(traces["near"].size) * dt)
        ratio = (kernel @ traces["far"]) / (kernel @ traces["near"])
        observed = 1j * np.log(ratio) / 0.025
        errors.append(float(np.max(np.abs((observed - continuum) / continuum))))
        np.testing.assert_allclose(observed.real, beta, rtol=0.004)
        np.testing.assert_allclose(-observed.imag, alpha, rtol=0.015)
    assert errors[1] < 0.4 * errors[0]


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ("cuda", "opencl"))
def test_device_magnetic_propagation_and_rejection(tmp_path, request, backend):
    device = request.getfixturevalue("gpu_device" if backend == "cuda" else "opencl_device")
    options = {"gpu" if backend == "cuda" else "opencl": [device], "gpu_precision": "double"}
    path = tmp_path / backend
    _run(path, **options)
    _check_phase(path, 1, 4, 0.005)
    with pytest.raises(ValueError, match="Insufficient spatial resolution.*medium"):
        _run(tmp_path / "unresolved", mr=100, **options)


@pytest.mark.parametrize("mr", (4, 100))
def test_mpi_magnetic_propagation_and_collective_rejection(tmp_path, mr):
    launcher = shutil.which("mpiexec", path=str(Path(sys.executable).parent))
    if launcher is None:
        pytest.skip("MPI runtime is not available")
    pytest.importorskip("mpi4py")
    path = tmp_path / "distributed"
    env = os.environ.copy()
    env.pop("MPI4PY_RC_FINALIZE", None)
    env.pop("FI_PROVIDER", None)
    env.update(
        PYTHONPATH=str(Path(__file__).resolve().parents[2]),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    result = subprocess.run(
        [launcher, "-n", "2", sys.executable, __file__, str(path), str(mr)],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    if mr == 4:
        assert result.returncode == 0, result.stdout + result.stderr
        _check_phase(path, 1, 4, 0.005)
    else:
        assert result.returncode != 0
        assert "Insufficient spatial resolution" in result.stdout + result.stderr
        assert not path.with_suffix(".h5").exists()


if __name__ == "__main__":
    _run(Path(sys.argv[1]), mr=int(sys.argv[2]), mpi=(2, 1, 1))
