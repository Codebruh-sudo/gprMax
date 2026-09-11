"""End-to-end initial hard fields, applied-time gates and output timing."""

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import numpy as np
import pytest

import gprMax

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
DL = 0.001
COMPONENTS = tuple(kind + axis for kind in "EHI" for axis in "xyz")
BACKENDS = ["cpu"] + [pytest.param(name, marks=pytest.mark.gpu) for name in ("cuda", "opencl", "metal")]


def backend_options(request, backend, precision):
    if backend == "cpu":
        return {"cpu_precision": precision}
    if backend == "metal":
        if precision == "double":
            pytest.skip("Metal supports single precision only")
        metal = pytest.importorskip("Metal")
        if metal.MTLCreateSystemDefaultDevice() is None:
            pytest.skip("No Metal device")
        return {"metal": True, "gpu_precision": precision}
    device = request.getfixturevalue("gpu_device" if backend == "cuda" else "opencl_device")
    return {"gpu" if backend == "cuda" else "opencl": [device], "gpu_precision": precision}


def scene(axis, case, *, dl=DL, ratio=None, iterations=32):
    model = gprMax.Scene()
    extent = 0.012 if ratio is None else 0.09
    point = (extent / 2,) * 3
    for obj in (
        gprMax.Domain(p1=(extent,) * 3),
        gprMax.Discretisation(p1=(dl,) * 3),
        gprMax.PMLThickness(thickness=0),
        gprMax.OMPThreads(n=1),
        gprMax.TimeWindow(iterations=iterations),
    ):
        model.add(obj)
    owner = model
    if ratio is not None:
        owner = gprMax.SubGridHSG(p1=(0.03,) * 3, p2=(0.06,) * 3, ratio=ratio, id="fine")
        model.add(owner)
    if case == "ramp":
        wave = gprMax.Waveform(wave_type="user", id="pulse", user_func=lambda t: 1 + 1e11 * t)
    else:
        wave = gprMax.Waveform(wave_type="impulse", amp=1, freq=1e9, id="pulse")
    owner.add(wave)
    window = {"start": 0, "stop": 1e-13} if case == "released" else {}
    if case == "delayed":
        window = {"start": 5e-12, "stop": 1.5e-11}
    source = gprMax.VoltageSource(
        p1=point, polarisation=axis, resistance=0, waveform_id="pulse", id="feed", spectrum_limit="nyquist", **window
    )
    owner.add(source)
    owner.add(gprMax.Rx(p1=point, id="edge", outputs=list(COMPONENTS)))
    owner.add(gprMax.Rx(p1=tuple(p + dl for p in point), id="remote", outputs=list(COMPONENTS)))
    return model, source


def check_output(path, axis, case, source, *, group="", precision="double"):
    with h5py.File(path) as f:
        owner = f[group] if group else f
        dt, count = float(owner.attrs["dt"]), int(owner.attrs["Iterations"])
        dl = float(owner.attrs["dx_dy_dz"]["xyz".index(axis)])
        edge = next(rx for rx in owner["rxs"].values() if rx.attrs["Name"] == "edge")
        electric = edge["E" + axis][:]
        time = np.arange(count) * dt
        active = (time >= source.start) & (time <= source.stop)
        local = time - source.start
        waveform = 1 + 1e11 * local if case == "ramp" else (local < dt).astype(float)
        waveform = np.where(active, waveform, 0)
        tolerance = 3e-6 if precision == "single" else 3e-14
        np.testing.assert_allclose(electric[active], -waveform[active] / dl, rtol=tolerance, atol=0)
        assert electric[0] == pytest.approx(0 if case == "delayed" else -1 / dl, rel=tolerance, abs=0)
        assert all(edge["H" + a][0] == 0 for a in "xyz")
        assert any(np.any(edge["H" + a][1:] != 0) for a in "xyz")
        if case == "released":
            assert electric[1] != 0  # No second hard assignment to zero.
        if case == "impulse":
            # Fine-grid coupling can store samples beyond the coarse source
            # stop time; those are correctly released, not clamped.
            assert np.count_nonzero(electric[1:][active[1:]]) == 0
        excitation = owner["srcs/src1/excitation"]
        assert excitation.attrs["TimeSampleOffset"] == 0
        assert excitation.attrs["WaveformEvaluationTimeOffset"] == 0
        np.testing.assert_allclose(excitation["samples"][:], waveform, rtol=tolerance, atol=0)
        port = owner["ports/feed"]
        assert port["Vtotal"].shape == port["Iloop"].shape == (count,)
        np.testing.assert_allclose(port["Vtotal"][:], -dl * electric, rtol=tolerance, atol=0)
        assert port.attrs["TimeSampleOffset"] == 0
        assert port.attrs["CurrentTimeSampleOffset"] == -0.5 * dt
        np.testing.assert_allclose(port["time"][:], time, rtol=tolerance, atol=0)
        np.testing.assert_allclose(port["time_current"][:], time - 0.5 * dt, rtol=tolerance, atol=dt * tolerance)
        np.testing.assert_allclose(port["frequency"][:], np.fft.rfftfreq(count, dt), rtol=tolerance, atol=0)
        assert port.attrs["IndependentFrequencyResolution"] == 1 / (count * dt)
        phase = np.exp(1j * np.pi * port["frequency"][:] * dt)
        np.testing.assert_allclose(
            port["Vtotal_spectrum"][:], dt * np.fft.rfft(port["Vtotal"][:]), rtol=tolerance, atol=dt * tolerance
        )
        np.testing.assert_allclose(
            port["Iloop_spectrum"][:], dt * phase * np.fft.rfft(port["Iloop"][:]), rtol=tolerance, atol=dt * tolerance
        )
        if case == "impulse" and np.all(active):
            np.testing.assert_allclose(port["Vtotal_spectrum"][:], dt, rtol=tolerance, atol=0)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("axis", "xyz")
@pytest.mark.parametrize("case", ["impulse", "ramp", "released", "delayed"])
def test_hard_source_initial_field_and_physical_time(tmp_path, request, backend, precision, axis, case):
    options = backend_options(request, backend, precision)
    model, api_source = scene(axis, case)
    stem = tmp_path / "hard"
    gprMax.run(scenes=[model], outputfile=stem, hide_progress_bars=True, log_level=50, **options)
    check_output(stem.with_suffix(".h5"), axis, case, api_source._source, precision=precision)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("step", [0, DL])
def test_reused_geometry_reinitialises_hard_sources_every_run(tmp_path, request, backend, step):
    precision = "single" if backend == "metal" else "double"
    options = backend_options(request, backend, precision)
    model, api_source = scene("z", "impulse", iterations=31)
    if step:
        # Legacy #src_steps moves only electric/magnetic dipoles, not
        # fixed voltage terminals. Startup must preserve that restriction.
        model.add(gprMax.SrcSteps(p1=(step, 0, 0)))
    stem = tmp_path / "reuse"
    gprMax.run(
        scenes=[model], n=3, geometry_fixed=True, outputfile=stem, hide_progress_bars=True, log_level=50, **options
    )
    for index in (1, 2, 3):
        check_output(tmp_path / f"reuse{index}.h5", "z", "impulse", api_source._source, precision=precision)
        with h5py.File(tmp_path / f"reuse{index}.h5") as output:
            np.testing.assert_allclose(output["srcs/src1"].attrs["Position"], (0.006, 0.006, 0.006))
    with h5py.File(tmp_path / "reuse1.h5") as a, h5py.File(tmp_path / "reuse3.h5") as b:
        for rx in a["rxs"]:
            for component in COMPONENTS:
                np.testing.assert_array_equal(a[f"rxs/{rx}/{component}"][:], b[f"rxs/{rx}/{component}"][:])


@pytest.mark.parametrize("backend", BACKENDS)
def test_snapshot_zero_includes_initial_hard_field(tmp_path, request, backend):
    precision = "single" if backend == "metal" else "double"
    model, _ = scene("z", "impulse", iterations=4)
    model.add(
        gprMax.Snapshot(p1=(0.004,) * 3, p2=(0.009,) * 3, dl=(DL,) * 3, iterations=0, filename="initial", fileext=".h5")
    )
    gprMax.run(
        scenes=[model],
        outputfile=tmp_path / "snapshot",
        hide_progress_bars=True,
        log_level=50,
        **backend_options(request, backend, precision),
    )
    files = list(tmp_path.rglob("initial*.h5"))
    assert len(files) == 1
    with h5py.File(files[0]) as output:
        # A z Yee edge contributes to four cell-centred Ez samples.
        assert np.count_nonzero(output["Ez"][:]) == 4
        np.testing.assert_array_equal(output["Ez"][:][output["Ez"][:] != 0], -250)
        for component in ("Ex", "Ey", "Hx", "Hy", "Hz"):
            np.testing.assert_array_equal(output[component][:], 0)


@pytest.mark.parametrize("ratio", [1, 3, 5])
@pytest.mark.parametrize("axis", "xyz")
@pytest.mark.parametrize("case", ["impulse", "released"])
def test_subgrid_uses_local_electric_clock_at_startup(tmp_path, ratio, axis, case):
    model, api_source = scene(axis, case, dl=0.003, ratio=ratio, iterations=8)
    stem = tmp_path / "subgrid"
    gprMax.run(
        scenes=[model],
        outputfile=stem,
        subgrid=True,
        autotranslate=True,
        hide_progress_bars=True,
        log_level=50,
        cpu_precision="double",
    )
    check_output(stem.with_suffix(".h5"), axis, case, api_source._source, group="subgrids/fine")


@pytest.mark.parametrize("axis", "xyz")
@pytest.mark.parametrize("case", ["impulse", "released"])
def test_mpi_initial_halos_include_hard_source_on_rank_face(tmp_path, axis, case):
    launcher = Path(sys.executable).parent / "mpiexec"
    launcher = str(launcher) if launcher.is_file() else shutil.which("mpiexec")
    if not launcher or importlib.util.find_spec("mpi4py") is None:
        pytest.skip("MPI launcher/mpi4py required")
    split = ("xyz".index(axis) + 1) % 3
    partition = [1, 1, 1]
    partition[split] = 2
    window = " 0 1e-13" if case == "released" else ""
    model = tmp_path / "hard.in"
    model.write_text(
        "\n".join(
            (
                "#domain: 0.012 0.012 0.012",
                "#dx_dy_dz: 0.001 0.001 0.001",
                "#time_window: 32",
                "#pml_cells: 0",
                "#omp_threads: 1",
                "#waveform: impulse 1 1e9 pulse",
                f"#voltage_source: {axis} 0.006 0.006 0.006 0 pulse{window}",
                "#rx: 0.006 0.006 0.006 edge Ex Ey Ez Hx Hy Hz Ix Iy Iz",
                "#rx: 0.005 0.005 0.005 neighbour Ex Ey Ez Hx Hy Hz Ix Iy Iz",
                "",
            )
        )
    )
    environment = os.environ.copy()
    environment.pop("MPI4PY_RC_FINALIZE", None)
    environment.pop("FI_PROVIDER", None)
    environment.update(PYTHONPATH=str(ROOT), OMP_NUM_THREADS="1")
    common = [
        sys.executable,
        "-m",
        "gprMax",
        str(model),
        "--hide-progress-bars",
        "-cpu_precision",
        "double",
        "-n",
        "2",
        "--geometry-fixed",
    ]
    for name, command in (
        ("serial", common),
        ("mpi", [launcher, "-n", "2", *common, "--mpi", *map(str, partition)]),
    ):
        result = subprocess.run(
            [*command, "-o", str(tmp_path / name)],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    for index in (1, 2):
        with h5py.File(tmp_path / f"serial{index}.h5") as a, h5py.File(tmp_path / f"mpi{index}.h5") as b:
            assert a[f"rxs/rx1/E{axis}"][0] == b[f"rxs/rx1/E{axis}"][0] == -1000
            for rx in a["rxs"]:
                for component in COMPONENTS:
                    np.testing.assert_array_equal(a[f"rxs/{rx}/{component}"][:], b[f"rxs/{rx}/{component}"][:])
            for name in a["ports/port1"]:
                np.testing.assert_array_equal(a[f"ports/port1/{name}"][:], b[f"ports/port1/{name}"][:])
