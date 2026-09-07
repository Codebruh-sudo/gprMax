"""Pre-run rejection and unchanged accepted CPU/GPU/MPI dispersive solves."""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

import gprMax
import gprMax.config as config
import gprMax.dispersive_stability as stability

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = Path(sys.executable).parent / "mpiexec"
MPIEXEC = str(LAUNCHER) if LAUNCHER.is_file() else shutil.which("mpiexec")
pytestmark = pytest.mark.integration


def scene(factor):
    dl = 0.001
    dt_cfl = dl / (config.c * np.sqrt(3))
    model = gprMax.Scene()
    for obj in (
        gprMax.Discretisation(p1=(dl,) * 3),
        gprMax.Domain(p1=(0.008,) * 3),
        gprMax.TimeStepStabilityFactor(f=factor),
        gprMax.PMLThickness(thickness=0),
        gprMax.TimeWindow(iterations=100),
        gprMax.OMPThreads(n=1),
        gprMax.Material(er=1, se=0, mr=1, sm=0, id="strong_lorentz"),
        gprMax.AddLorentzDispersion(
            poles=1,
            er_delta=[100],
            omega=[0.1 / dt_cfl],
            delta=[0.01 / dt_cfl],
            material_ids=["strong_lorentz"],
        ),
        gprMax.Box(p1=(0,) * 3, p2=(0.008,) * 3, material_id="strong_lorentz", averaging=False),
        gprMax.Waveform(wave_type="impulse", amp=0.001, freq=1e9, id="pulse"),
        gprMax.HertzianDipole(p1=(0.003, 0.004, 0.004), polarisation="z", waveform_id="pulse"),
        gprMax.Rx(p1=(0.005, 0.004, 0.004)),
    ):
        model.add(obj)
    return model


def run(output, factor=0.25, **options):
    gprMax.run(
        scenes=[scene(factor)],
        n=1,
        outputfile=output,
        hide_progress_bars=True,
        log_level=30,
        **options,
    )


def fields(output):
    with h5py.File(output.with_suffix(".h5")) as data:
        return {key: value[...] for key, value in data["rxs/rx1"].items()}


@pytest.mark.parametrize("precision", ("single", "double"))
def test_unsafe_material_stops_before_output(tmp_path, precision):
    output = tmp_path / "unsafe"
    with pytest.raises(ValueError, match="Dispersive timestep check failed.*strong_lorentz"):
        run(output, factor=1, cpu_precision=precision)
    assert not output.with_suffix(".h5").exists()


@pytest.mark.parametrize("precision", ("single", "double"))
def test_accepted_model_is_bitwise_unchanged(monkeypatch, tmp_path, precision):
    checked, reference = tmp_path / "checked", tmp_path / "reference"
    run(checked, cpu_precision=precision)
    monkeypatch.setattr(stability, "validate_grid_dispersive_timestep", lambda grid: None)
    run(reference, cpu_precision=precision)
    observed = fields(checked)
    assert np.max(abs(observed["Ez"])) > 0
    for component, trace in fields(reference).items():
        assert np.isfinite(observed[component]).all()
        np.testing.assert_array_equal(observed[component], trace)


def test_geometry_fixed_reuses_checked_coefficients(monkeypatch, tmp_path):
    calls = []
    original = stability.validate_grid_dispersive_timestep

    def checked(grid):
        calls.append(grid.dt)
        return original(grid)

    monkeypatch.setattr(stability, "validate_grid_dispersive_timestep", checked)
    output = tmp_path / "repeat"
    gprMax.run(
        scenes=[scene(0.25)],
        n=2,
        geometry_fixed=True,
        outputfile=output,
        hide_progress_bars=True,
        log_level=30,
    )
    assert len(calls) == 1
    first = fields(tmp_path / "repeat1")
    for component, trace in fields(tmp_path / "repeat2").items():
        np.testing.assert_array_equal(first[component], trace)


@pytest.mark.parametrize("ratio", (1, 3))
@pytest.mark.parametrize("factor", (1, 0.25))
def test_subgrid_checks_its_own_timestep(monkeypatch, tmp_path, ratio, factor):
    dl = 0.003
    local_cfl = dl / (config.c * np.sqrt(3) * ratio)
    model = gprMax.Scene()
    for obj in (
        gprMax.Discretisation(p1=(dl,) * 3),
        gprMax.Domain(p1=(0.06,) * 3),
        gprMax.TimeStepStabilityFactor(f=factor),
        gprMax.PMLThickness(thickness=0),
        gprMax.TimeWindow(iterations=5),
        gprMax.OMPThreads(n=1),
    ):
        model.add(obj)
    child = gprMax.SubGridHSG(p1=(0.021,) * 3, p2=(0.039,) * 3, ratio=ratio, id="dispersive_child")
    child.add(gprMax.Material(er=1, se=0, mr=1, sm=0, id="child_lorentz"))
    child.add(
        gprMax.AddLorentzDispersion(
            poles=1,
            er_delta=[100],
            omega=[0.1 / local_cfl],
            delta=[0.01 / local_cfl],
            material_ids=["child_lorentz"],
        )
    )
    child.add(gprMax.Box(p1=(0.027,) * 3, p2=(0.033,) * 3, material_id="child_lorentz"))
    model.add(child)
    timesteps = []
    original = stability.validate_grid_dispersive_timestep

    def checked(grid):
        if any(material.ID == "child_lorentz" for material in grid.materials):
            timesteps.append(grid.dt)
        return original(grid)

    monkeypatch.setattr(stability, "validate_grid_dispersive_timestep", checked)
    options = dict(
        scenes=[model],
        outputfile=tmp_path / "subgrid",
        subgrid=True,
        autotranslate=True,
        hide_progress_bars=True,
        log_level=30,
    )
    if factor == 1:
        with pytest.raises(ValueError, match="Dispersive timestep check failed.*child_lorentz"):
            gprMax.run(**options)
    else:
        gprMax.run(**options)
    assert timesteps == pytest.approx([factor * local_cfl], rel=1e-12)


@pytest.mark.parametrize("database", (False, True))
@pytest.mark.parametrize("factor", (1, 0.25))
def test_hash_input_and_database_share_the_guard(tmp_path, database, factor):
    dt_cfl = 0.001 / (config.c * np.sqrt(3))
    frequency, damping = 0.1 / dt_cfl, 0.01 / dt_cfl
    material_commands = (
        "#material: 1 0 1 0 strong_lorentz\n"
        f"#add_dispersion_lorentz: 1 100 {frequency:.17g} {damping:.17g} strong_lorentz\n"
    )
    if database:
        document = {
            "schema": "gprMax-material-database",
            "schema_version": 1,
            "database": {"id": "local", "name": "Test", "version": "1.0"},
            "materials": {
                "strong_lorentz": {
                    "name": "Strong Lorentz pole",
                    "model": "lorentz",
                    "base": {
                        "relative_permittivity": 1,
                        "electric_conductivity_s_per_m": 0,
                        "relative_permeability": 1,
                        "magnetic_conductivity_s_per_m": 0,
                    },
                    "poles": [
                        {
                            "relative_permittivity_difference": 100,
                            "resonance_frequency_hz": frequency,
                            "damping_coefficient_per_s": damping,
                        }
                    ],
                }
            },
        }
        (tmp_path / "local.json").write_text(json.dumps(document))
        material_commands = "#material_from_database: local strong_lorentz\n"
    inputfile = tmp_path / "hash_model.in"
    inputfile.write_text(
        "#domain: 0.008 0.008 0.008\n"
        "#dx_dy_dz: 0.001 0.001 0.001\n"
        f"#time_step_stability_factor: {factor}\n"
        "#time_window: 10\n"
        "#pml_cells: 0\n"
        "#omp_threads: 1\n" + material_commands + "#box: 0 0 0 0.008 0.008 0.008 strong_lorentz n\n"
        "#waveform: impulse 0.001 1e9 pulse\n"
        "#hertzian_dipole: z 0.003 0.004 0.004 pulse\n"
        "#rx: 0.005 0.004 0.004\n"
    )
    options = dict(inputfile=inputfile, hide_progress_bars=True, log_level=30)
    if factor == 1:
        with pytest.raises(ValueError, match="Dispersive timestep check failed.*strong_lorentz"):
            gprMax.run(**options)
        assert not inputfile.with_suffix(".h5").exists()
    else:
        gprMax.run(**options)
        assert inputfile.with_suffix(".h5").exists()


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ("cuda", "opencl"))
@pytest.mark.parametrize("precision", ("single", "double"))
def test_accelerators_reject_unsafe_and_preserve_safe_parity(tmp_path, request, backend, precision):
    device = request.getfixturevalue("gpu_device" if backend == "cuda" else "opencl_device")
    options = {"gpu" if backend == "cuda" else "opencl": [device], "gpu_precision": precision}
    with pytest.raises(ValueError, match="Dispersive timestep check failed"):
        run(tmp_path / "unsafe", factor=1, **options)
    run(tmp_path / "cpu", cpu_precision=precision)
    run(tmp_path / "device", **options)
    actual, reference = fields(tmp_path / "device"), fields(tmp_path / "cpu")
    tolerance = 3e-5 if precision == "single" else 3e-11
    scale = max(np.max(abs(trace)) for trace in reference.values())
    for component, trace in reference.items():
        np.testing.assert_allclose(actual[component], trace, rtol=tolerance, atol=tolerance * scale)


def _mpi(output, factor, *, collective_only=False):
    command = [
        MPIEXEC,
        "-n",
        "2",
        sys.executable,
        str(Path(__file__).resolve()),
        str(output),
        "--factor",
        str(factor),
    ]
    if collective_only:
        command.append("--collective-only")
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=str(ROOT),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
    )
    environment.pop("MPI4PY_RC_FINALIZE", None)
    environment.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    environment.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    return subprocess.run(
        command, env=environment, cwd=ROOT, capture_output=True, text=True, timeout=90
    )


@pytest.mark.skipif(
    MPIEXEC is None or importlib.util.find_spec("mpi4py") is None, reason="MPI required"
)
@pytest.mark.parametrize("collective_only", (False, True))
def test_mpi_failure_stops_the_job_without_hanging(tmp_path, collective_only):
    completed = _mpi(tmp_path / "unsafe", 1, collective_only=collective_only)
    output = completed.stdout + completed.stderr
    if collective_only:
        assert completed.returncode == 2, output
        assert output.count("STABILITY_REJECTED_RANK_") == 2, output
    else:
        # MPIContext aborts the job on setup failures. This can interrupt
        # another rank's logging, so require a diagnostic, not duplicate logs.
        assert completed.returncode != 0, output
    assert "Dispersive timestep check failed" in output
    assert not (tmp_path / "unsafe.h5").exists()


@pytest.mark.skipif(
    MPIEXEC is None or importlib.util.find_spec("mpi4py") is None, reason="MPI required"
)
def test_mpi_accepted_result_matches_cpu(tmp_path):
    run(tmp_path / "cpu", cpu_precision="double")
    completed = _mpi(tmp_path / "distributed", 0.25)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    actual, reference = fields(tmp_path / "distributed"), fields(tmp_path / "cpu")
    scale = max(np.max(abs(trace)) for trace in reference.values())
    for component, trace in reference.items():
        np.testing.assert_allclose(actual[component], trace, rtol=3e-11, atol=3e-11 * scale)


def main():
    from mpi4py import MPI

    if not MPI.Is_initialized():
        MPI.Init()
    comm = MPI.COMM_WORLD
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--factor", type=float, required=True)
    parser.add_argument("--collective-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.collective_only:
            from types import SimpleNamespace

            from gprMax.materials import DispersiveMaterial, Material

            config.sim_config = SimpleNamespace(em_consts={"e0": config.e0, "m0": config.m0})
            config.get_model_config = lambda: SimpleNamespace(mode="3D")
            material = Material(0, "background")
            if comm.rank == 1:
                material = DispersiveMaterial(0, "rank_one_strong_pole")
                material.er, material.type, material.poles = 1, "lorentz", 1
                material.deltaer, material.tau, material.alpha = [100], [1e11], [1e10]
            grid = SimpleNamespace(
                dt=1e-12,
                dx=0.001,
                dy=0.001,
                dz=0.001,
                materials=[material],
                comm=comm,
                name=f"rank_{comm.rank}",
            )
            stability.validate_grid_dispersive_timestep(grid)
        else:
            run(args.output, args.factor, cpu_precision="double", mpi=(2, 1, 1))
    except ValueError as exc:
        if "Dispersive timestep check failed" not in str(exc):
            raise
        print(f"STABILITY_REJECTED_RANK_{comm.rank}: {exc}", flush=True)
        MPI.Finalize()
        raise SystemExit(2)
    MPI.Finalize()


if __name__ == "__main__":
    main()
