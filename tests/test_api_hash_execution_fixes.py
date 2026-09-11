"""End-to-end regressions for the 2026-09-11 API/hash lifecycle review."""
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
from gprMax.model import Model

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ["cpu"] + [pytest.param(name, marks=pytest.mark.gpu) for name in ("cuda", "opencl", "metal")]
HASH_BASE = """#domain: 0.064 0.064 0.064
#dx_dy_dz: 0.002 0.002 0.002
#time_window: 24
#pml_cells: 0
#omp_threads: 1
#waveform: gaussian 1 1e10 w
#hertzian_dipole: z 0.032 0.032 0.032 w
#rx: 0.032 0.032 0.032 probe Ez
"""


def backend_options(request, backend):
    if backend == "cpu":
        return {"cpu_precision": "double"}
    if backend == "metal":
        metal = pytest.importorskip("Metal")
        if metal.MTLCreateSystemDefaultDevice() is None:
            pytest.skip("No Metal device")
        return {"metal": True}
    index = request.getfixturevalue("gpu_device" if backend == "cuda" else "opencl_device")
    return {"gpu" if backend == "cuda" else "opencl": [index], "gpu_precision": "double"}


def scene(*, ratio=None):
    model = gprMax.Scene()
    for obj in (
        gprMax.Domain(p1=(0.064,) * 3),
        gprMax.Discretisation(p1=(0.002,) * 3),
        gprMax.TimeWindow(iterations=24),
        gprMax.PMLThickness(thickness=0),
        gprMax.OMPThreads(1),
    ):
        model.add(obj)
    owner = model
    if ratio is not None:
        owner = gprMax.SubGridHSG(p1=(0.020,) * 3, p2=(0.044,) * 3, ratio=ratio, id="fine")
        model.add(owner)
    owner.add(gprMax.Waveform(wave_type="gaussian", amp=1, freq=1e10, id="w"))
    owner.add(gprMax.HertzianDipole(p1=(0.032,) * 3, polarisation="z", waveform_id="w"))
    owner.add(gprMax.Rx(p1=(0.032,) * 3, id="probe", outputs=["Ez"]))
    return model


def run(model, target, **kwargs):
    return gprMax.run(
        scenes=[model] * kwargs.get("n", 1), outputfile=target, hide_progress_bars=True, log_level=50, **kwargs
    )


def trace(path, group="rxs/rx1/Ez"):
    with h5py.File(path) as handle:
        return handle[group][:]


@pytest.mark.parametrize("ratio", [1, 3])
@pytest.mark.parametrize("explicit_false", [False, True])
def test_subgrid_omission_is_rejected_without_outputs(tmp_path, ratio, explicit_false):
    kwargs = {"subgrid": False} if explicit_false else {}
    with pytest.raises(ValueError, match="subgrid=True"):
        run(scene(ratio=ratio), tmp_path / "omitted", **kwargs)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("ratio", [1, 3])
def test_enabled_subgrid_actually_advances_across_reuse(tmp_path, ratio):
    run(
        scene(ratio=ratio),
        tmp_path / "fine",
        n=2,
        geometry_fixed=True,
        subgrid=True,
        autotranslate=True,
        cpu_precision="double",
    )
    first = trace(tmp_path / "fine1.h5", "subgrids/fine/rxs/rx1/Ez")
    assert np.isfinite(first).all() and np.count_nonzero(first) > 1
    np.testing.assert_array_equal(first, trace(tmp_path / "fine2.h5", "subgrids/fine/rxs/rx1/Ez"))


def test_duplicate_subgrid_id_never_starts_build_or_truncates_output(tmp_path, monkeypatch):
    model = scene(ratio=1)
    model.add(gprMax.SubGridHSG(p1=(0.020,) * 3, p2=(0.044,) * 3, ratio=1, id="fine"))
    target = tmp_path / "existing.h5"
    target.write_bytes(b"must survive validation")
    monkeypatch.setattr(Model, "build", lambda self: pytest.fail("Invalid Scene reached build"))
    with pytest.raises(ValueError, match="Duplicate subgrid ID"):
        run(model, target, subgrid=True)
    assert target.read_bytes() == b"must survive validation"


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("restart", [1, 3])
def test_output_dir_and_snapshots_survive_geometry_reuse(tmp_path, request, backend, restart):
    model = scene()
    requested = tmp_path / "requested"
    model.add(gprMax.OutputDir(dir=requested))
    model.add(
        gprMax.Snapshot(
            p1=(0.028,) * 3, p2=(0.038,) * 3, dl=(0.002,) * 3, iterations=3, filename="frame", fileext=".h5"
        )
    )
    run(model, tmp_path / "fallback", n=2, i=restart, geometry_fixed=True, **backend_options(request, backend))
    arrays = []
    frames = []
    for index in (restart, restart + 1):
        arrays.append(trace(requested / f"fallback{index}.h5"))
        with h5py.File(requested / f"fallback{index}_snaps/frame.h5") as handle:
            frames.append(handle["Ez"][:])
        assert not (tmp_path / f"fallback{index}.h5").exists()
    assert np.count_nonzero(arrays[0]) > 0
    np.testing.assert_array_equal(*arrays)
    np.testing.assert_array_equal(*frames)


@pytest.mark.parametrize("backend", BACKENDS)
def test_nested_include_material_resolution_and_hash_output_dir(tmp_path, monkeypatch, request, backend):
    modeldir, caller = tmp_path / "model", tmp_path / "caller"
    nested = modeldir / "nested"
    nested.mkdir(parents=True)
    caller.mkdir()
    (caller / "outer.in").write_text("#material: 9 0 1 0 sample\n")
    (modeldir / "outer.in").write_text("#include_file: nested/inner.in\n")
    (nested / "inner.in").write_text("#material: 4 0 1 0 sample\n#box: 0.040 0.040 0.040 0.048 0.048 0.048 pec\n")
    source = modeldir / "nested.in"
    source.write_text(HASH_BASE + "#include_file: outer.in\n#output_dir: results\n")
    built = []
    original = Model.build

    def observe(model):
        original(model)
        built.append((np.count_nonzero(model.G.solid == 0), next(m.er for m in model.G.materials if m.ID == "sample")))

    monkeypatch.setattr(Model, "build", observe)
    monkeypatch.chdir(caller)
    gprMax.run(
        inputfile=source,
        n=2,
        geometry_fixed=True,
        hide_progress_bars=True,
        log_level=50,
        **backend_options(request, backend),
    )
    assert built == [(64, 4), (64, 4)]
    for index in (1, 2):
        assert np.isfinite(trace(modeldir / f"results/nested{index}.h5")).all()
    assert not (caller / "results").exists()


def test_api_relative_output_dir_keeps_cwd_semantics(tmp_path, monkeypatch):
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    model = scene()
    model.add(gprMax.OutputDir(dir="results"))
    run(model, tmp_path / "output")
    assert (caller / "results/output.h5").is_file()
    assert not (tmp_path / "results").exists()


def fractal_scene(*, mixing=False):
    model = scene()
    if mixing:
        model.add(
            gprMax.MaterialRange(
                er_lower=2,
                er_upper=4,
                sigma_lower=0,
                sigma_upper=0,
                mr_lower=1,
                mr_upper=1,
                ro_lower=0,
                ro_upper=0,
                id="mix",
            )
        )
    model.add(
        gprMax.FractalBox(
            p1=(0.040,) * 3,
            p2=(0.052,) * 3,
            frac_dim=1.5,
            weighting=(1, 1, 1),
            n_materials=4 if mixing else 1,
            mixing_model_id="mix" if mixing else "pec",
            id="fb",
            seed=1,
        )
    )
    model.add(
        gprMax.AddSurfaceRoughness(
            p1=(0.040, 0.040, 0.052),
            p2=(0.052,) * 3,
            frac_dim=1.5,
            weighting=(1, 1),
            limits=(0.050, 0.054),
            fractal_box_id="fb",
            seed=1,
        )
    )
    return model


@pytest.mark.parametrize("mixing", [False, True])
@pytest.mark.parametrize("workflow", ["preview", "solve", "retry", "shared_scenes"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_fractal_definition_rebuilds_fresh_volume(tmp_path, monkeypatch, request, mixing, workflow, backend):
    options = backend_options(request, backend)
    built = []
    original = Model.build

    def observe(model):
        original(model)
        grid = model.G
        assert len(grid.fractalvolumes) == 1
        assert int(grid.ID.max()) < len(grid.materials)
        built.append((grid.fractalvolumes[0], grid.solid.copy()))

    monkeypatch.setattr(Model, "build", observe)
    model = fractal_scene(mixing=mixing)
    if workflow == "retry":
        modifier = model.geometry_objects[-1]
        modifier.kwargs["fractal_box_id"] = "missing"
        with pytest.raises(ValueError, match="cannot find FractalBox missing"):
            run(model, tmp_path / "failed", geometry_only=True, **options)
        modifier.kwargs["fractal_box_id"] = "fb"
    elif workflow == "shared_scenes":
        run(model, tmp_path / "shared", n=2, **options)
    else:
        run(model, tmp_path / "first", geometry_only=workflow == "preview", **options)
    run(model, tmp_path / "rebuilt", **options)
    run(fractal_scene(mixing=mixing), tmp_path / "fresh", **options)
    assert len({id(volume) for volume, _ in built}) == len(built)
    for volume, solid in built:
        assert volume.fractalvolume is None and volume.mask is None
        np.testing.assert_array_equal(solid, built[-1][1])
    np.testing.assert_array_equal(trace(tmp_path / "rebuilt.h5"), trace(tmp_path / "fresh.h5"))


def test_mpi_nested_include_output_dir_reuse_matches_serial(tmp_path):
    launcher = Path(sys.executable).parent / "mpiexec"
    launcher = str(launcher) if launcher.is_file() else shutil.which("mpiexec")
    if not launcher or importlib.util.find_spec("mpi4py") is None:
        pytest.skip("MPI launcher/mpi4py required")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "geometry.in").write_text("#box: 0.040 0.040 0.040 0.048 0.048 0.048 pec\n")
    (tmp_path / "outer.in").write_text("#include_file: nested/geometry.in\n")
    environment = os.environ.copy()
    environment.pop("FI_PROVIDER", None)
    environment.pop("MPI4PY_RC_FINALIZE", None)
    environment.update(PYTHONPATH=str(ROOT), OMP_NUM_THREADS="1")
    for mode in ("serial", "mpi"):
        source = tmp_path / f"{mode}.in"
        source.write_text(HASH_BASE + f"#include_file: outer.in\n#output_dir: {mode}_results\n")
        command = [
            sys.executable,
            "-m",
            "gprMax",
            str(source),
            "-n",
            "2",
            "-i",
            "3",
            "--geometry-fixed",
            "--hide-progress-bars",
            "-cpu_precision",
            "double",
        ]
        if mode == "mpi":
            command = [launcher, "-n", "2", *command, "--mpi", "2", "1", "1"]
        result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
    for index in (3, 4):
        a = trace(tmp_path / f"serial_results/serial{index}.h5")
        b = trace(tmp_path / f"mpi_results/mpi{index}.h5")
        assert np.count_nonzero(a) > 0
        np.testing.assert_array_equal(a, b)
