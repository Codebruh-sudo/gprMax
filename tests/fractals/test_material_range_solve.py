"""Public range builds against explicit materials and real MPI decomposition."""

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
from numpy.testing import assert_allclose, assert_array_equal

import gprMax

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = Path(sys.executable).parent / "mpiexec"
MPIEXEC = str(LAUNCHER) if LAUNCHER.is_file() else shutil.which("mpiexec")
CONDUCTIVITIES = (5e-6, 1.5e-5, 2.5e-5, 3.5e-5)


def _scene(output, averaging, explicit=False):
    scene = gprMax.Scene()
    for obj in (
        gprMax.Discretisation(p1=(0.001,) * 3),
        gprMax.Domain(p1=(0.016,) * 3),
        gprMax.PMLThickness(thickness=2),
        gprMax.TimeWindow(iterations=100),
        gprMax.OMPThreads(n=1),
    ):
        scene.add(obj)
    if explicit:
        for index, se in enumerate(CONDUCTIVITIES):
            scene.add(gprMax.Material(er=4, se=se, mr=1, sm=0, id=f"bin{index}"))
        scene.add(gprMax.MaterialList(list_of_materials=[f"bin{i}" for i in range(4)], id="range"))
    else:
        scene.add(
            gprMax.MaterialRange(
                er_lower=4,
                er_upper=4,
                sigma_lower=0,
                sigma_upper=4e-5,
                mr_lower=1,
                mr_upper=1,
                ro_lower=0,
                ro_upper=0,
                id="range",
            )
        )
    scene.add(
        gprMax.FractalBox(
            p1=(0.004,) * 3,
            p2=(0.012,) * 3,
            frac_dim=1.5,
            weighting=(1, 1, 1),
            n_materials=4,
            mixing_model_id="range",
            id="volume",
            seed=1,
            averaging=averaging,
        )
    )
    scene.add(gprMax.Waveform(wave_type="impulse", amp=0.001, freq=1e9, id="pulse"))
    scene.add(
        gprMax.HertzianDipole(
            p1=(0.006, 0.008, 0.008),
            polarisation="z",
            waveform_id="pulse",
        )
    )
    scene.add(gprMax.Rx(p1=(0.010, 0.008, 0.008)))
    scene.add(
        gprMax.GeometryObjectsWrite(
            p1=(0, 0, 0),
            p2=(0.016,) * 3,
            filename=str(output) + "_geometry",
        )
    )
    return scene


def _run(output, averaging="n", precision="double", explicit=False, **options):
    gprMax.run(
        scenes=[_scene(output, averaging, explicit)],
        outputfile=output,
        cpu_precision=precision,
        hide_progress_bars=True,
        log_level=30,
        **options,
    )


def _properties(output):
    entries = json.loads(Path(str(output) + "_geometry_materials.json").read_text())["materials"]
    with h5py.File(str(output) + "_geometry.h5") as data:
        keys = data["material_keys"].asstr()[...]
        table = np.array(
            [
                [
                    entries[key]["base"][prop]
                    for prop in (
                        "relative_permittivity",
                        "electric_conductivity_s_per_m",
                        "relative_permeability",
                        "magnetic_conductivity_s_per_m",
                    )
                ]
                for key in keys
            ],
            dtype=float,
        )
        return {name: table[data[name][...]] for name in ("data", "ID")}


def _check_bins(output):
    cells = _properties(output)["data"]
    shell = cells[4:12, 4:12, 4:12]
    assert_allclose(np.unique(shell[..., 1]), CONDUCTIVITIES, rtol=4e-16, atol=1e-20)
    assert_array_equal(shell[..., 0], np.full((8, 8, 8), 4.0))
    assert_array_equal(cells[:4, ..., 1], 0.0)


def _compare(first, second, precision, tolerance=None):
    if tolerance is None:
        tolerance = 3e-5 if precision == "single" else 2e-12
    for key, values in _properties(first).items():
        assert_allclose(values, _properties(second)[key], rtol=1e-14, atol=1e-20)
    with h5py.File(first.with_suffix(".h5")) as reference, h5py.File(
        second.with_suffix(".h5")
    ) as actual:
        assert np.max(np.abs(reference["rxs/rx1/Ez"][...])) > 0
        for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            a = reference[f"rxs/rx1/{component}"][...]
            b = actual[f"rxs/rx1/{component}"][...]
            assert np.isfinite(b).all()
            scale = max(np.max(np.abs(a)), 1e-20)
            # A trace-wide peak scale avoids relative errors at zero crossings.
            assert np.max(np.abs(a - b)) / scale < tolerance


@pytest.mark.parametrize("averaging", ["y", "n"])
@pytest.mark.parametrize("precision", ["single", "double"])
def test_range_fractal_matches_explicit_material_list(tmp_path, averaging, precision):
    ranged, explicit = tmp_path / "range", tmp_path / "explicit"
    _run(ranged, averaging, precision)
    _run(explicit, averaging, precision, explicit=True)
    _check_bins(ranged)
    _compare(ranged, explicit, precision)


@pytest.mark.parametrize("averaging", ["y", "n"])
def test_hash_range_fractal_preserves_small_conductivities(tmp_path, averaging):
    model = tmp_path / "range.in"
    model.write_text(
        "#dx_dy_dz: 0.001 0.001 0.001\n"
        "#domain: 0.016 0.016 0.016\n"
        "#pml_cells: 2\n"
        "#time_window: 100\n"
        "#omp_threads: 1\n"
        "#material_range: 4 4 0 0.00004 1 1 0 0 range\n"
        f"#fractal_box: 0.004 0.004 0.004 0.012 0.012 0.012 1.5 1 1 1 4 range volume 1 {averaging}\n"
        f"#geometry_objects_write: 0 0 0 0.016 0.016 0.016 {tmp_path / 'range_geometry'}\n"
    )
    gprMax.run(inputfile=model, geometry_only=True, hide_progress_bars=True, log_level=30)
    _check_bins(model.with_suffix(""))


def test_fixed_geometry_reuses_range_materials_and_resets_fields(tmp_path):
    output = tmp_path / "repeat"
    _run(output, n=2, geometry_fixed=True)
    with h5py.File(tmp_path / "repeat1.h5") as first, h5py.File(tmp_path / "repeat2.h5") as second:
        assert np.max(np.abs(first["rxs/rx1/Ez"][...])) > 0
        for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            assert_array_equal(
                first[f"rxs/rx1/{component}"][...], second[f"rxs/rx1/{component}"][...]
            )


def _shared_scene(output, averaging="n", rough=False, separate=False, reverse=False):
    scene = gprMax.Scene()
    for obj in (
        gprMax.Discretisation(p1=(0.001,) * 3),
        gprMax.Domain(p1=(0.032, 0.016, 0.016)),
        gprMax.PMLThickness(thickness=2),
        gprMax.TimeWindow(iterations=120),
        gprMax.OMPThreads(n=1),
    ):
        scene.add(obj)
    for name in ("range0", "range1") if separate else ("range",):
        scene.add(
            gprMax.MaterialRange(
                er_lower=4,
                er_upper=4,
                sigma_lower=0,
                sigma_upper=4e-5,
                mr_lower=1,
                mr_upper=1,
                ro_lower=0,
                ro_upper=0,
                id=name,
            )
        )
    for i in (1, 0) if reverse else (0, 1):
        start = (2, 20)[i]
        scene.add(
            gprMax.FractalBox(
                p1=(start * 0.001, 0.004, 0.004),
                p2=((start + 8) * 0.001, 0.012, 0.012),
                frac_dim=1.5,
                weighting=(1, 1, 1),
                n_materials=(4, 8)[i],
                mixing_model_id=f"range{i}" if separate else "range",
                id=f"volume{i}",
                seed=1,
                averaging=averaging,
            )
        )
        if rough:
            scene.add(
                gprMax.AddSurfaceRoughness(
                    p1=(start * 0.001, 0.004, 0.012),
                    p2=((start + 8) * 0.001, 0.012, 0.012),
                    frac_dim=1.5,
                    weighting=(1, 1),
                    limits=(0.010, 0.014),
                    fractal_box_id=f"volume{i}",
                    seed=2,
                )
            )
    scene.add(gprMax.Waveform(wave_type="impulse", amp=0.001, freq=1e9, id="pulse"))
    scene.add(
        gprMax.HertzianDipole(p1=(0.006, 0.008, 0.008), polarisation="z", waveform_id="pulse")
    )
    scene.add(gprMax.Rx(p1=(0.010, 0.008, 0.008)))
    scene.add(
        gprMax.GeometryObjectsWrite(
            p1=(0, 0, 0),
            p2=(0.032, 0.016, 0.016),
            filename=str(output) + "_geometry",
        )
    )
    return scene


def _run_shared(
    output,
    averaging="n",
    precision="double",
    rough=False,
    separate=False,
    reverse=False,
    **options,
):
    gprMax.run(
        scenes=[_shared_scene(output, averaging, rough, separate, reverse)],
        outputfile=output,
        cpu_precision=precision,
        hide_progress_bars=True,
        log_level=30,
        **options,
    )


def _check_shared_bins(output, rough=False):
    cells = _properties(output)["data"]
    for start, expected in (
        (2, CONDUCTIVITIES),
        (20, np.array([2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5]) * 1e-6),
    ):
        # Below the roughness interval, all cells belong to the fractal volume.
        interior = cells[start : start + 8, 4:12, 4:10]
        observed = np.unique(interior[..., 1])
        if rough:
            # Roughness changes the generated extent and removes some voxels;
            # not every bin must survive in this interior slice.
            assert np.all(
                np.any(np.isclose(observed[:, None], expected, rtol=4e-16, atol=1e-20), axis=1)
            )
        else:
            assert_allclose(observed, expected, rtol=4e-16, atol=1e-20)
        assert_array_equal(interior[..., 0], 4.0)


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("averaging", ["n", "y"])
@pytest.mark.parametrize("rough", [False, True])
def test_shared_range_matches_separate_definitions_and_reversed_build_order(
    tmp_path, precision, averaging, rough
):
    shared, separate, reverse = (tmp_path / name for name in ("shared", "separate", "reverse"))
    _run_shared(shared, averaging, precision, rough)
    _run_shared(separate, averaging, precision, rough, separate=True)
    _run_shared(reverse, averaging, precision, rough, reverse=True)
    for output in (shared, separate, reverse):
        _check_shared_bins(output, rough)
    _compare(shared, separate, precision)
    _compare(shared, reverse, precision)


@pytest.mark.parametrize("rough", [False, True])
def test_shared_range_geometry_fixed_and_rebuilt_runs_agree(tmp_path, rough):
    fixed = tmp_path / "fixed"
    _run_shared(fixed, rough=rough, n=2, geometry_fixed=True)
    fresh = tmp_path / "fresh"
    _run_shared(fresh, rough=rough)
    with h5py.File(fresh.with_suffix(".h5")) as reference:
        for run in (1, 2):
            with h5py.File(tmp_path / f"fixed{run}.h5") as result:
                for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
                    assert_array_equal(
                        result[f"rxs/rx1/{component}"][...],
                        reference[f"rxs/rx1/{component}"][...],
                    )


@pytest.mark.parametrize("averaging", ["n", "y"])
def test_hash_shared_range_uses_independent_bin_counts(tmp_path, averaging):
    model = tmp_path / "shared.in"
    model.write_text(
        "#dx_dy_dz: 0.001 0.001 0.001\n"
        "#domain: 0.032 0.016 0.016\n"
        "#pml_cells: 2\n"
        "#time_window: 100\n"
        "#omp_threads: 1\n"
        "#material_range: 4 4 0 0.00004 1 1 0 0 range\n"
        f"#fractal_box: 0.002 0.004 0.004 0.010 0.012 0.012 1.5 1 1 1 4 range first 1 {averaging}\n"
        f"#fractal_box: 0.020 0.004 0.004 0.028 0.012 0.012 1.5 1 1 1 8 range second 1 {averaging}\n"
        f"#geometry_objects_write: 0 0 0 0.032 0.016 0.016 {tmp_path / 'shared_geometry'}\n"
    )
    gprMax.run(inputfile=model, geometry_only=True, hide_progress_bars=True, log_level=30)
    _check_shared_bins(model.with_suffix(""))


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("backend", ["cuda", "opencl"])
@pytest.mark.parametrize("precision", ["single", "double"])
def test_shared_range_device_matches_cpu(tmp_path, request, backend, precision):
    options = (
        {"gpu": [request.getfixturevalue("gpu_device")]}
        if backend == "cuda"
        else {"opencl": [request.getfixturevalue("opencl_device")]}
    )
    serial, device, independent = (tmp_path / name for name in ("serial", backend, "independent"))
    _run_shared(serial, "y", precision, rough=True)
    _run_shared(device, "y", precision, rough=True, gpu_precision=precision, **options)
    _run_shared(
        independent, "y", precision, rough=True, separate=True, gpu_precision=precision, **options
    )
    _check_shared_bins(device, rough=True)
    _compare(device, independent, precision)
    _compare(serial, device, precision)


@pytest.mark.skipif(
    MPIEXEC is None or importlib.util.find_spec("mpi4py_fft") is None or not h5py.get_config().mpi,
    reason="requires MPI, mpi4py-fft/FFTW and parallel HDF5",
)
@pytest.mark.parametrize("partition", [(2, 1, 1), (1, 2, 1), (1, 1, 2)])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("rough", [False, True])
def test_shared_range_matches_real_mpi(tmp_path, partition, precision, rough):
    serial, distributed = tmp_path / "serial", tmp_path / "mpi"
    _run_shared(serial, "y", precision, rough)
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
            str(distributed),
            "--shared",
            *(["--rough"] if rough else []),
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
    _check_shared_bins(distributed, rough)
    _compare(serial, distributed, precision)


@pytest.mark.skipif(
    MPIEXEC is None or importlib.util.find_spec("mpi4py_fft") is None or not h5py.get_config().mpi,
    reason="requires MPI, mpi4py-fft/FFTW and parallel HDF5",
)
@pytest.mark.parametrize("partition", [(2, 1, 1), (1, 2, 1), (1, 1, 2)])
@pytest.mark.parametrize("precision", ["single", "double"])
def test_range_fractal_matches_real_mpi(tmp_path, partition, precision):
    serial, distributed = tmp_path / "serial", tmp_path / "mpi"
    _run(serial, "y", precision)
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
            str(distributed),
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
    _check_bins(distributed)
    _compare(serial, distributed, precision)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--precision", choices=("single", "double"), default="double")
    parser.add_argument("--partition", type=int, nargs=3)
    parser.add_argument("--shared", action="store_true")
    parser.add_argument("--rough", action="store_true")
    args = parser.parse_args()
    if args.shared:
        _run_shared(args.output, "y", args.precision, args.rough, mpi=tuple(args.partition))
    else:
        _run(args.output, "y", args.precision, mpi=tuple(args.partition))
