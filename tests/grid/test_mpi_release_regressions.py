"""Public-API regressions that need real rank ownership and MPI transport."""

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

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = Path(sys.executable).parent / "mpiexec"
MPIEXEC = str(LAUNCHER) if LAUNCHER.is_file() else shutil.which("mpiexec")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        MPIEXEC is None or importlib.util.find_spec("mpi4py") is None,
        reason="requires mpi4py and an MPI launcher",
    ),
]


def _run(case, output, partition=None, **options):
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), case, str(output)]
    if partition is not None:
        command = [
            MPIEXEC,
            "-n",
            str(np.prod(partition)),
            *command,
            "--partition",
            *map(str, partition),
        ]
    elif case == "taskfarm":
        command = [MPIEXEC, "-n", "3", *command]
    for key, value in options.items():
        command.extend(("--" + key.replace("_", "-"), str(value)))
    environment = os.environ.copy()
    environment.pop("MPI4PY_RC_FINALIZE", None)
    environment.pop("FI_PROVIDER", None)
    environment.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    environment.update(
        PYTHONPATH=str(ROOT) + os.pathsep + environment.get("PYTHONPATH", ""),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
    )
    return subprocess.run(
        command, env=environment, cwd=output.parent, capture_output=True, text=True, timeout=120
    )


def _success(result):
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Traceback" not in result.stderr + result.stdout


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.slow
@pytest.mark.parametrize(
    "direction,start_cell",
    (
        pytest.param(-1, 22, id="negative-outer"),
        pytest.param(1, 18, id="positive-internal"),
        pytest.param(-1, 42, id="negative-internal"),
    ),
)
@pytest.mark.parametrize("kind", ("hertzian", "magnetic", "rx"))
def test_stepped_objects_keep_their_origin_after_rank_migration(
    tmp_path, axis, direction, start_cell, kind
):
    partition = [1, 1, 1]
    partition[axis] = 3
    kwargs = dict(axis=axis, direction=direction, start_cell=start_cell, kind=kind)
    serial, distributed = tmp_path / "serial", tmp_path / "mpi"
    _success(_run("scan", serial, **kwargs))
    _success(_run("scan", distributed, partition, **kwargs))
    for trace in range(1, 5):
        with h5py.File(f"{serial}{trace}.h5") as reference, h5py.File(
            f"{distributed}{trace}.h5"
        ) as actual:
            np.testing.assert_array_equal(
                actual["rxs/rx1"].attrs["Position"], reference["rxs/rx1"].attrs["Position"]
            )
            for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
                np.testing.assert_allclose(
                    actual[f"rxs/rx1/{component}"][...],
                    reference[f"rxs/rx1/{component}"][...],
                    rtol=2e-12,
                    atol=1e-12,
                )
    restart = tmp_path / "restart"
    _success(_run("scan", restart, partition, restart=2, **kwargs))
    for trace in range(2, 5):
        with h5py.File(f"{serial}{trace}.h5") as reference, h5py.File(
            f"{restart}{trace}.h5"
        ) as actual:
            np.testing.assert_array_equal(
                actual["rxs/rx1"].attrs["Position"], reference["rxs/rx1"].attrs["Position"]
            )
            np.testing.assert_allclose(
                actual["rxs/rx1/Ez"][...], reference["rxs/rx1/Ez"][...], rtol=2e-12, atol=1e-12
            )


@pytest.mark.parametrize(
    "partition", ((2, 1, 1), (1, 2, 1), (1, 1, 2), pytest.param((2, 2, 2), marks=pytest.mark.slow))
)
@pytest.mark.parametrize("precision", ("single", "double"))
def test_network_port_outputs_match_serial(tmp_path, partition, precision):
    serial, distributed = tmp_path / "serial", tmp_path / "mpi"
    _success(_run("network", serial, precision=precision))
    _success(_run("network", distributed, partition, precision=precision))
    with h5py.File(serial.with_suffix(".h5")) as reference, h5py.File(
        distributed.with_suffix(".h5")
    ) as actual:
        assert set(actual["ports"]) == {"load"}
        for key in reference["ports/load"]:
            np.testing.assert_allclose(
                actual[f"ports/load/{key}"][...],
                reference[f"ports/load/{key}"][...],
                rtol=2e-5 if precision == "single" else 2e-12,
                atol=1e-12,
                equal_nan=True,
            )


def test_mpi_errors_return_nonzero(tmp_path):
    result = _run("bad", tmp_path / "bad", (2, 1, 1))
    assert result.returncode != 0
    assert "missing_waveform" in result.stdout + result.stderr
    assert not (tmp_path / "bad.h5").exists()


@pytest.mark.parametrize("repeat", range(3))
def test_taskfarm_finishes_good_jobs_but_reports_failed_jobs(tmp_path, repeat):
    result = _run("taskfarm", tmp_path / "farm")
    assert result.returncode != 0
    assert "1 task-farm job(s) failed: job 1" in result.stdout + result.stderr
    assert "missing_waveform" in result.stdout + result.stderr
    assert not (tmp_path / "farm1.h5").exists()
    assert (tmp_path / "farm2.h5").is_file()


@pytest.mark.parametrize("result_kind", ["success", "failure", "caught"])
def test_taskfarm_collective_completion_and_catchable_errors(tmp_path, result_kind):
    result = _run("taskfarm", tmp_path / "farm", farm_result=result_kind)
    if result_kind == "failure":
        assert result.returncode != 0
        assert "2 task-farm job(s) failed" in result.stdout + result.stderr
        assert not list(tmp_path.glob("farm*.h5"))
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        if result_kind == "success":
            _success(result)
            assert all((tmp_path / f"farm{index}.h5").is_file() for index in (1, 2))
        else:
            assert all((tmp_path / f"caught_rank{rank}.txt").is_file() for rank in range(3))
            assert (tmp_path / "farm2.h5").is_file()


@pytest.mark.parametrize("averaging", ("y", "n"))
def test_parallel_geometry_export_retains_cell_materials_and_tags(tmp_path, averaging):
    if not h5py.get_config().mpi:
        pytest.skip("requires parallel HDF5")
    # Identical basenames preserve the database namespace on re-import.
    serial, distributed = tmp_path / "serial/model", tmp_path / "mpi/model"
    _success(_run("export", serial, averaging=averaging))
    _success(_run("export", distributed, (2, 1, 1), averaging=averaging))
    # The worker imports and re-exports the geometry, exercising the public reader too.
    for suffix in ("_geometry.h5", "_roundtrip.h5"):
        with h5py.File(str(serial) + suffix) as reference, h5py.File(
            str(distributed) + suffix
        ) as actual:
            for key in ("material_keys", "data", "ID", "rigidE", "rigidH", "tag_names", "tag_data"):
                np.testing.assert_array_equal(actual[key][...], reference[key][...])
        serial_materials = json.loads(
            Path(str(serial) + suffix.replace(".h5", "_materials.json")).read_text()
        )["materials"]
        mpi_materials = json.loads(
            Path(str(distributed) + suffix.replace(".h5", "_materials.json")).read_text()
        )["materials"]
        for catalogue in (serial_materials, mpi_materials):
            for entry in catalogue.values():
                provenance = entry.get("metadata", {}).get("source_database", {})
                if "source" in provenance:
                    source = Path(provenance["source"])
                    assert source.is_file()
                    # Each re-import correctly records its own directory.
                    # Compare every other field, including entry hashes.
                    provenance["source"] = source.name
        assert mpi_materials == serial_materials


@pytest.mark.parametrize("precision", ("single", "double"))
@pytest.mark.slow
def test_parallel_snapshot_preserves_dtype_and_values(tmp_path, precision):
    if not h5py.get_config().mpi:
        pytest.skip("requires parallel HDF5")
    serial, distributed = tmp_path / "serial", tmp_path / "mpi"
    _success(_run("snapshots", serial, precision=precision))
    _success(_run("snapshots", distributed, (2, 2, 2), precision=precision))
    for stride in (1, 2):
        for extension, group in ((".h5", ""), (".vtkhdf", "VTKHDF/CellData/")):
            with h5py.File(
                tmp_path / f"serial_snaps/fields{stride}{extension}"
            ) as reference, h5py.File(tmp_path / f"mpi_snaps/fields{stride}{extension}") as actual:
                for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
                    key = group + component
                    assert actual[key].dtype == np.dtype(
                        "float32" if precision == "single" else "float64"
                    )
                    np.testing.assert_array_equal(actual[key][...], reference[key][...])


def _worker(args):
    import gprMax

    output = Path(args.output)
    dl = 0.002
    size = np.array([0.024] * 3)
    if args.case == "scan":
        size[args.axis] = 0.120

    def base():
        scene = gprMax.Scene()
        scene.add(gprMax.Discretisation(p1=(dl,) * 3))
        scene.add(gprMax.Domain(p1=tuple(size)))
        scene.add(gprMax.PMLThickness(thickness=2))
        scene.add(gprMax.TimeWindow(iterations=120))
        return scene

    scene = base()
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=6e9, id="pulse"))
    options = dict(
        outputfile=output,
        cpu_precision=args.precision,
        hide_progress_bars=True,
        log_level=30,
        mpi=tuple(args.partition) if args.partition else None,
    )
    if args.case == "taskfarm":
        scene.add(
            gprMax.HertzianDipole(p1=(0.010,) * 3, polarisation="z", waveform_id="missing_waveform")
        )
        good = base()
        good.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=6e9, id="pulse"))
        good.add(gprMax.HertzianDipole(p1=(0.010,) * 3, polarisation="z", waveform_id="pulse"))
        good.add(gprMax.Rx(p1=(0.012,) * 3))
        models = [good, good] if args.farm_result == "success" else [scene, scene] if args.farm_result == "failure" else [scene, good]
        if args.farm_result == "caught":
            from gprMax.taskfarm import TaskfarmError
            from mpi4py import MPI

            try:
                gprMax.run(scenes=models, n=2, taskfarm=True, **options)
            except TaskfarmError as error:
                assert set(error.failures) == {0}
                # The communicator remains usable and the exception is
                # catchable on every rank; no implicit MPI.Abort is needed.
                MPI.COMM_WORLD.Barrier()
                (output.parent / f"caught_rank{MPI.COMM_WORLD.rank}.txt").write_text(str(error))
            else:
                raise AssertionError("Every rank must receive the failed-batch exception")
        else:
            gprMax.run(scenes=models, n=2, taskfarm=True, **options)
        return
    if args.case == "scan":
        position = np.array([0.012] * 3)
        position[args.axis] = (
            args.start_cell * dl
            if args.start_cell is not None
            else (0.036 if args.direction > 0 else 0.044)
        )
        source = gprMax.MagneticDipole if args.kind == "magnetic" else gprMax.HertzianDipole
        scene.add(source(p1=tuple(position), polarisation="z", waveform_id="pulse"))
        receiver = position.copy()
        receiver[(args.axis + 1) % 3] += dl
        scene.add(gprMax.Rx(p1=tuple(receiver)))
        step = np.zeros(3)
        step[args.axis] = args.direction * 2 * dl
        scene.add(gprMax.RxSteps(p1=tuple(step)))
        if args.kind != "rx":
            scene.add(gprMax.SrcSteps(p1=tuple(step)))
        options.update(n=4, i=args.restart, geometry_fixed=True)
    elif args.case == "network":
        scene.add(gprMax.RationalNetwork(id="network", conductance=0.02, capacitance=0.2e-12))
        scene.add(
            gprMax.NetworkTerminal(
                p1=(0.012,) * 3, polarisation="z", network_id="network", id="load"
            )
        )
        scene.add(gprMax.NetworkExcitation("load", "pulse"))
        scene.add(gprMax.NetworkPort("load", spectrum_limit="nyquist"))
    elif args.case == "export":
        scene.add(gprMax.Material(er=4, se=0, mr=1, sm=0, id="voxel"))
        scene.add(gprMax.MaterialDensity(material_ids=["voxel"], density=1000))
        scene.add(
            gprMax.Box(
                p1=(0.010,) * 3,
                p2=(0.012,) * 3,
                material_id="voxel",
                averaging=args.averaging,
                tag="target",
            )
        )
        scene.add(
            gprMax.GeometryObjectsWrite(
                p1=(0, 0, 0), p2=tuple(size), filename=str(output) + "_geometry"
            )
        )
        options["geometry_only"] = True
    else:
        scene.add(
            gprMax.HertzianDipole(
                p1=(0.010,) * 3,
                polarisation="z",
                waveform_id="missing_waveform" if args.case == "bad" else "pulse",
            )
        )
        if args.case == "snapshots":
            for stride in (1, 2):
                for extension in (".h5", ".vtkhdf"):
                    scene.add(
                        gprMax.Snapshot(
                            p1=(0.004,) * 3,
                            p2=(0.020,) * 3,
                            dl=(dl * stride,) * 3,
                            filename=f"fields{stride}",
                            iterations=80,
                            fileext=extension,
                        )
                    )
    gprMax.run(scenes=[scene], **options)
    if args.case == "export":
        imported = base()
        imported.add(
            gprMax.GeometryObjectsRead(
                p1=(0, 0, 0),
                geofile=str(output) + "_geometry.h5",
                material_database=output.name + "_geometry_materials",
            )
        )
        imported.add(
            gprMax.GeometryObjectsWrite(
                p1=(0, 0, 0), p2=tuple(size), filename=str(output) + "_roundtrip"
            )
        )
        gprMax.run(scenes=[imported], **options)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "case", choices=("scan", "network", "bad", "export", "snapshots", "taskfarm")
    )
    parser.add_argument("output")
    parser.add_argument("--partition", type=int, nargs=3)
    parser.add_argument("--precision", default="double")
    parser.add_argument("--axis", type=int, default=0)
    parser.add_argument("--direction", type=int, default=1)
    parser.add_argument("--start-cell", type=int)
    parser.add_argument("--kind", default="hertzian")
    parser.add_argument("--restart", type=int, default=1)
    parser.add_argument("--averaging", default="y")
    parser.add_argument("--farm-result", default="mixed", choices=("mixed", "success", "failure", "caught"))
    _worker(parser.parse_args())
