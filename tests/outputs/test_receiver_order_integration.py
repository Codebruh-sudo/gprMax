"""Actual solver/gather/reuse checks, with labels that defeat ID sorting."""

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
MPIEXEC = str(Path(sys.executable).parent / "mpiexec")
if not Path(MPIEXEC).is_file():
    MPIEXEC = shutil.which("mpiexec")


def run_model(tmp_path, name, extra=()):
    model = tmp_path / "receivers.in"
    model.write_text(
        "\n".join(
            (
                "#domain: 0.04 0.04 0.04",
                "#dx_dy_dz: 0.001 0.001 0.001",
                "#time_window: 80",
                "#pml_cells: 0",
                "#omp_threads: 1",
                "#waveform: ricker 1 2e10 pulse",
                "#voltage_source: z 0.012 0.012 0.012 50 pulse",
                "#rx: 0.018 0.018 0.018 zulu Ex Ey Ez Hx Hy Hz Ix Iy Iz",
                "#rx: 0.022 0.022 0.022 alpha Ez",
                "#rx: 0.018 0.018 0.018 duplicate Ez",
                "#rx: 0.018 0.018 0.018 duplicate Ez",
                "#rx_array: 0.007 0.016 0.018 0.027 0.016 0.018 0.002 0 0",
                "#rx_steps: 0.003 0.003 0.003",
                "",
            )
        )
    )
    command = [
        sys.executable,
        "-m",
        "gprMax",
        str(model),
        "-n",
        "3",
        "--geometry-fixed",
        "--hide-progress-bars",
        "-cpu_precision",
        "double",
        "-o",
        str(tmp_path / name),
        *extra,
    ]
    if "--mpi" in extra:
        command = [MPIEXEC, "-n", "2", *command]
    environment = os.environ.copy()
    environment.pop("MPI4PY_RC_FINALIZE", None)
    environment.pop("FI_PROVIDER", None)
    environment.update(
        PYTHONPATH=str(ROOT),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OMPI_MCA_rmaps_base_oversubscribe="1",
    )
    result = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return [tmp_path / f"{name}{run}.h5" for run in range(1, 4)]


def check_order(files):
    layouts = []
    for run, path in enumerate(files):
        with h5py.File(path) as output:
            assert output.attrs["ReceiverOrder"] == "construction"
            assert output.attrs["nrx"] == 15  # private voltage monitor excluded
            layouts.append(output.attrs["ReceiverLayout"])
            names = [output[f"rxs/rx{i}"].attrs["Name"] for i in range(1, 16)]
            assert names[:4] == ["zulu", "alpha", "duplicate", "duplicate"]
            assert [output[f"rxs/rx{i}"].attrs["BuildIndex"] for i in range(1, 16)] == list(range(15))
            positions = [output[f"rxs/rx{i}"].attrs["Position"] for i in range(5, 16)]
            expected = np.array([(x * 0.001, 0.016, 0.018) for x in range(7, 28, 2)]) + run * 0.003
            np.testing.assert_allclose(positions, expected, rtol=0, atol=1e-16)
            assert output["rxs/rx1"].attrs["NameKind"] == "user"
            assert output["rxs/rx5"].attrs["NameKind"] == "generated"
            assert "ports/port1" in output
    assert len(set(layouts)) == 1


@pytest.mark.integration
def test_serial_reuse_preserves_array_and_duplicate_receiver_order(tmp_path):
    check_order(run_model(tmp_path, "serial"))


@pytest.mark.integration
@pytest.mark.skipif(
    MPIEXEC is None or importlib.util.find_spec("mpi4py") is None, reason="requires MPI launcher and mpi4py"
)
@pytest.mark.parametrize("partition", [(2, 1, 1), (1, 2, 1), (1, 1, 2)])
def test_mpi_gather_and_migration_match_serial_identity_and_samples(tmp_path, partition):
    serial = run_model(tmp_path, "serial")
    distributed = run_model(tmp_path, "mpi", ("--mpi", *map(str, partition)))
    check_order(serial)
    check_order(distributed)
    for reference, candidate in zip(serial, distributed):
        with h5py.File(reference) as left, h5py.File(candidate) as right:
            assert left.attrs["ReceiverLayout"] == right.attrs["ReceiverLayout"]
            for rx in left["rxs"]:
                for key in left[f"rxs/{rx}"].attrs:
                    np.testing.assert_array_equal(left[f"rxs/{rx}"].attrs[key], right[f"rxs/{rx}"].attrs[key])
                for component in left[f"rxs/{rx}"]:
                    np.testing.assert_allclose(
                        left[f"rxs/{rx}/{component}"], right[f"rxs/{rx}/{component}"], rtol=2e-12, atol=1e-12
                    )


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize("backend", ("cuda", "opencl"))
def test_device_receiver_pages_keep_identity_across_reuse(tmp_path, request, backend):
    device = request.getfixturevalue("gpu_device" if backend == "cuda" else "opencl_device")
    serial = run_model(tmp_path, "serial")
    option = "-gpu" if backend == "cuda" else "-opencl"
    accelerated = run_model(tmp_path, backend, (option, str(device), "-gpu_precision", "double"))
    check_order(accelerated)
    for reference, candidate in zip(serial, accelerated):
        with h5py.File(reference) as left, h5py.File(candidate) as right:
            for key in left["rxs"]:
                assert left[f"rxs/{key}"].attrs["Name"] == right[f"rxs/{key}"].attrs["Name"]
                for comp in left[f"rxs/{key}"]:
                    expected = left[f"rxs/{key}/{comp}"][:]
                    # Symmetry-null channels contain roundoff, not a signal
                    # against which a relative tolerance can be normalised.
                    np.testing.assert_allclose(
                        right[f"rxs/{key}/{comp}"],
                        expected,
                        rtol=2e-10,
                        atol=2e-12,
                        err_msg=f"{candidate.name}: {key}/{comp}",
                    )
