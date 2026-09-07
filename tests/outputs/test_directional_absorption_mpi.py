"""Real distributed absorption with different generated tensors on each rank."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


def _launcher():
    local = Path(sys.executable).with_name("mpiexec")
    return str(local) if local.is_file() else shutil.which("mpiexec")


@pytest.mark.integration
@pytest.mark.skipif(_launcher() is None, reason="mpiexec is not installed")
@pytest.mark.parametrize("decomposition", ((2, 1, 1), (1, 2, 1)))
def test_directional_mpi_and_geometry_fixed_match_serial(tmp_path, decomposition):
    model = tmp_path / "directional.in"
    export = tmp_path / "export"
    model.write_text(
        "\n".join(
            (
                "#domain: 0.048 0.032 0.032",
                "#dx_dy_dz: 0.002 0.002 0.002",
                "#time_window: 2e-9",
                "#pml_cells: 2",
                "#omp_threads: 1",
                "#material: 2 0.1 1 0 x",
                "#material: 2 0.3 1 0 y",
                "#material: 2 0.6 1 0 z",
                "#material_density: 1000 x y z",
                "#add_dispersion_debye: 1 2 1e-10 z",
                "#box: 0.014 0.01 0.01 0.022 0.022 0.022 x y z n target",
                "#box: 0.028 0.01 0.01 0.036 0.022 0.022 z y x n target",
                f"#geometry_objects_write: 0 0 0 0.048 0.032 0.032 {export}",
                "#waveform: ricker 1 1e9 pulse",
                "#hertzian_dipole: z 0.01 0.014 0.014 pulse",
                "#sar: 0.75e9 1.25e9 3 pulse 1 10 dose target",
                "#radiometry: 0.75e9 1.25e9 3 pulse 1 10 rad target",
                "",
            )
        ),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.pop("MPI4PY_RC_FINALIZE", None)
    environment.update(FI_PROVIDER="shm", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

    def run(arguments):
        result = subprocess.run(
            arguments, env=environment, cwd=tmp_path, capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, result.stdout + result.stderr

    common = [
        sys.executable,
        "-m",
        "gprMax",
        str(model),
        "--hide-progress-bars",
        "-cpu_precision",
        "double",
    ]
    run([*common, "-o", str(tmp_path / "serial")])
    run(
        [
            _launcher(),
            "-n",
            "2",
            *common,
            "--mpi",
            *map(str, decomposition),
            "-n",
            "2",
            "--geometry-fixed",
            "-o",
            str(tmp_path / "mpi"),
        ]
    )
    # The last export was made collectively from the distributed geometry.
    imported = tmp_path / "imported.in"
    lines = model.read_text().splitlines()
    lines = [line for line in lines if not line.startswith(("#box:", "#geometry_objects_write:"))]
    lines.insert(0, f"#geometry_objects_read: 0 0 0 {export}.h5 export_materials n")
    imported.write_text("\n".join(lines) + "\n", encoding="utf-8")
    run(
        [
            sys.executable,
            "-m",
            "gprMax",
            str(imported),
            "--hide-progress-bars",
            "-cpu_precision",
            "double",
            "-o",
            str(tmp_path / "imported"),
        ]
    )
    with h5py.File(tmp_path / "serial.h5") as reference:
        for path in ("mpi1.h5", "mpi2.h5", "imported.h5"):
            with h5py.File(tmp_path / path) as actual:
                for output, value in (
                    ("sar/dose", "sar"),
                    ("radiometry/rad", "absorbed_power_density"),
                ):
                    np.testing.assert_array_equal(
                        actual[output + "/cell_indices"], reference[output + "/cell_indices"]
                    )
                    np.testing.assert_allclose(
                        actual[output + "/" + value],
                        reference[output + "/" + value],
                        rtol=2e-11,
                        atol=1e-16,
                    )
                    if path.startswith("mpi"):
                        group = actual[output]
                        assert len(group["materials/name"]) == 2
                        assert len(np.unique(group["material_id"])) == 2
