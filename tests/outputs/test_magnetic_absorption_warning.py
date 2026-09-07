"""Magnetic-loss limitations are visible without changing electric absorption."""

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import gprMax
from gprMax.materials import Material, create_directional_material
from gprMax.sar import _warn_magnetic_absorption


def _material(number, name, *, mr=1, sm=0):
    material = Material(number, name)
    material.mr, material.sm = mr, sm
    return material


@pytest.mark.unit
@pytest.mark.parametrize("kind", ("SAR", "Radiometry"))
@pytest.mark.parametrize("mr,sm", ((1, 100), (4, 0), (4, 100)))
def test_selected_magnetic_material_warns_once(caplog, kind, mr, sm):
    grid = SimpleNamespace(materials=[_material(3, "medium", mr=mr, sm=sm)])
    with caplog.at_level(logging.WARNING, logger="gprMax.sar"):
        _warn_magnetic_absorption(grid, [3, 3, 3], output_kind=kind, output_id="sample")
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert f"{kind} 'sample' selects magnetic material(s): 'medium'" in message
    assert "electric absorption only; magnetic absorption is not included" in message
    assert "Magnetic properties still affect the FDTD fields" in message
    assert ("do not interpret these results as total" in message) == (sm != 0)
    assert ("lossless permeability alone" in message) == (sm == 0)


@pytest.mark.unit
@pytest.mark.parametrize("selection", ([], [1], [2], [3]))
def test_unused_nonmagnetic_and_ideal_pmc_do_not_warn(caplog, selection):
    grid = SimpleNamespace(
        materials=[
            _material(0, "unused", mr=4, sm=100),
            _material(1, "tissue"),
            _material(2, "pmc", sm=float("inf")),
            _material(3, "custom_pmc", sm=float("inf")),
        ]
    )
    _warn_magnetic_absorption(grid, selection, output_kind="SAR", output_id="sample")
    assert not caplog.records


@pytest.mark.unit
def test_directional_pmc_does_not_hide_finite_magnetic_loss(caplog):
    grid = SimpleNamespace(
        materials=[
            _material(0, "pmc", sm=float("inf")),
            _material(1, "lossless", mr=4),
            _material(2, "lossy", sm=100),
        ]
    )
    tensor = create_directional_material(grid, grid.materials)
    assert tensor.is_pmc  # Its scalar mean must not hide individual constituents.
    _warn_magnetic_absorption(grid, [tensor.numID], output_kind="SAR", output_id="sample")
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "selects magnetic material(s): 'lossless', 'lossy'" in message
    assert "Nonzero magnetic conductivity is present in 'lossy'" in message


@pytest.mark.unit
def test_remote_selection_warns_when_local_rank_has_no_cells(caplog):
    calls = []

    def allgather(value):
        calls.append(value)
        return [value, (("remote",), ("remote",))]

    grid = SimpleNamespace(
        materials=[], lower_extent=(0, 0, 0), comm=SimpleNamespace(allgather=allgather)
    )
    _warn_magnetic_absorption(grid, [], output_kind="Radiometry", output_id="sample")
    assert calls == [((), ())]
    assert len(caplog.records) == 1
    assert "'remote'" in caplog.text


def _input(tmp_path, *, mr=1, sm=100, placement="selected"):
    # The target is entirely on the upper-x MPI rank; the default logging
    # rank owns no target cells. The last four x cells form the domain PML.
    lines = [
        "#domain: 0.048 0.032 0.032",
        "#dx_dy_dz: 0.002 0.002 0.002",
        "#time_window: 2e-9",
        "#pml_cells: 4",
        "#omp_threads: 1",
        "#material: 2 0.3 1 0 tissue",
        f"#material: 2 0.3 {mr} {sm} medium",
        "#material_density: 1000 tissue medium",
        "#box: 0.028 0.01 0.01 0.036 0.022 0.022 "
        + ("medium" if placement == "selected" else "tissue")
        + " n target",
        "#waveform: ricker 1 1e9 pulse",
        "#hertzian_dipole: z 0.018 0.016 0.016 pulse",
        "#sar: 1e9 1e9 1 pulse 1 10 dose target",
        "#radiometry: 1e9 1e9 1 pulse 1 10 rad target",
    ]
    if placement == "pml":
        lines.append("#box: 0.044 0.01 0.01 0.048 0.022 0.022 medium n target")
    elif placement == "unselected":
        lines.append("#box: 0.01 0.01 0.01 0.014 0.022 0.022 medium n other")
    path = tmp_path / "magnetic.in"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _assert_warnings(text, expected=True):
    for kind, name in (("SAR", "dose"), ("Radiometry", "rad")):
        assert text.count(f"{kind} '{name}' selects magnetic material(s)") == int(expected)
    if expected:
        assert "magnetic absorption is not included" in text


@pytest.mark.integration
@pytest.mark.parametrize(
    "mr,sm,placement",
    (
        (1, 100, "selected"),
        (4, 0, "selected"),
        (1, 0, "selected"),
        (1, 100, "pml"),
        (1, 100, "unselected"),
    ),
)
def test_cpu_warning_uses_selected_non_pml_materials(tmp_path, capsys, mr, sm, placement):
    model = _input(tmp_path, mr=mr, sm=sm, placement=placement)
    gprMax.run(inputfile=model, outputfile=tmp_path / "cpu", hide_progress_bars=True)
    _assert_warnings(capsys.readouterr().out, placement == "selected" and (mr != 1 or sm != 0))
    with h5py.File(tmp_path / "cpu.h5") as data:
        for path in ("sar/dose", "radiometry/rad"):
            assert np.all(data[path + "/valid"])
            assert np.all(np.isfinite(data[path + "/absorbed_power_density"]))
            assert np.max(data[path + "/absorbed_power_density"]) > 0


def _compare_warning_enabled_and_suppressed(tmp_path, capsys, monkeypatch, **options):
    model = _input(tmp_path)
    gprMax.run(inputfile=model, outputfile=tmp_path / "warning", hide_progress_bars=True, **options)
    _assert_warnings(capsys.readouterr().out)
    monkeypatch.setattr("gprMax.sar._warn_magnetic_absorption", lambda *args, **kwargs: None)
    gprMax.run(
        inputfile=model, outputfile=tmp_path / "reference", hide_progress_bars=True, **options
    )
    _assert_warnings(capsys.readouterr().out, False)
    with h5py.File(tmp_path / "warning.h5") as actual, h5py.File(
        tmp_path / "reference.h5"
    ) as reference:
        for path in (
            "sar/dose/sar",
            "radiometry/rad/absorbed_power_density",
            "radiometry/rad/normalised_absorption_density",
        ):
            np.testing.assert_array_equal(actual[path], reference[path])


@pytest.mark.integration
def test_cpu_warning_does_not_change_results(tmp_path, capsys, monkeypatch):
    _compare_warning_enabled_and_suppressed(tmp_path, capsys, monkeypatch, cpu_precision="double")


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ("cuda", "opencl"))
def test_device_warning_does_not_change_results(tmp_path, capsys, monkeypatch, request, backend):
    device = request.getfixturevalue("gpu_device" if backend == "cuda" else "opencl_device")
    options = {"gpu" if backend == "cuda" else "opencl": [device], "gpu_precision": "single"}
    _compare_warning_enabled_and_suppressed(tmp_path, capsys, monkeypatch, **options)


@pytest.mark.integration
def test_real_mpi_warning_for_remote_cells_and_geometry_reuse(tmp_path):
    launcher = Path(sys.executable).with_name("mpiexec")
    launcher = str(launcher) if launcher.is_file() else shutil.which("mpiexec")
    if launcher is None or not h5py.get_config().mpi:
        pytest.skip("MPI launcher and MPI-enabled HDF5 are required")
    model = _input(tmp_path)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", OMPI_MCA_rmaps_base_oversubscribe="1")
    result = subprocess.run(
        [
            launcher,
            "-n",
            "2",
            sys.executable,
            "-m",
            "gprMax",
            str(model),
            "--mpi",
            "2",
            "1",
            "1",
            "-n",
            "2",
            "--geometry-fixed",
            "--hide-progress-bars",
            "-o",
            str(tmp_path / "mpi"),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    _assert_warnings(result.stdout)
    with h5py.File(tmp_path / "mpi1.h5") as first, h5py.File(tmp_path / "mpi2.h5") as second:
        for path in ("sar/dose/sar", "radiometry/rad/absorbed_power_density"):
            assert np.max(first[path]) > 0
            np.testing.assert_array_equal(first[path], second[path])
