"""Real solves: passive receiving ports and intentional under-resolution."""

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


def _scene(mr=1, passive_frequency=None, active=True):
    scene = gprMax.Scene()
    for item in (
        gprMax.Domain(p1=(0.012,) * 3),
        gprMax.Discretisation(p1=(0.001,) * 3),
        gprMax.TimeWindow(time=6e-9),
        gprMax.PMLThickness(thickness=0),
        gprMax.OMPThreads(n=1),
        gprMax.Material(er=1, se=0, mr=mr, sm=0, id="medium"),
        gprMax.Box(p1=(0, 0, 0), p2=(0.012,) * 3, material_id="medium", averaging=False),
        gprMax.Rx(p1=(0.007, 0.007, 0.007), id="field"),
    ):
        scene.add(item)
    if active:
        scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=2e9, id="drive"))
        scene.add(
            gprMax.HertzianDipole(p1=(0.005, 0.005, 0.005), polarisation="z", waveform_id="drive")
        )
    if passive_frequency is not None:
        scene.add(gprMax.Waveform(wave_type="ricker", amp=0, freq=passive_frequency, id="receive"))
        scene.add(
            gprMax.VoltageSource(
                p1=(0.006, 0.005, 0.005),
                polarisation="z",
                resistance=50,
                waveform_id="receive",
                id="passive",
            )
        )
        scene.add(
            gprMax.TransmissionLine(
                p1=(0.007, 0.005, 0.005), polarisation="z", resistance=50, waveform_id="receive"
            )
        )
    return scene


def _run(path, mr=1, passive_frequency=None, active=True, **options):
    gprMax.run(
        scenes=[_scene(mr, passive_frequency, active)],
        outputfile=path,
        cpu_precision="double",
        hide_progress_bars=True,
        log_level=30,
        **options,
    )


def _histories(path):
    with h5py.File(path.with_suffix(".h5")) as output:
        values = {"field": output["rxs/rx1/Ez"][...]}
        if "ports/passive" in output:
            for group_name, fields in (
                ("ports/passive", ("Vtotal", "Vgenerator")),
                ("tls/tl1", ("Vtotal", "Vinc")),
            ):
                for field in fields:
                    values[f"{group_name}/{field}"] = output[f"{group_name}/{field}"][...]
        return values


def test_all_passive_ports_remain_and_record_zero_without_excitation(tmp_path, capsys):
    path = tmp_path / "passive"
    _run(path, active=False, passive_frequency=2e11)
    data = _histories(path)
    assert len(data) == 5
    for values in data.values():
        np.testing.assert_array_equal(values, 0)
    assert "Insufficient spatial resolution" not in capsys.readouterr().out


def _check_passive_pair(tmp_path, **options):
    reference_path = tmp_path / "normal_passive"
    high_path = tmp_path / "high_passive"
    _run(reference_path, passive_frequency=2e9, **options)
    _run(high_path, passive_frequency=2e11, **options)
    reference, actual = _histories(reference_path), _histories(high_path)
    for key in actual:
        np.testing.assert_array_equal(actual[key], reference[key])
    assert np.max(np.abs(actual["ports/passive/Vtotal"])) > 0
    assert np.max(np.abs(actual["tls/tl1/Vtotal"])) > 0
    np.testing.assert_array_equal(actual["ports/passive/Vgenerator"], 0)
    np.testing.assert_array_equal(actual["tls/tl1/Vinc"], 0)


def test_passive_voltage_and_tl_still_receive_identical_signals(tmp_path):
    _check_passive_pair(tmp_path)


def test_override_does_not_change_a_resolved_model(tmp_path):
    paths = [tmp_path / name for name in ("default", "override")]
    _run(paths[0])
    _run(paths[1], allow_underresolved=True)
    np.testing.assert_array_equal(_histories(paths[0])["field"], _histories(paths[1])["field"])


@pytest.mark.parametrize("enabled", (False, True))
def test_hash_model_cli_override(tmp_path, enabled):
    model = tmp_path / "model.in"
    model.write_text(
        "#domain: 0.012 0.012 0.012\n"
        "#dx_dy_dz: 0.001 0.001 0.001\n"
        "#time_window: 6e-9\n"
        "#pml_cells: 0\n"
        "#omp_threads: 1\n"
        "#material: 1 0 400 0 medium\n"
        "#box: 0 0 0 0.012 0.012 0.012 medium n\n"
        "#waveform: ricker 1 2e9 drive\n"
        "#hertzian_dipole: z 0.005 0.005 0.005 drive\n"
        "#rx: 0.007 0.007 0.007\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "-m",
        "gprMax",
        str(model),
        "--hide-progress-bars",
        "--log-level",
        "30",
    ]
    if enabled:
        command.append("--allow-underresolved")
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert model.with_suffix(".h5").exists() is enabled
    if enabled:
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Continuing with an under-resolved mesh" in result.stdout + result.stderr
    else:
        assert result.returncode != 0
        assert "Insufficient spatial resolution" in result.stdout + result.stderr


def test_override_does_not_disable_sar_output_sampling_guard(tmp_path):
    scene = _scene(mr=400)
    scene.add(gprMax.MaterialDensity(density=1000, material_ids="medium"))
    scene.add(
        gprMax.Box(
            p1=(0.007,) * 3, p2=(0.009,) * 3, material_id="medium", averaging=False, tag="selected"
        )
    )
    scene.add(gprMax.SAR(frequencies=(2e9,), tags="selected", waveform_id="drive", id="sar"))
    with pytest.raises(ValueError, match="SAR frequency.*requires at least lambda/10"):
        gprMax.run(
            scenes=[scene],
            outputfile=tmp_path / "sar",
            allow_underresolved=True,
            hide_progress_bars=True,
            log_level=30,
        )


def test_override_allows_reused_underresolved_geometry_without_persisting(tmp_path, capsys):
    with pytest.raises(ValueError, match="Insufficient spatial resolution"):
        _run(tmp_path / "rejected", mr=400)
    assert not (tmp_path / "rejected.h5").exists()
    _run(tmp_path / "reuse", mr=400, allow_underresolved=True, n=2, geometry_fixed=True)
    first = _histories(tmp_path / "reuse1")["field"]
    second = _histories(tmp_path / "reuse2")["field"]
    assert np.all(np.isfinite(first))
    assert np.max(np.abs(first)) > 0
    np.testing.assert_array_equal(first, second)
    assert "Continuing with an under-resolved mesh" in capsys.readouterr().out
    # A new run without the option must restore the normal rejection.
    with pytest.raises(ValueError, match="Insufficient spatial resolution"):
        _run(tmp_path / "rejected_again", mr=400)


def test_override_reaches_embedded_grid_diagnostic(tmp_path, capsys):
    def scene():
        result = gprMax.Scene()
        for item in (
            gprMax.Domain(p1=(0.064,) * 3),
            gprMax.Discretisation(p1=(0.002,) * 3),
            gprMax.TimeWindow(time=6e-9),
            gprMax.OMPThreads(n=1),
            gprMax.PMLThickness(thickness=4),
        ):
            result.add(item)
        local = gprMax.SubGridHSG(p1=(0.020,) * 3, p2=(0.044,) * 3, ratio=1, id="local")
        for item in (
            gprMax.Material(er=1, se=0, mr=400, sm=0, id="magnetic"),
            gprMax.Box(p1=(0.024,) * 3, p2=(0.040,) * 3, material_id="magnetic"),
            gprMax.Waveform(wave_type="ricker", amp=1, freq=1e9, id="drive"),
            gprMax.HertzianDipole(p1=(0.032,) * 3, polarisation="z", waveform_id="drive"),
        ):
            local.add(item)
        result.add(local)
        return result

    options = dict(
        subgrid=True, autotranslate=True, geometry_only=True, hide_progress_bars=True, log_level=30
    )
    with pytest.raises(ValueError, match="Insufficient spatial resolution"):
        gprMax.run(scenes=[scene()], outputfile=tmp_path / "rejected", **options)
    gprMax.run(
        scenes=[scene()], outputfile=tmp_path / "allowed", allow_underresolved=True, **options
    )
    assert "[local] Continuing with an under-resolved mesh" in capsys.readouterr().out


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ("cuda", "opencl"))
def test_device_passive_ports_and_override(tmp_path, request, backend):
    device = request.getfixturevalue("gpu_device" if backend == "cuda" else "opencl_device")
    options = {"gpu" if backend == "cuda" else "opencl": [device], "gpu_precision": "double"}
    _check_passive_pair(tmp_path, **options)
    with pytest.raises(ValueError, match="Insufficient spatial resolution"):
        _run(tmp_path / "rejected", mr=400, **options)
    _run(tmp_path / "allowed", mr=400, allow_underresolved=True, **options)
    assert np.all(np.isfinite(_histories(tmp_path / "allowed")["field"]))


@pytest.mark.parametrize("case", ("rejected", "allowed", "passive"))
def test_mpi_override_and_passive_ports(tmp_path, case):
    launcher = shutil.which("mpiexec", path=str(Path(sys.executable).parent))
    if launcher is None:
        pytest.skip("MPI runtime is not available")
    pytest.importorskip("mpi4py")
    path = tmp_path / "mpi"
    env = os.environ.copy()
    env.pop("MPI4PY_RC_FINALIZE", None)
    env.pop("FI_PROVIDER", None)
    env.update(
        PYTHONPATH=str(Path(__file__).resolve().parents[2]),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        OMPI_MCA_rmaps_base_oversubscribe="1",
    )
    result = subprocess.run(
        [launcher, "-n", "2", sys.executable, __file__, str(path), case],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    if case == "rejected":
        assert result.returncode != 0
        assert "Insufficient spatial resolution" in result.stdout + result.stderr
        assert not path.with_suffix(".h5").exists()
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        actual = _histories(path)
        if case == "allowed":
            assert "Continuing with an under-resolved mesh" in result.stdout + result.stderr
            _run(tmp_path / "cpu", mr=400, allow_underresolved=True)
        else:
            _run(tmp_path / "cpu", passive_frequency=2e11)
        reference = _histories(tmp_path / "cpu")
        for key, values in actual.items():
            np.testing.assert_allclose(values, reference[key], rtol=2e-10, atol=1e-13)


if __name__ == "__main__":
    case = sys.argv[2]
    options = dict(mpi=(2, 1, 1))
    if case == "passive":
        options["passive_frequency"] = 2e11
    else:
        options.update(mr=400, allow_underresolved=(case == "allowed"))
    _run(Path(sys.argv[1]), **options)
