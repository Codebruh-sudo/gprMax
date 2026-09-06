# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

"""Real CUDA regressions for per-family conventional-source constants."""

from itertools import combinations

import h5py
import numpy as np
import pytest

import gprMax
from gprMax.updates.cuda_updates import CUDAUpdates

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

FAMILIES = ("electric", "magnetic", "voltage")
SUBSETS = tuple(subset for length in (1, 2, 3) for subset in combinations(FAMILIES, length))
COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


@pytest.fixture
def checked_source_modules(monkeypatch):
    """Read each actual module's constant memory before the context closes."""
    checked = []
    original = CUDAUpdates._set_src_knls

    def initialise(updates):
        compiler = updates.source_module
        modules = []

        def capture(*args, **kwargs):
            module = compiler(*args, **kwargs)
            modules.append(module)
            return module

        updates.source_module = capture
        try:
            original(updates)
        finally:
            updates.source_module = compiler

        families = [
            family
            for family, sources in zip(
                FAMILIES,
                (updates.grid.hertziandipoles, updates.grid.magneticdipoles, updates.grid.voltagesources),
            )
            if sources
        ]
        assert len(modules) == len(families)
        for family, module in zip(families, modules):
            for name in ("updatecoeffsE", "updatecoeffsH"):
                expected = getattr(updates.grid, name)
                address, nbytes = module.get_global(name)
                assert nbytes == expected.nbytes
                actual = np.empty_like(expected)
                updates.drv.memcpy_dtoh(actual, address)
                np.testing.assert_array_equal(actual, expected, err_msg=f"{family} module {name}")
                assert actual.tobytes() == expected.tobytes()
            checked.append(family)

    monkeypatch.setattr(CUDAUpdates, "_set_src_knls", initialise)
    return checked


def _scene(families, active):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(0.001,) * 3))
    scene.add(gprMax.Domain(p1=(0.020, 0.022, 0.024)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.OMPThreads(n=1))
    scene.add(gprMax.TimeWindow(time=1e-10))
    sources = {}
    specs = {
        "electric": (gprMax.HertzianDipole, (0.008, 0.010, 0.012), "z", 1e-3),
        "magnetic": (gprMax.MagneticDipole, (0.012, 0.010, 0.012), "x", 1e-3),
        "voltage": (gprMax.VoltageSource, (0.010, 0.014, 0.012), "z", 0.7),
    }
    for family in families:
        source_class, position, polarisation, amplitude = specs[family]
        waveform = f"wave_{family}"
        scene.add(
            gprMax.Waveform(wave_type="gaussian", amp=amplitude if family in active else 0, freq=2e10, id=waveform)
        )
        options = {"resistance": 50, "id": "voltage_port"} if family == "voltage" else {}
        source = source_class(p1=position, polarisation=polarisation, waveform_id=waveform, **options)
        sources[family] = source
        scene.add(source)
    # Separate points, waveform IDs and default full windows isolate the
    # module-initialisation defect from cache/window and co-location issues.
    scene.add(gprMax.Rx(p1=(0.010, 0.011, 0.012), id="field"))
    scene.add(gprMax.Rx(p1=(0.013, 0.014, 0.015), id="off_axis"))
    return scene, sources


def _read(path):
    result = {}
    with h5py.File(str(path) + ".h5", "r") as output:
        for rx_id, receiver in output["rxs"].items():
            for component in COMPONENTS:
                if component in receiver:
                    result[f"rxs/{rx_id}/{component}"] = receiver[component][:]
        if "ports" in output:
            for port_id, port in output["ports"].items():
                for name in ("time", "Vgenerator", "Vtotal"):
                    result[f"ports/{port_id}/{name}"] = port[name][:]
    return result


def _run(path, families, active, precision, *, gpu_device=None):
    scene, _ = _scene(families, active)
    backend = {"cpu_precision": precision} if gpu_device is None else {"gpu": [gpu_device], "gpu_precision": precision}
    gprMax.run(scenes=[scene], outputfile=path, hide_progress_bars=True, log_level=30, **backend)
    return _read(path)


def _assert_parity(cpu, cuda, precision):
    assert cpu.keys() == cuda.keys()
    # These no-PML cases have a measured double-precision discrepancy below
    # 3.1e-14. The original reproducer includes PML and has a different backend
    # baseline; it does not justify relaxing this source-only regression.
    tolerance = 3e-5 if precision == "single" else 2e-12
    field_scales = {
        kind: max(
            float(np.max(np.abs(values)))
            for name, values in cpu.items()
            if name.startswith("rxs/") and name.rsplit("/", 1)[-1].startswith(kind)
        )
        for kind in ("E", "H")
    }
    assert min(field_scales.values()) > 1e-7
    for name, reference in cpu.items():
        assert np.isfinite(cuda[name]).all(), name
        if name.endswith("/time"):
            np.testing.assert_array_equal(cuda[name], reference)
            continue
        scale = max(float(np.max(np.abs(reference))), 1e-12)
        if name.startswith("rxs/"):
            scale = max(scale, field_scales[name.rsplit("/", 1)[-1][0]])
        np.testing.assert_allclose(cuda[name], reference, rtol=tolerance, atol=tolerance * scale, err_msg=name)


def _assert_bitwise(left, right):
    assert left.keys() == right.keys()
    for name, values in left.items():
        reference = right[name]
        np.testing.assert_array_equal(values, reference, err_msg=name)
        assert values.dtype == reference.dtype
        assert values.tobytes() == reference.tobytes(), name


@pytest.mark.parametrize("precision", ("single", "double"))
@pytest.mark.parametrize("families", SUBSETS, ids=lambda value: "+".join(value))
def test_cuda_mixed_source_families_match_cpu(tmp_path, gpu_device, checked_source_modules, precision, families):
    schedules = [families]
    if len(families) > 1:
        schedules.extend((family,) for family in families)
    for active in schedules:
        suffix = "+".join(active)
        cpu = _run(tmp_path / f"cpu_{suffix}", families, active, precision)
        cuda = _run(tmp_path / f"cuda_{suffix}", families, active, precision, gpu_device=gpu_device)
        _assert_parity(cpu, cuda, precision)
        if "voltage" in families:
            assert "ports/voltage_port/Vtotal" in cpu
            if "voltage" not in active:
                np.testing.assert_array_equal(cpu["ports/voltage_port/Vgenerator"], 0)
                assert np.max(np.abs(cpu["ports/voltage_port/Vtotal"])) > 1e-7
    assert checked_source_modules == list(families) * len(schedules)


@pytest.mark.parametrize("precision", ("single", "double"))
def test_zero_magnetic_source_is_exact_no_load_control(tmp_path, gpu_device, checked_source_modules, precision):
    for backend, device in (("cpu", None), ("cuda", gpu_device)):
        isolated = _run(tmp_path / f"{backend}_isolated", ("electric",), ("electric",), precision, gpu_device=device)
        mixed = _run(
            tmp_path / f"{backend}_zero_magnetic", ("electric", "magnetic"), ("electric",), precision, gpu_device=device
        )
        _assert_bitwise(isolated, mixed)
    assert checked_source_modules == ["electric", "electric", "magnetic"]


@pytest.mark.parametrize("precision", ("single", "double"))
def test_cuda_mixed_dipole_study_matches_fresh_scenes(tmp_path, gpu_device, checked_source_modules, precision):
    families = ("electric", "magnetic")
    schedules = (families, ("electric",), ("magnetic",))
    for backend, device in (("cpu", None), ("cuda", gpu_device)):
        scene, sources = _scene(families, families)
        study = gprMax.GPRStudy(
            [
                gprMax.StudyCase(
                    f"case{index}",
                    [gprMax.ObjectState(sources[family], scale=int(family in active)) for family in families],
                )
                for index, active in enumerate(schedules, start=1)
            ]
        )
        options = {"cpu_precision": precision} if device is None else {"gpu": [device], "gpu_precision": precision}
        gprMax.run(
            scenes=[scene],
            study=study,
            outputfile=tmp_path / f"{backend}_study",
            hide_progress_bars=True,
            log_level=30,
            **options,
        )
        for index, active in enumerate(schedules, start=1):
            reused_path = tmp_path / f"{backend}_study{index}"
            fresh = _run(tmp_path / f"{backend}_fresh{index}", families, active, precision, gpu_device=device)
            _assert_bitwise(_read(reused_path), fresh)
            with h5py.File(str(reused_path) + ".h5", "r") as output:
                assert bool(output["study"].attrs["GeometryReused"]) == (index > 1)
    for index in range(1, len(schedules) + 1):
        _assert_parity(_read(tmp_path / f"cpu_study{index}"), _read(tmp_path / f"cuda_study{index}"), precision)
    assert checked_source_modules == list(families) * (2 * len(schedules))
