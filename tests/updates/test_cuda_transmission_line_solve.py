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

"""End-to-end CPU/GPU parity for a device-resident transmission line."""

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose

import gprMax
from gprMax.updates.cuda_updates import CUDAUpdates
from gprMax.updates.opencl_updates import OpenCLUpdates

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

try:
    import pycuda.driver as _cuda_driver

    _cuda_driver.init()
    HAS_CUDA = _cuda_driver.Device.count() > 0
except Exception:
    HAS_CUDA = False

try:
    import pyopencl as _cl

    HAS_OPENCL = bool(_cl.get_platforms())
except Exception:
    HAS_OPENCL = False


def _scene(case="tl_only"):
    dl = 1e-3
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.012, 0.012, 0.012)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-10))
    scene.add(gprMax.Waveform(wave_type="gaussian", amp=1, freq=2e10, id="w"))
    scene.add(
        gprMax.TransmissionLine(
            polarisation="z",
            p1=(0.006, 0.006, 0.006),
            resistance=50,
            waveform_id="w",
        )
    )
    if case == "active_passive":
        scene.add(gprMax.Waveform(wave_type="gaussian", amp=0, freq=2e10, id="passive"))
        scene.add(
            gprMax.TransmissionLine(
                polarisation="z",
                p1=(0.008, 0.006, 0.006),
                resistance=50,
                waveform_id="passive",
            )
        )
    elif case == "overlap_magnetic_dipole":
        # Hx(6, 5, 6) is one of the four edges in this z-directed TL's
        # Ampere loop. Its correction must be visible in the same H step.
        scene.add(gprMax.Waveform(wave_type="gaussian", amp=1e-5, freq=2e10, id="magnetic"))
        scene.add(gprMax.MagneticDipole(polarisation="x", p1=(0.006, 0.005, 0.006), waveform_id="magnetic"))
    scene.add(gprMax.Rx(p1=(0.006, 0.006, 0.006), id="rx"))
    return scene


def _traces(path):
    """Read all TL histories and receiver fields, excluding runtime metadata."""
    result = {}
    with h5py.File(str(path) + ".h5", "r") as output:
        for group_name in ("tls", "rxs"):
            output[group_name].visititems(
                lambda name, obj: result.update({f"{group_name}/{name}": obj[:]})
                if isinstance(obj, h5py.Dataset)
                else None
            )
    return result


@pytest.mark.parametrize("backend", ["cuda", "opencl"])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("case", ["tl_only", "active_passive", "overlap_magnetic_dipole"])
def test_device_transmission_line_matches_cpu(tmp_path, request, monkeypatch, backend, precision, case):
    if backend == "cuda" and not HAS_CUDA:
        pytest.skip("No CUDA device/pycuda available")
    if backend == "opencl" and not HAS_OPENCL:
        pytest.skip("No OpenCL platform/pyopencl available")

    if backend == "cuda":
        device_options = {"gpu": [request.getfixturevalue("gpu_device")]}
    else:
        device_options = {"opencl": [request.getfixturevalue("opencl_device")]}

    cpu_path = tmp_path / f"cpu_tl_{precision}_{case}"
    device_path = tmp_path / f"{backend}_tl_{precision}_{case}"
    gprMax.run(
        scenes=[_scene(case)],
        n=1,
        outputfile=cpu_path,
        hide_progress_bars=True,
        cpu_precision=precision,
    )
    gprMax.run(
        scenes=[_scene(case)],
        n=1,
        outputfile=device_path,
        hide_progress_bars=True,
        gpu_precision=precision,
        **device_options,
    )

    cpu_traces = _traces(cpu_path)
    device_traces = _traces(device_path)
    assert cpu_traces.keys() == device_traces.keys()
    tolerance = 2e-5 if precision == "single" else 2e-12
    # Hz vanishes by symmetry for the TL-only scene. Normalize receiver
    # error by its E/H vector scale, not the roundoff-sized component.
    field_scales = {
        kind: max(float(np.max(np.abs(cpu_traces[f"rxs/rx1/{kind}{axis}"]))) for axis in "xyz") for kind in ("E", "H")
    }
    for name in cpu_traces:
        if name.startswith("rxs/") or name.rsplit("/", 1)[-1] in ("Vinc", "Iinc", "Vtotal", "Itotal"):
            scale = max(float(np.max(np.abs(cpu_traces[name]))), 1e-12)
            if name.startswith("rxs/"):
                scale = max(scale, field_scales[name.rsplit("/", 1)[-1][0]])
            assert np.isfinite(device_traces[name]).all(), name
            assert_allclose(device_traces[name], cpu_traces[name], rtol=tolerance, atol=tolerance * scale, err_msg=name)
    if case == "active_passive":
        assert np.max(np.abs(cpu_traces["tls/tl2/Vtotal"])) > 1e-6
        np.testing.assert_array_equal(cpu_traces["tls/tl2/Vinc"], 0)

    if case != "overlap_magnetic_dipole":
        # No extra magnetic writer touches these models: relocating the
        # exact existing launch must leave their accelerator output bitwise
        # unchanged compared with the pre-relocation schedule.
        updates_cls = CUDAUpdates if backend == "cuda" else OpenCLUpdates
        sample = updates_cls.update_magnetic_edge_devices
        write_sources = updates_cls.update_magnetic_sources

        def legacy_magnetic_sources(self, iteration):
            sample(self, iteration)
            write_sources(self, iteration)

        legacy_path = tmp_path / f"legacy_{backend}_tl_{precision}_{case}"
        with monkeypatch.context() as legacy:
            legacy.setattr(updates_cls, "update_magnetic_sources", legacy_magnetic_sources)
            legacy.setattr(updates_cls, "update_magnetic_edge_devices", lambda self, iteration: None)
            gprMax.run(
                scenes=[_scene(case)],
                n=1,
                outputfile=legacy_path,
                hide_progress_bars=True,
                gpu_precision=precision,
                **device_options,
            )
        legacy_traces = _traces(legacy_path)
        for name in device_traces:
            np.testing.assert_array_equal(device_traces[name], legacy_traces[name], err_msg=name)

    assert np.max(np.abs(cpu_traces["tls/tl1/Vtotal"])) > 1e-3
    spectral_names = ("frequency", "S11", "Zin", "valid_Zin")
    cpu = {name: cpu_traces[f"tls/tl1/{name}"] for name in spectral_names}
    device = {name: device_traces[f"tls/tl1/{name}"] for name in spectral_names}

    assert_allclose(device["frequency"], cpu["frequency"], rtol=0, atol=0)
    valid = cpu["valid_Zin"].astype(bool) & device["valid_Zin"].astype(bool)
    assert valid.any()
    tolerance = 4e-5 if precision == "single" else 4e-12
    assert_allclose(device["S11"][valid], cpu["S11"][valid], rtol=tolerance, atol=tolerance)
    impedance_scale = max(float(np.max(np.abs(cpu["Zin"][valid]))), 1.0)
    # Near an open circuit, Zin = Z0(1 + S11)/(1 - S11) amplifies a small
    # single-precision S11 difference. Keep the tighter comparison on S11
    # above and allow the corresponding conditioning in this secondary value.
    impedance_tolerance = 2e-4 if precision == "single" else 4e-12
    assert_allclose(
        device["Zin"][valid],
        cpu["Zin"][valid],
        rtol=impedance_tolerance,
        atol=impedance_tolerance * impedance_scale,
    )
