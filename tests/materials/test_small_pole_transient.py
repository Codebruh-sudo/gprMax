"""Precision regression for an intentionally extreme, accepted Debye pole.

This is a consistency test against double-precision FDTD, not an analytical
material validation. Previously, single precision effectively removed the
pole and differed by about 57% in this short transient.
"""

import h5py
import numpy as np
import pytest

import gprMax


def _scene():
    scene = gprMax.Scene()
    for obj in (
        gprMax.Discretisation(p1=(0.001,) * 3),
        gprMax.Domain(p1=(0.04, 0.03, 0.03)),
        gprMax.PMLThickness(thickness=5),
        gprMax.TimeWindow(iterations=1000),
        gprMax.Waveform(wave_type="ricker", amp=1, freq=2e9, id="pulse"),
        gprMax.Material(er=2, se=0, mr=1, sm=0, id="slow_pole"),
        gprMax.AddDebyeDispersion(poles=1, er_delta=[1e6], tau=[1e-4], material_ids=["slow_pole"]),
        gprMax.Box(p1=(0, 0, 0), p2=(0.04, 0.03, 0.03), material_id="slow_pole", averaging="n"),
        gprMax.HertzianDipole(p1=(0.012, 0.015, 0.015), polarisation="z", waveform_id="pulse"),
        gprMax.Rx(p1=(0.024, 0.015, 0.015)),
    ):
        scene.add(obj)
    return scene


def _run(path, **options):
    gprMax.run(scenes=[_scene()], outputfile=path, hide_progress_bars=True, log_level=30, **options)
    with h5py.File(path.with_suffix(".h5")) as output:
        return output["rxs/rx1/Ez"][...]


@pytest.mark.integration
@pytest.mark.parametrize(
    "backend",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.gpu),
        pytest.param("opencl", marks=pytest.mark.gpu),
    ],
)
def test_small_pole_transient_retains_coupling(tmp_path, request, backend):
    options = {"cpu_precision": "single"}
    if backend != "cpu":
        device = request.getfixturevalue("gpu_device" if backend == "cuda" else "opencl_device")
        options = {"gpu" if backend == "cuda" else "opencl": [device], "gpu_precision": "single"}
    reference = _run(tmp_path / "double", cpu_precision="double")
    actual = _run(tmp_path / "single", **options)
    assert np.linalg.norm(reference) > 0
    assert np.all(np.isfinite(actual))
    assert np.linalg.norm(actual - reference) / np.linalg.norm(reference) < 3e-5
    assert np.max(np.abs(actual - reference)) / np.max(np.abs(reference)) < 3e-5
