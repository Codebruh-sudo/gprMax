"""A source-spectrum zero must not be accepted for SAR normalisation."""

import decimal

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import gprMax
from gprMax.utilities.utilities import round_value

pytestmark = pytest.mark.integration


def _scene(waveform_file):
    dt = 2e-12
    dl = 0.002
    cfl_dt = round_value(
        1 / (299792458.0 * np.sqrt(3 / dl**2)),
        decimalplaces=decimal.getcontext().prec - 1,
    )
    scene = gprMax.Scene()
    for obj in (
        gprMax.Domain(p1=(0.012,) * 3),
        gprMax.Discretisation(p1=(dl,) * 3),
        gprMax.TimeStepStabilityFactor(f=dt / cfl_dt),
        gprMax.TimeWindow(iterations=40000),
        gprMax.PMLThickness(thickness=1),
        gprMax.OMPThreads(n=1),
        gprMax.Material(er=1, se=0.01, mr=1, sm=0, id="lossy"),
        gprMax.MaterialDensity(density=1000, material_ids="lossy"),
        gprMax.Box(p1=(0.004,) * 3, p2=(0.006,) * 3, material_id="lossy", tag="target"),
        gprMax.ExcitationFile(waveform_file, kind="previous", fill_value="extrapolate"),
        gprMax.HertzianDipole(p1=(0.006,) * 3, polarisation="z", waveform_id="pulse"),
    ):
        scene.add(obj)
    for normalisation in ("waveform", "current_moment"):
        scene.add(
            gprMax.SAR(
                frequencies=(5e9, 10e9),
                waveform_id="pulse",
                tags="target",
                id=normalisation,
                normalisation=normalisation,
                source_floor_db=-100,
                spectrum_limit=10,
            )
        )
    return scene


def test_single_and_double_precision_sar_reject_analytical_source_zero(tmp_path):
    waveform_file = tmp_path / "two_impulses.txt"
    samples = np.zeros(40000)
    samples[[0, 32775]] = 1
    np.savetxt(waveform_file, samples, header="pulse", comments="")
    supported_sar = {}
    for precision in ("single", "double"):
        output = tmp_path / precision
        gprMax.run(
            scenes=[_scene(waveform_file)],
            n=1,
            outputfile=output,
            hide_progress_bars=True,
            log_level=40,
            cpu_precision=precision,
        )
        with h5py.File(output.with_suffix(".h5")) as data:
            stored_source = data["srcs/src1/excitation/samples"][...]
            assert_array_equal(stored_source, samples)
            assert float(data["srcs/src1/excitation"].attrs["SampleInterval"]) == 2e-12
            expected_dtype = np.complex64 if precision == "single" else np.complex128
            supported_sar[precision] = {}
            for normalisation in ("waveform", "current_moment"):
                sar = data[f"sar/{normalisation}"]
                assert sar["source_spectrum"].dtype == expected_dtype
                assert_array_equal(sar["mesh_valid"][...], [1, 1])
                assert_array_equal(sar["source_valid"][...], [1, 0])
                assert_array_equal(sar["valid"][...], [1, 0])
                assert sar["source_relative_db"][1] < -100
                assert np.isnan(sar["sar"][1]).all()
                assert np.isnan(sar["absorbed_power_density"][1]).all()
                assert np.isfinite(sar["sar"][0]).all()
                assert np.all(sar["sar"][0] > 0)
                # The 5 GHz source phasor is dt*(1+j); at 10 GHz it is zero.
                assert_allclose(sar["source_spectrum"][0], 2e-12 * (1 + 1j), rtol=2e-7)
                supported_sar[precision][normalisation] = sar["sar"][0]
            # Current-moment normalisation includes the dipole's 2 mm length.
            assert_allclose(
                supported_sar[precision]["current_moment"] * 0.002**2,
                supported_sar[precision]["waveform"],
                rtol=2e-6,
            )
    for normalisation in ("waveform", "current_moment"):
        assert_allclose(
            supported_sar["single"][normalisation],
            supported_sar["double"][normalisation],
            rtol=3e-5,
        )


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("backend", ["cuda", "opencl"])
def test_device_sar_rejects_analytical_source_zero(tmp_path, request, backend):
    options = (
        {"gpu": [request.getfixturevalue("gpu_device")]}
        if backend == "cuda"
        else {"opencl": [request.getfixturevalue("opencl_device")]}
    )
    waveform_file = tmp_path / "two_impulses.txt"
    samples = np.zeros(40000, dtype=np.float32)
    samples[[0, 32775]] = 1
    np.savetxt(waveform_file, samples, header="pulse", comments="")
    output = tmp_path / backend
    gprMax.run(
        scenes=[_scene(waveform_file)],
        n=1,
        outputfile=output,
        hide_progress_bars=True,
        log_level=40,
        gpu_precision="single",
        **options,
    )
    with h5py.File(output.with_suffix(".h5")) as data:
        assert_array_equal(data["srcs/src1/excitation/samples"][...], samples)
        for normalisation in ("waveform", "current_moment"):
            sar = data[f"sar/{normalisation}"]
            assert sar.attrs["CollectionBackend"] == f"{backend}_device"
            assert sar["source_spectrum"].dtype == np.complex64
            assert_array_equal(sar["mesh_valid"][...], [1, 1])
            assert_array_equal(sar["source_valid"][...], [1, 0])
            assert_array_equal(sar["valid"][...], [1, 0])
            assert sar["source_relative_db"][1] < -100
            assert np.isnan(sar["sar"][1]).all()
            assert np.isnan(sar["absorbed_power_density"][1]).all()
            assert_allclose(sar["source_spectrum"][0], 2e-12 * (1 + 1j), rtol=2e-7)
            assert np.isfinite(sar["sar"][0]).all()
            assert np.all(sar["sar"][0] > 0)
