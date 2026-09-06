"""OpenCL PML launch ranges remain correct when geometry is reused."""

import h5py
import numpy as np
import pytest

import gprMax
from gprMax.pml import OpenCLPML

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def _scene():
    scene = gprMax.Scene()
    for obj in (
        gprMax.Domain(p1=(0.020, 0.018, 0.016)),
        gprMax.Discretisation(p1=(0.001,) * 3),
        gprMax.PMLThickness(thickness=3),
        gprMax.OMPThreads(1),
        gprMax.TimeWindow(iterations=100),
        gprMax.Waveform(wave_type="gaussian", amp=1, freq=2e10, id="pulse"),
        gprMax.HertzianDipole(p1=(0.010, 0.009, 0.008), polarisation="z", waveform_id="pulse"),
        gprMax.Rx(p1=(0.005, 0.005, 0.005)),
    ):
        scene.add(obj)
    return scene


@pytest.mark.parametrize("precision", ("single", "double"))
def test_opencl_pml_ranges_survive_geometry_fixed(tmp_path, monkeypatch, opencl_device, precision):
    uploads = []
    original = OpenCLPML.htod_field_arrays

    def record_upload(self):
        original(self)
        uploads.append((self, self._electric_update_range, self._magnetic_update_range))

    monkeypatch.setattr(OpenCLPML, "htod_field_arrays", record_upload)
    paths = {}
    for reuse in (False, True):
        uploads.clear()
        output = tmp_path / ("reused" if reuse else "fresh")
        gprMax.run(
            scenes=[_scene()] if reuse else [_scene(), _scene()],
            n=2,
            geometry_fixed=reuse,
            outputfile=output,
            opencl=[opencl_device],
            gpu_precision=precision,
            hide_progress_bars=True,
            log_level=40,
        )
        assert len(uploads) == 12  # Six slabs uploaded for each solve.
        for first, second in zip(uploads[:6], uploads[6:]):
            assert (first[0] is second[0]) == reuse
            assert first[1:] == second[1:]
            assert first[1].stop != first[2].stop  # E/H pitches are distinct.
        paths[reuse] = output

    for model in (1, 2):
        with h5py.File(f"{paths[False]}{model}.h5") as fresh, h5py.File(f"{paths[True]}{model}.h5") as reused:
            for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
                expected = fresh[f"rxs/rx1/{component}"][...]
                actual = reused[f"rxs/rx1/{component}"][...]
                assert np.isfinite(actual).all()
                np.testing.assert_array_equal(actual, expected)
                assert actual.dtype == expected.dtype
                assert actual.tobytes() == expected.tobytes()
            assert np.max(np.abs(fresh["rxs/rx1/Ez"][...])) > 0
