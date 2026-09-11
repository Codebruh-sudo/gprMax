"""Sampled API/hash excitations must reach each backend on the right lattice."""

import decimal

import h5py
import numpy as np
import pytest

import gprMax
from gprMax.utilities.utilities import round_value
from tests.updates.test_hard_source_timing_solve import backend_options, BACKENDS

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("interface", ["api", "hash"])
@pytest.mark.parametrize("family", ["hard", "resistive", "hertzian", "magnetic"])
def test_sampled_source_matches_callable_and_writes_correct_excitation(tmp_path, request, backend, interface, family):
    precision = "single" if backend == "metal" else "double"
    options = backend_options(request, backend, precision)
    dl, count = 0.002, 16
    dt = round_value(1 / (299792458.0 * np.sqrt(3 / dl**2)), decimalplaces=decimal.getcontext().prec - 1)
    times, values = np.arange(count) * dt, 1 + 0.05 * np.arange(count)
    point = (0.012,) * 3

    def scene(callable_control):
        result = gprMax.Scene()
        for obj in (
            gprMax.Domain((0.024,) * 3),
            gprMax.Discretisation((dl,) * 3),
            gprMax.TimeWindow(iterations=count),
            gprMax.PMLThickness(2),
            gprMax.OMPThreads(1),
        ):
            result.add(obj)
        result.add(
            gprMax.Waveform(
                wave_type="user",
                id="sampled",
                **(
                    {"user_func": lambda time: np.interp(time, times, values, left=0, right=0)}
                    if callable_control
                    else {"user_values": values}
                ),
            )
        )
        kwargs = dict(p1=point, polarisation="z", waveform_id="sampled")
        source = (
            gprMax.MagneticDipole(**kwargs)
            if family == "magnetic"
            else gprMax.HertzianDipole(**kwargs)
            if family == "hertzian"
            else gprMax.VoltageSource(**kwargs, resistance=0 if family == "hard" else 50)
        )
        result.add(source)
        result.add(gprMax.Rx(point, outputs=[kind + axis for kind in "EH" for axis in "xyz"]))
        return result

    run_options = dict(log_level=50, hide_progress_bars=True, **options)
    if interface == "api":
        gprMax.run(scenes=[scene(False)], outputfile=tmp_path / "sampled", **run_options)
    else:
        samples = tmp_path / "wave.txt"
        np.savetxt(samples, values, header="sampled", comments="")
        command = {
            "hard": "#voltage_source: z 0.012 0.012 0.012 0 sampled",
            "resistive": "#voltage_source: z 0.012 0.012 0.012 50 sampled",
            "hertzian": "#hertzian_dipole: z 0.012 0.012 0.012 sampled",
            "magnetic": "#magnetic_dipole: z 0.012 0.012 0.012 sampled",
        }[family]
        source = tmp_path / "model.in"
        source.write_text(
            f"#domain: 0.024 0.024 0.024\n#dx_dy_dz: {dl} {dl} {dl}\n"
            f"#time_window: {count}\n#pml_cells: 2\n#omp_threads: 1\n"
            f"#excitation_file: {samples}\n{command}\n#rx: 0.012 0.012 0.012\n"
        )
        gprMax.run(inputfile=source, outputfile=tmp_path / "sampled", **run_options)
    gprMax.run(scenes=[scene(True)], outputfile=tmp_path / "control", **run_options)
    rtol, atol = (3e-5, 1e-4) if precision == "single" else (3e-12, 3e-11)
    with h5py.File(tmp_path / "sampled.h5") as sampled, h5py.File(tmp_path / "control.h5") as control:
        assert sampled.attrs["dt"] == dt
        for component in (kind + axis for kind in "EH" for axis in "xyz"):
            np.testing.assert_allclose(
                sampled[f"rxs/rx1/{component}"][:], control[f"rxs/rx1/{component}"][:], rtol=rtol, atol=atol
            )
        field = sampled["rxs/rx1/Hz" if family == "magnetic" else "rxs/rx1/Ez"][:]
        assert np.isfinite(field).all() and np.count_nonzero(field) > 1
        excitation = sampled["srcs/src1/excitation"]
        expected = np.interp(
            times + (0.5 * dt if family in ("resistive", "hertzian") else 0), times, values, left=0, right=0
        )
        np.testing.assert_allclose(excitation["samples"][:], expected, rtol=rtol, atol=atol)
        if family == "hard":
            np.testing.assert_allclose(field, -values / dl, rtol=rtol, atol=atol)
            assert excitation.attrs["TimeSampleOffset"] == 0
