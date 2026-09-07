"""Compare native impedance and antenna/SAR power on real FDTD feed histories.

These are consistency checks between output paths, not an analytical validation
of the fields. The independent Debye shunt check is in test_port_power.py.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import gprMax
from gprMax.ports import evaluate_port_power_spectrum


@pytest.mark.integration
@pytest.mark.parametrize(
    "backend",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.gpu),
        pytest.param("opencl", marks=pytest.mark.gpu),
    ],
)
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("medium", ["air", "lossy", "debye", "lorentz", "drude", "mixed"])
def test_feed_power_matches_native_impedance(tmp_path, request, backend, precision, medium):
    scene = gprMax.Scene()
    for obj in (
        gprMax.Domain(p1=(0.030,) * 3),
        gprMax.Discretisation(p1=(0.001,) * 3),
        gprMax.TimeWindow(time=6e-9),
        gprMax.PMLThickness(thickness=6),
        gprMax.OMPThreads(n=1),
        gprMax.Waveform(wave_type="ricker", amp=1, freq=3e9, id="pulse"),
    ):
        scene.add(obj)
    if medium != "air":
        scene.add(gprMax.Material(er=3.5, se=0.2, mr=1, sm=0, id="feed_material"))
        scene.add(
            gprMax.Box(
                p1=(0.013,) * 3,
                p2=(0.017,) * 3,
                material_id="feed_material",
                averaging=False,
            )
        )
    if medium in ("debye", "mixed"):
        scene.add(
            gprMax.AddDebyeDispersion(
                poles=1, er_delta=(5,), tau=(80e-12,), material_ids=("feed_material",)
            )
        )
    if medium in ("lorentz", "mixed"):
        scene.add(
            gprMax.AddLorentzDispersion(
                poles=1,
                er_delta=(2,),
                omega=(3e9,),
                delta=(2e9,),
                material_ids=("feed_material",),
            )
        )
    if medium in ("drude", "mixed"):
        scene.add(
            gprMax.AddDrudeDispersion(
                poles=1, omega=(1e9,), alpha=(2e9,), material_ids=("feed_material",)
            )
        )
    source = gprMax.VoltageSource(
        p1=(0.015,) * 3,
        polarisation="z",
        resistance=50,
        waveform_id="pulse",
        id="feed",
        spectrum_limit="nyquist",
    )
    scene.add(source)
    if backend == "cpu":
        options = {"cpu_precision": precision}
    else:
        device = request.getfixturevalue("gpu_device" if backend == "cuda" else "opencl_device")
        options = {"gpu" if backend == "cuda" else "opencl": [device], "gpu_precision": precision}
    gprMax.run(
        scenes=[scene],
        outputfile=tmp_path / "feed",
        hide_progress_bars=True,
        log_level=40,
        **options,
    )
    monitor = source._monitor
    native = monitor.result
    assert monitor.background_is_dispersive == (medium not in ("air", "lossy"))
    indices = np.flatnonzero(
        (native.frequency >= 0.5e9) & (native.frequency <= 6e9) & native.valid_zin
    )
    assert indices.size >= 20
    # In Nyquist research mode this adapter needs only dt; all field histories
    # and constitutive properties below come from the completed real solve.
    spectrum = evaluate_port_power_spectrum(
        monitor, SimpleNamespace(dt=monitor.dt), native.frequency[indices]
    )
    voltage = native.total_spectrum[indices]
    current = voltage / native.zin[indices]
    power = 0.5 * np.real(voltage * np.conj(current))
    assert spectrum.terminal_valid.all()
    assert np.isfinite(power).all() and np.max(power) > 0
    # Single precision also rounds the native FFT frequency axis before the
    # arbitrary-frequency DFT. Scale absolute error by the same quantity.
    tolerance = 5e-5 if precision == "single" else 2e-11
    # The bare air gap is almost purely reactive. A relative error divided by
    # its near-zero real power magnifies harmless roundoff; bound the absolute
    # error using apparent power (same units) as well as the relative error.
    np.testing.assert_allclose(
        spectrum.accepted_power,
        power,
        rtol=tolerance,
        atol=tolerance * float(np.max(0.5 * np.abs(voltage) * np.abs(current))),
    )
    # Near an open circuit, Z0*(1+S11)/(1-S11) amplifies single-precision
    # roundoff. Compare the bounded reflection coefficient as well as the
    # terminal current inferred from native Zin, rather than dividing by a
    # nearly zero current again.
    terminal_s11 = (
        spectrum.terminal_voltage - monitor.reference_impedance * spectrum.terminal_current
    ) / (spectrum.terminal_voltage + monitor.reference_impedance * spectrum.terminal_current)
    for actual, expected in (
        (spectrum.terminal_voltage, voltage),
        (spectrum.terminal_current, current),
        (terminal_s11, native.s11[indices]),
    ):
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=tolerance,
            atol=tolerance * float(np.max(np.abs(expected))),
        )
