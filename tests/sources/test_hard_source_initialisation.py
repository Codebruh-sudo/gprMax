"""Physical-time hard assignments, independent of waveform amplitude."""

from types import SimpleNamespace

import numpy as np
import pytest

from gprMax.sources import VoltageSource, initialise_hard_source_fields


def source(axis="z", *, start=0.0, stop=10.0, resistance=0.0, values=None):
    result = VoltageSource()
    result.polarisation = axis
    result.coord[:] = (1, 1, 1)
    result.start, result.stop, result.resistance = start, stop, resistance
    result.waveformvalues_wholedt = np.arange(6.0) + 1 if values is None else values
    return result


def grid(sources, dtype=np.float64):
    return SimpleNamespace(
        voltagesources=sources,
        dt=0.1,
        dx=0.5,
        dy=0.25,
        dz=0.125,
        Ex=np.full((3, 3, 3), 9, dtype=dtype),
        Ey=np.full((3, 3, 3), 9, dtype=dtype),
        Ez=np.full((3, 3, 3), 9, dtype=dtype),
    )


@pytest.mark.parametrize("axis", "xyz")
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_initial_field_then_next_sample_and_final_lookahead(axis, dtype):
    src = source(axis)
    g = grid([src], dtype)
    values = src.waveformvalues_wholedt.copy()
    field, dl = getattr(g, "E" + axis), getattr(g, "d" + axis)
    assert initialise_hard_source_fields(g)
    assert field[1, 1, 1] == -values[0] / dl
    for iteration in range(5):
        # The hard branch must not read coefficients/IDs or additive samples.
        src.update_electric(iteration, None, None, g.Ex, g.Ey, g.Ez, g)
        assert field[1, 1, 1] == -values[iteration + 1] / dl
    np.testing.assert_array_equal(src.waveformvalues_wholedt, values)


def test_initialisation_skips_other_source_types_and_delayed_hard_sources():
    delayed = source(start=0.05)
    soft = source(resistance=50)
    soft.waveformvalues_wholedt = None
    g = grid([soft, delayed])
    assert not initialise_hard_source_fields(g)
    for axis in "xyz":
        np.testing.assert_array_equal(getattr(g, "E" + axis), 9)


def test_zero_amplitude_clamps_and_coincident_initialisation_keeps_list_order():
    first = source(values=np.ones(6))
    last = source(values=np.zeros(6))
    g = grid([first, last])
    assert initialise_hard_source_fields(g)
    assert g.Ez[1, 1, 1] == 0
    g.voltagesources.reverse()
    assert initialise_hard_source_fields(g)
    assert g.Ez[1, 1, 1] == -1 / g.dz


@pytest.mark.parametrize("bound", [0.1, np.nextafter(0.1, 0), np.nextafter(0.1, np.inf)])
def test_inclusive_activity_uses_applied_electric_time(bound):
    for start, stop in ((0, bound), (bound, 0.5)):
        src = source(start=start, stop=stop, values=np.zeros(6))
        g = grid([src])
        initialise_hard_source_fields(g)
        for iteration in range(5):
            g.Ez.fill(9)
            src.update_electric(iteration, None, None, g.Ex, g.Ey, g.Ez, g)
            active = start <= (iteration + 1) * g.dt <= stop
            assert g.Ez[1, 1, 1] == (0 if active else 9)


def test_initialisation_repeats_after_reset_with_current_waveform_and_position():
    src = source()
    g = grid([src])
    initialise_hard_source_fields(g)
    g.Ez.fill(0)
    src.coord[:] = (2, 1, 1)
    src.waveformvalues_wholedt = np.full(6, 3.0)
    initialise_hard_source_fields(g)
    assert g.Ez[1, 1, 1] == 0
    assert g.Ez[2, 1, 1] == -3 / g.dz
