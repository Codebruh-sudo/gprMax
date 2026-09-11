"""Sample counts, interpolation boundaries and source-specific time lattices."""

import numpy as np
import pytest
from scipy.interpolate import interp1d

from gprMax.sources import VoltageSource
from gprMax.user_objects.cmds_multiuse import ExcitationFile, Waveform

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("iterations", [5, 7, 16, 125, 189, 200, 257])
@pytest.mark.parametrize("dl", [0.001, 0.002, 0.005])
def test_implicit_time_axis_has_exactly_iterations_samples(fake_grid, iterations, dl):
    dt = dl / (299792458.0 * np.sqrt(3))
    grid = fake_grid(iterations=iterations, dt=dt, timewindow=(iterations - 1) * dt)
    values = np.arange(iterations, dtype=float)
    Waveform(wave_type="user", user_values=values, id="sampled").build(grid)
    np.testing.assert_array_equal(grid.waveforms[0].userfunc.x, np.arange(iterations) * dt)
    np.testing.assert_array_equal(grid.waveforms[0].userfunc(np.arange(iterations) * dt), values)


@pytest.mark.parametrize("path", ["api", "file", "file_with_time"])
@pytest.mark.parametrize("fill", [None, 0, 3.5, "extrapolate"])
def test_sampled_boundaries_use_default_zero_or_explicit_fill(tmp_path, fake_grid, path, fill):
    grid = fake_grid(iterations=3, dt=1.0, timewindow=2.0)
    kwargs = {} if fill is None else dict(kind="linear", fill_value=fill)
    if path == "api":
        Waveform(wave_type="user", user_values=[1, 2, 3], user_time=[0, 1, 2], id="sampled", **kwargs).build(grid)
    else:
        filename = tmp_path / "samples.txt"
        filename.write_text("time sampled\n0 1\n1 2\n2 3\n" if path == "file_with_time" else "sampled\n1\n2\n3\n")
        ExcitationFile(filename, **kwargs).build(grid)
    function = grid.waveforms[0].userfunc
    np.testing.assert_array_equal(function([0, 0.5, 2]), [1, 1.5, 3])
    expected = [0.5, 3.5] if fill == "extrapolate" else [0 if fill is None else fill] * 2
    np.testing.assert_array_equal(function([-0.5, 2.5]), expected)


def test_api_explicit_none_interpolation_options_use_defaults(fake_grid):
    grid = fake_grid(iterations=3, dt=1.0, timewindow=2.0)
    Waveform(wave_type="user", user_values=[1, 2, 3], id="sampled", kind=None, fill_value=None).build(grid)
    np.testing.assert_array_equal(grid.waveforms[0].userfunc([-0.5, 0.5, 2.5]), [0, 1.5, 0])


def test_hard_source_never_evaluates_unused_half_steps(fake_grid):
    grid = fake_grid(iterations=16, dt=1.0, timewindow=15.0)
    Waveform(wave_type="user", user_values=np.ones(16), id="sampled").build(grid)
    # Strict interpolation proves that zero-extension is not hiding an
    # unnecessary hard-source lookup past the final whole-step sample.
    grid.waveforms[0].userfunc = interp1d(np.arange(16), np.ones(16), bounds_error=True)
    source = VoltageSource()
    source.resistance, source.waveformID, source.stop = 0, "sampled", grid.timewindow
    source.calculate_waveform_values(grid)
    assert source.waveformvalues_halfdt is None
    np.testing.assert_array_equal(source.waveformvalues_wholedt, [1] * 16 + [0])


@pytest.mark.parametrize("resistances", [(0, 50, 0, 75), (50, 0, 75, 0)])
def test_hard_and_resistive_cache_donors_do_not_cross_lattices(fake_grid, resistances):
    grid = fake_grid(iterations=4, dt=1.0, timewindow=3.0)
    Waveform(wave_type="user", user_func=lambda time: 1 + time, id="ramp").build(grid)
    for resistance in resistances:
        source = VoltageSource()
        source.resistance, source.waveformID, source.stop = resistance, "ramp", grid.timewindow
        source.calculate_waveform_values(grid)
        grid.voltagesources.append(source)
        values = source.waveformvalues_wholedt if resistance == 0 else source.waveformvalues_halfdt
        np.testing.assert_array_equal(values[:4], np.arange(4) + (1 if resistance == 0 else 1.5))
    for first, second in ((0, 2), (1, 3)):
        name = "waveformvalues_wholedt" if resistances[first] == 0 else "waveformvalues_halfdt"
        assert getattr(grid.voltagesources[first], name) is getattr(grid.voltagesources[second], name)
