"""Source-band estimation and the explicit mesh-resolution override."""

import argparse
import sys

import pytest

from gprMax import config
from gprMax import gprMax as application
from gprMax.waveforms import Waveform

pytestmark = pytest.mark.unit


def _grid(make_grid, make_material, waveforms, mr=1):
    grid = make_grid(arrays=False)
    grid.dt = 1e-12
    grid.materials = [make_material(ID="medium", mr=mr)]
    grid.waveforms = waveforms
    return grid


@pytest.mark.parametrize("kind", Waveform.types)
def test_zero_amplitude_waveform_is_not_evaluated(
    make_grid, make_material, make_waveform, monkeypatch, kind
):
    waveform = make_waveform(kind, freq=2e11, amp=0)

    def forbidden(*args):
        pytest.fail("A zero-amplitude waveform must be excluded before sampling")

    monkeypatch.setattr(waveform, "calculate_coefficients", forbidden)
    monkeypatch.setattr(waveform, "calculate_value", forbidden)
    grid = _grid(make_grid, make_material, [waveform])
    result = grid._dispersion_analysis(16000)
    assert result["N"] is None
    assert result["maxfreq"] == []
    assert result["error"] == "no non-zero-amplitude waveform detected."
    grid.dispersion_analysis(16000)
    assert grid.waveforms == [waveform]  # Diagnostic exclusion is not source removal.


@pytest.mark.parametrize("kind", ("ricker", "sine", "impulse", "user"))
def test_passive_waveform_does_not_change_active_band(
    make_grid, make_material, make_waveform, kind
):
    active = make_waveform("ricker", freq=1e9)
    passive = make_waveform(kind, freq=2e11, amp=0)
    grid = _grid(make_grid, make_material, [active])
    reference = grid._dispersion_analysis(16000)
    for waveforms in ([active, passive], [passive, active]):
        grid.waveforms = waveforms
        actual = grid._dispersion_analysis(16000)
        for key in ("N", "maxfreq", "error", "deltavp"):
            assert actual[key] == reference[key]


@pytest.mark.parametrize("amp", (1, -1, 1e-12, -1e3))
def test_nonzero_sign_and_amplitude_do_not_change_band(
    make_grid, make_material, make_waveform, amp
):
    grid = _grid(make_grid, make_material, [make_waveform("gaussian", amp=1)])
    reference = grid._dispersion_analysis(16000)
    grid.waveforms = [make_waveform("gaussian", amp=amp)]
    actual = grid._dispersion_analysis(16000)
    assert actual["error"] == reference["error"] == ""
    assert actual["maxfreq"] == reference["maxfreq"]


def test_waveform_order_cannot_disable_resolution_rejection(
    make_grid, make_material, make_waveform
):
    waveforms = [make_waveform("ricker", freq=2e9), make_waveform("ricker", freq=0.5e9)]
    grid = _grid(make_grid, make_material, waveforms, mr=400)
    results = []
    for ordered in (waveforms, list(reversed(waveforms))):
        grid.waveforms = ordered
        results.append(grid._dispersion_analysis(16000))
        with pytest.raises(ValueError, match="Insufficient spatial resolution"):
            grid.dispersion_analysis(16000)
    assert results[0]["N"] == results[1]["N"] < 3
    assert results[0]["error"] == results[1]["error"] == ""


@pytest.mark.parametrize("iterations,frequency", ((16000, 2e11), (1, 1e9)))
def test_unresolved_waveform_reports_missing_band_instead_of_index_error(
    make_grid, make_material, make_waveform, caplog, iterations, frequency
):
    grid = _grid(make_grid, make_material, [make_waveform("ricker", freq=frequency)])
    with caplog.at_level("WARNING"):
        grid.dispersion_analysis(iterations)
    assert "not carried out" in caplog.text
    assert "timestep" in caplog.text


def test_partial_band_does_not_hide_known_underresolution(
    make_grid, make_material, make_waveform, caplog
):
    grid = _grid(
        make_grid,
        make_material,
        [make_waveform("ricker", freq=2e9), make_waveform("impulse")],
        mr=400,
    )
    with pytest.raises(ValueError, match="Insufficient spatial resolution"):
        grid.dispersion_analysis(16000)
    assert "incomplete source-band" in caplog.text


def test_override_warns_and_preserves_computed_metrics(
    make_grid, make_material, make_waveform, monkeypatch, caplog
):
    grid = _grid(make_grid, make_material, [make_waveform("ricker", freq=2e9)], mr=400)
    reference = grid._dispersion_analysis(16000)
    monkeypatch.setattr(config.sim_config, "allow_underresolved", True, raising=False)
    with caplog.at_level("WARNING"):
        grid.dispersion_analysis(16000)
    assert "Continuing with an under-resolved mesh" in caplog.text
    actual = grid._dispersion_analysis(16000)
    for key in ("N", "maxfreq", "phase_velocity", "deltavp"):
        assert actual[key] == reference[key]


def test_override_does_not_swallow_material_response_errors(make_grid, monkeypatch):
    grid = make_grid(arrays=False)
    monkeypatch.setattr(config.sim_config, "allow_underresolved", True, raising=False)

    def invalid(iterations):
        raise ValueError("singular material response")

    monkeypatch.setattr(grid, "_dispersion_analysis", invalid)
    with pytest.raises(ValueError, match="singular material response"):
        grid.dispersion_analysis(10)


@pytest.mark.parametrize("enabled", (False, True))
def test_override_api_and_cli_use_same_boolean(tmp_path, monkeypatch, enabled):
    monkeypatch.setattr(application, "run_main", lambda args: args)
    assert application.run(allow_underresolved=enabled).allow_underresolved is enabled
    model = tmp_path / "model.in"
    model.touch()
    monkeypatch.setattr(
        sys, "argv", ["gprMax", str(model)] + (["--allow-underresolved"] if enabled else [])
    )
    assert application.cli().allow_underresolved is enabled


@pytest.mark.parametrize("value", (False, True, "False", None, 1))
def test_override_config_requires_explicit_boolean(value):
    args = argparse.Namespace(**application.args_defaults)
    args.inputfile = "model.in"
    args.allow_underresolved = value
    if isinstance(value, bool):
        assert config.SimulationConfig(args).allow_underresolved is value
    else:
        with pytest.raises(ValueError, match="allow_underresolved must be True or False"):
            config.SimulationConfig(args)
