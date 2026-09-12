"""Automatic SIBC timestep selection happens before time-dependent builds."""

import h5py
import numpy as np
import pytest

import gprMax
from gprMax.model import Model


@pytest.fixture
def run_model(tmp_path, monkeypatch):
    captured = []
    original = Model.build

    def capture(model):
        original(model)
        captured.append((model.G, model.dt, model.dt_mod, model.iterations))

    monkeypatch.setattr(Model, "build", capture)

    def run(scene=None, *, label="model", **kwargs):
        captured.clear()
        args = {"scenes": [scene]} if scene is not None else {}
        gprMax.run(
            **args,
            outputfile=tmp_path / label,
            hide_progress_bars=True,
            cpu_precision="double",
            **kwargs
        )
        return tuple(captured)

    return run


def scene(*, factor=None, sibc=True, copper=False, time=None):
    model = gprMax.Scene()
    for obj in (
        gprMax.Domain(p1=(0.005,) * 3),
        gprMax.Discretisation(p1=(0.001,) * 3),
        gprMax.TimeWindow(iterations=4) if time is None else gprMax.TimeWindow(time=time),
        gprMax.PMLThickness(thickness=0),
        gprMax.OMPThreads(1),
    ):
        model.add(obj)
    if factor is not None:
        model.add(gprMax.TimeStepStabilityFactor(f=factor))
    if time is not None:
        model.add(gprMax.Rx(p1=(0.001, 0.002, 0.002)))
    if sibc:
        model.add(
            gprMax.SurfaceImpedance(
                id="wall", preset="copper", fit_frequency_range=(1e9, 20e9), fit_order=4
            )
            if copper
            else gprMax.SurfaceImpedance(id="wall", resistance=50)
        )
        model.add(gprMax.Box(p1=(0.002,) * 3, p2=(0.003,) * 3, material_id="wall"))
    return model


@pytest.mark.integration
@pytest.mark.parametrize(
    "requested,effective", [(None, 0.99), (1.0, 0.99), (0.9999, 0.99), (0.99, 0.99), (0.8, 0.8)]
)
def test_cap_preserves_stricter_factor_and_reports_automatic_adjustment(
    run_model,
    capsys,
    requested,
    effective,
):
    _, base_dt, _, _ = run_model(scene(sibc=False), label="baseline")[0]
    capsys.readouterr()
    model = scene(factor=requested)
    commands = tuple(model.single_use_objects)
    _, dt, factor, _ = run_model(model)[0]
    output = capsys.readouterr().out
    assert dt == effective * base_dt
    assert factor == effective
    changed = requested is None or requested > 0.99
    assert ("automatically applying #time_step_stability_factor: 0.99" in output) == changed
    assert tuple(model.single_use_objects) == commands
    if requested is not None:
        command = next(obj for obj in commands if isinstance(obj, gprMax.TimeStepStabilityFactor))
        assert command.stability_factor == requested
        assert command.kwargs["f"] == requested


@pytest.mark.integration
def test_non_sibc_models_keep_existing_timestep_behavior(run_model, capsys):
    _, base_dt, factor, _ = run_model(scene(sibc=False), label="baseline")[0]
    assert factor == 1
    _, dt, factor, _ = run_model(scene(sibc=False, factor=0.9999))[0]
    assert factor == 0.9999
    assert dt == 0.9999 * base_dt
    assert "automatically applying" not in capsys.readouterr().out


@pytest.mark.integration
@pytest.mark.parametrize("requested", [0.0, -1.0, 1.01, np.inf, np.nan])
def test_automatic_cap_does_not_hide_invalid_user_factor(run_model, requested):
    with pytest.raises(ValueError, match="finite time step stability factor"):
        run_model(scene(factor=requested))


@pytest.mark.integration
def test_duplicate_user_factors_are_still_rejected(run_model):
    model = scene(factor=0.8)
    model.add(gprMax.TimeStepStabilityFactor(f=0.7))
    with pytest.raises(ValueError):
        run_model(model)


@pytest.mark.integration
def test_time_window_ade_coefficients_and_output_use_effective_dt(run_model, tmp_path):
    duration = 1e-10
    automatic = run_model(scene(time=duration, copper=True), label="auto")[0]
    explicit = run_model(scene(time=duration, copper=True, factor=0.99), label="explicit")[0]
    grid, dt, factor, iterations = automatic
    assert factor == 0.99
    assert iterations == int(np.ceil(duration / dt)) + 1
    assert automatic[1:] == explicit[1:]
    np.testing.assert_array_equal(grid.updatecoeffsE, explicit[0].updatecoeffsE)
    np.testing.assert_array_equal(grid.updatecoeffsH, explicit[0].updatecoeffsH)
    for name in ("edge_runtime", "model_f", "model_q", "model_Z0"):
        np.testing.assert_array_equal(
            getattr(grid.impedance_surfaces, name), getattr(explicit[0].impedance_surfaces, name)
        )
    with h5py.File(tmp_path / "auto.h5") as data:
        assert data.attrs["dt"] == dt
        assert data.attrs["Iterations"] == iterations


@pytest.mark.integration
def test_hash_input_uses_same_automatic_cap(run_model, tmp_path, capsys):
    inputfile = tmp_path / "input.in"
    inputfile.write_text(
        "#domain: 0.005 0.005 0.005\n#dx_dy_dz: 0.001 0.001 0.001\n"
        "#time_window: 4\n#pml_cells: 0\n#num_threads: 1\n"
        "#time_step_stability_factor: 1\n#surface_impedance: wall resistance 50\n"
        "#box: 0.002 0.002 0.002 0.003 0.003 0.003 wall\n",
        encoding="utf-8",
    )
    _, hash_dt, factor, _ = run_model(inputfile=inputfile, label="hash")[0]
    assert factor == 0.99
    assert "automatically applying #time_step_stability_factor: 0.99" in capsys.readouterr().out
    assert run_model(scene())[0][1] == hash_dt


@pytest.mark.integration
def test_geometry_reuse_and_scene_rebuild_do_not_compound_factor(run_model, capsys):
    model = scene()
    first, second = run_model(model, n=2, geometry_fixed=True)
    assert first[1:] == second[1:]
    assert first[2] == 0.99
    assert capsys.readouterr().out.count("automatically applying #time_step_stability_factor") == 1
    assert run_model(model, label="rebuilt")[0][1:] == first[1:]


@pytest.mark.integration
def test_declaration_applies_cap_before_geometry_usage_is_known(run_model):
    model = scene()
    model.geometry_objects.clear()
    grid, _, factor, _ = run_model(model)[0]
    assert factor == 0.99
    assert grid.impedance_surfaces is None
