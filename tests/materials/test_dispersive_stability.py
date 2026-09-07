"""Coupled material restrictions, independent limiting cases and setup errors."""

from types import SimpleNamespace

import numpy as np
import pytest

import gprMax.config as config
from gprMax.dispersive_stability import (
    _conductance_deficit,
    check_dispersive_timestep,
    validate_grid_dispersive_timestep,
)
from gprMax.materials import DispersiveMaterial, Material, create_electric_average_material

pytestmark = pytest.mark.unit


def lorentz(strength, omega, damping=0):
    beta = np.sqrt(omega**2 - damping**2)
    return -1j * strength * omega**2 / beta, -damping + 1j * beta


@pytest.mark.parametrize("poles", (1, 2, 4, 8))
def test_lossless_certificate_matches_closed_interlacing_bound(poles):
    rng = np.random.default_rng(110 + poles)
    for _ in range(80):
        er = rng.uniform(1, 5)
        strength = 10 ** rng.uniform(-2, 2, poles)
        omega = rng.uniform(0.05, np.pi - 0.05, poles)
        spatial = rng.uniform(0, 3)
        expected = er - np.sum(strength * (1 / np.cos(omega / 2) - 1))
        result = check_dispersive_timestep(
            er,
            0,
            [lorentz(d, w) for d, w in zip(strength, omega)],
            1,
            spatial,
            e0=1,
        )
        assert result.nyquist_permittivity == pytest.approx(expected, rel=2e-12, abs=2e-12)
        assert result.passed == (expected > spatial)


def test_poles_must_be_checked_together():
    pole = lorentz(4, 0.5)
    assert check_dispersive_timestep(1, 0, [pole], 1, 0.8, e0=1).passed
    result = check_dispersive_timestep(1, 0, [pole, pole], 1, 0.8, e0=1)
    assert not result.passed
    assert "CFL" in result.reason


@pytest.mark.parametrize("dt", (0.01, 0.1, 1, 10, 100))
def test_debye_and_drude_include_the_complete_conductivity(dt):
    debye = check_dispersive_timestep(2, 0, [(3, -1)], dt, 0, e0=1)
    assert debye.passed
    assert debye.nyquist_permittivity == pytest.approx(2 + 3 * (1 - 1 / np.cosh(dt / 2)))
    # omega_p=1, alpha=2: the negative exponential needs sigma_D=1/2.
    drude = check_dispersive_timestep(2, 0.5, [(-0.5, -2)], dt, 0, e0=1)
    assert drude.passed
    assert not check_dispersive_timestep(2, 0, [(-0.5, -2)], dt, 0, e0=1).passed


def test_positive_nyquist_margin_does_not_replace_passivity():
    result = check_dispersive_timestep(1, 0, [lorentz(100, 5, 0.01)], 1, 0, e0=1)
    assert result.nyquist_permittivity > result.spatial_bound
    assert not result.passed
    assert "passivity" in result.reason


@pytest.mark.parametrize("terms", ([(1, 0)], [(1, 0.1)], [(float("inf"), -1)]))
def test_invalid_representation_is_not_certified(terms):
    assert not check_dispersive_timestep(2, 0, terms, 1, 0, e0=1).passed


def test_nyquist_pole_and_non_positive_denominator_are_rejected():
    assert (
        "Nyquist singularity"
        in check_dispersive_timestep(
            2,
            0,
            [lorentz(1, np.pi)],
            1,
            0,
            e0=1,
        ).reason
    )
    assert "denominator" in check_dispersive_timestep(1, 0, [(-10, -1)], 1, 0, e0=1).reason


def test_slow_poles_do_not_lose_the_nyquist_correction():
    result = check_dispersive_timestep(2, 0, [(1, -1e-10)], 1, 0, e0=1)
    # D=1e10, so D*(1-sech(5e-11)) tends to 1.25e-11.
    assert result.nyquist_permittivity - 2 == pytest.approx(1.25e-11, abs=3e-16)


def test_complex_pair_deficit_covers_its_negative_real_part():
    rng = np.random.default_rng(91)
    for _ in range(80):
        q = -(10 ** rng.uniform(-3, 1)) + 1j * 10 ** rng.uniform(-2, 1)
        c = rng.uniform(-2, 2) + 1j * rng.uniform(-2, 2)
        deficit = _conductance_deficit(q, c)
        s = 1j * np.r_[0, np.logspace(-5, 5, 2000)]
        response = 0.5 * s * (c / (s - q) + c.conjugate() / (s - q.conjugate()))
        assert np.min(response.real) + deficit >= -1e-8 * max(1, deficit)


def _grid(monkeypatch, *, dt=1e-12, er=1, strength=100, frequency=1e11, mode="3D"):
    config.get_model_config().mode = mode
    m = DispersiveMaterial(0, "strong_pole")
    m.er, m.type, m.poles = er, "lorentz", 1
    m.deltaer, m.tau, m.alpha = [strength], [frequency], [0]
    # Vacuum CFL is 1 ps; the pole's zero-curl restriction already fails.
    dl = config.c * 1e-12 * np.sqrt(3)
    return SimpleNamespace(dt=dt, dx=dl, dy=dl, dz=dl, materials=[m], name="test_grid")


def test_failure_identifies_material_and_preserves_inputs(monkeypatch):
    grid = _grid(monkeypatch)
    material = grid.materials[0]
    before = (grid.dt, material.er, material.se, tuple(material.deltaer), tuple(material.tau))
    with pytest.raises(ValueError, match="strong_pole") as caught:
        validate_grid_dispersive_timestep(grid)
    text = str(caught.value)
    assert "TimeStepStabilityFactor" in text
    assert "checked smaller candidate" in text
    assert "not changed" in text
    assert (
        grid.dt,
        material.er,
        material.se,
        tuple(material.deltaer),
        tuple(material.tau),
    ) == before
    grid.dt *= 0.25
    validate_grid_dispersive_timestep(grid)


def test_all_materials_are_checked_after_an_accepted_first_material(monkeypatch):
    grid = _grid(monkeypatch)
    stable = Material(1, "background")
    stable.er = 4
    grid.materials.insert(0, stable)
    with pytest.raises(ValueError, match="strong_pole"):
        validate_grid_dispersive_timestep(grid)


def test_no_dispersion_needs_no_extra_grid_or_config_state():
    validate_grid_dispersive_timestep(SimpleNamespace(materials=[Material(0, "free_space")]))


@pytest.mark.parametrize("mode", ("2D TMx", "2D TMy", "2D TMz", "2D TEx", "2D TEy", "2D TEz"))
def test_invariant_direction_does_not_add_a_spatial_derivative(monkeypatch, mode):
    grid = _grid(monkeypatch, er=2, strength=0.1, frequency=1e9, mode=mode)
    setattr(grid, "d" + mode[-1], 1e-20)
    validate_grid_dispersive_timestep(grid)


def test_common_magnetic_bound_not_own_material_permeability(monkeypatch):
    grid = _grid(monkeypatch, er=2, strength=0.1, frequency=1e9)
    other = Material(1, "low_mu")
    other.mr = 0.1
    grid.materials.append(other)
    with pytest.raises(ValueError, match="spatial bound"):
        validate_grid_dispersive_timestep(grid)


def test_exact_averaged_material_is_checked_as_its_final_mixed_poles(monkeypatch):
    grid = _grid(monkeypatch, er=2, strength=0.1, frequency=1e9)
    other = DispersiveMaterial(1, "debye")
    other.er, other.type, other.poles = 3, "debye", 1
    other.deltaer, other.tau = [2], [1e-10]
    averaged = create_electric_average_material(
        2, "mixture", [grid.materials[0], grid.materials[0], other, other]
    )
    grid.materials = [averaged]
    validate_grid_dispersive_timestep(grid)


def test_rank_without_poles_receives_other_rank_failure(monkeypatch):
    config.get_model_config().mode = "3D"

    class Comm:
        def __init__(self):
            self.calls = 0

        def allgather(self, value):
            self.calls += 1
            if self.calls == 1:
                return [value, (True, 1.0, None)]
            return [value, "Dispersive timestep check failed on the other rank"]

    grid = SimpleNamespace(
        dt=1e-12,
        dx=0.001,
        dy=0.001,
        dz=0.001,
        materials=[Material(0, "background")],
        comm=Comm(),
    )
    with pytest.raises(ValueError, match="other rank"):
        validate_grid_dispersive_timestep(grid)
    assert grid.comm.calls == 2


def test_active_pole_has_no_false_smaller_timestep_hint(monkeypatch):
    grid = _grid(monkeypatch)
    m = grid.materials[0]
    m.inclusive_w, m.inclusive_q = [1], [1]
    with pytest.raises(ValueError, match="No smaller timestep was certified"):
        validate_grid_dispersive_timestep(grid)
