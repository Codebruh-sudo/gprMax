"""Independent limits and sign checks for the analytical validation references."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.constants import c, epsilon_0, mu_0

from testing.validation.impedance_surface.analytical import (
    ETA0, bulk_discrete_permittivity, conductor_mie_coefficients,
    discrete_normal_reflection, impedance_mie_coefficients, mie_amplitudes,
    normal_reflection, sphere_reference,
)
from testing.validation.mie_dielectric import dielectric_mie_coefficients
from testing.validation.mie_pec import pec_mie_coefficients, pec_mie_amplitudes


@pytest.mark.parametrize("x", [0.05, 1., 8.])
def test_impedance_sphere_has_correct_complex_pec_limit(x):
    actual = impedance_mie_coefficients(x, 0.)
    legacy = pec_mie_coefficients(x)
    assert_allclose(actual, -np.conj(legacy), rtol=3e-14)
    angles = np.linspace(0, np.pi, 25)
    assert_allclose(mie_amplitudes(actual, angles), -np.conj(pec_mie_amplitudes(x, angles)), rtol=3e-14)


@pytest.mark.parametrize("er", [1., 4.-0.2j, 1.-100j])
@pytest.mark.parametrize("x", [0.1, 1., 5.])
def test_scaled_bulk_sphere_matches_unscaled_dielectric_reference(x, er):
    assert_allclose(conductor_mie_coefficients(x, er), dielectric_mie_coefficients(x, er), atol=2e-15, rtol=2e-12)


def test_copper_sphere_is_finite_and_converges_to_impedance_sphere():
    frequencies = np.array([1e9, 4e9, 8e9])
    radius, conductivity = 0.016, 5.8e7
    z = (1+1j)*np.sqrt(np.pi*frequencies*mu_0/conductivity)
    bulk, bulk_amp = sphere_reference(frequencies, radius, conductivity, [0., np.pi/2, np.pi])
    sibc, sibc_amp = sphere_reference(frequencies, radius, conductivity, [0., np.pi/2, np.pi], impedance=z)
    assert np.all(np.isfinite(bulk))
    assert_allclose(sibc_amp, bulk_amp, rtol=2e-10)
    assert_allclose(sibc, bulk, rtol=4e-10)


def test_passive_impedance_sphere_has_positive_absorption():
    a, b = impedance_mie_coefficients(2., 5+5j)
    n = np.arange(1, len(a)+1)
    absorption = np.sum((2*n+1)*np.real(a+b-abs(a)**2-abs(b)**2))
    assert absorption > 0


def test_planar_reflection_limits_and_phase():
    assert_allclose(normal_reflection(0.), -1.)
    assert_allclose(normal_reflection(ETA0), 0.)
    reflected = normal_reflection(5+5j)
    assert abs(reflected) < 1
    assert reflected.imag > 0
    assert np.angle(reflected) < np.pi


def test_debye_discrete_symbol_converges_to_physical_permittivity():
    frequencies = np.array([1e9, 4e9, 8e9])
    physical = 2.5+2/(1+2j*np.pi*frequencies*80e-12)
    errors = []
    for dt in (2e-12, 1e-12, 0.5e-12):
        numerical = bulk_discrete_permittivity(frequencies, dt, er_inf=2.5, delta_er=2.)
        assert np.all(numerical.imag < 0)
        errors.append(np.linalg.norm(numerical-physical))
    assert 3.9 < errors[0]/errors[1] < 4.1
    assert 3.9 < errors[1]/errors[2] < 4.1


@pytest.mark.parametrize("delta", [0., 2.])
def test_discrete_reflection_satisfies_half_cell_ampere_equation(delta):
    frequencies = np.array([1e9, 4e9, 8e9])
    dl, dt, z = 0.001, 1.5e-12, 5+5j
    reflection, k = discrete_normal_reflection(frequencies, dt, dl, z, er_inf=2.5, delta_er=delta)
    theta = 2*np.pi*frequencies*dt
    omega = 2*np.sin(theta/2)/dt
    er = bulk_discrete_permittivity(frequencies, dt, er_inf=2.5, delta_er=delta)
    eta = ETA0/np.sqrt(er)
    e_wall = 1+reflection
    h_left = (np.exp(1j*k*dl/2)-reflection*np.exp(-1j*k*dl/2))/eta
    surface_current = np.cos(theta/2)*e_wall/z
    assert_allclose(h_left-surface_current, 1j*omega*epsilon_0*er*(dl/2)*e_wall, atol=1e-17, rtol=2e-12)


def test_discrete_reflection_converges_to_continuum():
    frequencies = np.array([1e9, 4e9, 8e9])
    z = (1+1j)*np.sqrt(np.pi*frequencies*mu_0/1000)
    exact = normal_reflection(z, 2.5+2/(1+2j*np.pi*frequencies*80e-12))
    errors = []
    for dl in (0.001, 0.0005):
        discrete, _ = discrete_normal_reflection(frequencies, dl/(np.sqrt(3)*c), dl, z, er_inf=2.5, delta_er=2.)
        errors.append(np.linalg.norm(discrete-exact))
    assert 3.9 < errors[0]/errors[1] < 4.1


@pytest.mark.integration
def test_debye_wall_phase_validation_detects_missing_polarization_history(tmp_path, monkeypatch):
    """A correct wall passes; omitting its P history must fail the phase gate."""
    from types import SimpleNamespace
    import gprMax.impedance_surfaces as implementation
    from testing.validation.impedance_surface.validate_reflection_phase import trace, analyse, LIMITS

    args = SimpleNamespace(duration=4e-9, threads=2, reuse=False)
    (tmp_path / "_cache").mkdir()
    incident, dt, _ = trace(tmp_path, 0.001, "debye", None, args)
    total, _, wall = trace(tmp_path, 0.001, "debye", 1000., args)
    *_, metrics, passed = analyse(incident, total, dt, 0.001, "debye", wall)
    assert passed, metrics

    compile_original = implementation.compile_impedance_surfaces

    def omit_history(grid):
        system = compile_original(grid)
        assert system.state_p.size > 0
        # Keep the instantaneous mass but remove Re(c P) from Ampere's law.
        system.pole_coeffs[:, 4:6] = 0
        return system

    monkeypatch.setattr(implementation, "compile_impedance_surfaces", omit_history)
    wrong, _, wall = trace(tmp_path, 0.001, "debye", 1000., args)
    *_, metrics, passed = analyse(incident, wrong, dt, 0.001, "debye", wall)
    assert not passed
    assert metrics["discrete_phase_rms_deg"] > 5*LIMITS["discrete_phase_rms_deg"]
