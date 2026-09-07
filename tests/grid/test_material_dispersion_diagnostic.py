"""Independent continuum and discrete-wave checks of the setup diagnostic."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import brentq

from gprMax import config
from gprMax.dispersion import material_propagation, spatial_resolution
from gprMax.materials import DispersiveMaterial, Material, create_directional_material

pytestmark = pytest.mark.unit


def _material(er=1, mr=1, se=0, sm=0, name="medium"):
    m = Material(3, name)
    m.er, m.mr, m.se, m.sm = er, mr, se, sm
    return m


def _grid(materials, spacing=(0.001,) * 3):
    return SimpleNamespace(
        materials=materials,
        dx=spacing[0],
        dy=spacing[1],
        dz=spacing[2],
        dt=min(spacing) / (config.c * np.sqrt(3)),
    )


@pytest.mark.parametrize("er,mr", ((1, 1), (4, 1), (1, 4), (4, 9)))
def test_phase_velocity_includes_both_constitutive_parameters(er, mr):
    grid = _grid([_material(er, mr)])
    result = spatial_resolution(grid, 1e9, "3D")
    velocity = config.c / np.sqrt(er * mr)
    assert result["phase_velocity"] == pytest.approx(velocity, rel=1e-14)
    assert result["wavelength"] == pytest.approx(velocity / 1e9, rel=1e-14)
    assert result["N"] == pytest.approx(velocity / 1e9 / grid.dx, rel=1e-14)
    # Solve the finite-difference wave equation directly for beta_num, rather
    # than repeating the production arcsine expression.
    omega = 2 * np.pi * result["phase_error_frequency"]
    lhs = (np.sin(omega * grid.dt / 2) / (velocity * grid.dt)) ** 2
    residual = lambda beta: (np.sin(beta * grid.dx / 2) / grid.dx) ** 2 - lhs
    beta_num = brentq(residual, 0, np.pi / grid.dx, xtol=1e-13)
    expected = ((omega / beta_num) / velocity - 1) * 100
    assert result["deltavp"] == pytest.approx(expected, abs=3e-12)


def test_permittivity_and_permeability_swap_preserves_diagnostic():
    a = spatial_resolution(_grid([_material(9, 1)]), 2e9, "3D")
    b = spatial_resolution(_grid([_material(1, 9)]), 2e9, "3D")
    for key in ("phase_velocity", "wavelength", "N", "deltavp"):
        assert a[key] == b[key]


def test_limiting_material_is_not_necessarily_highest_permittivity():
    dielectric = _material(er=9, name="dielectric")
    magnetic = _material(mr=100, name="magnetic")
    result = spatial_resolution(_grid([dielectric, magnetic]), 1e9, "3D")
    assert result["material"] is magnetic


@pytest.mark.parametrize("magnetic", (False, True))
def test_lossy_wave_number_matches_closed_form_conductor(magnetic):
    frequency = np.asarray([0.5e9, 1e9, 2e9])
    omega = 2 * np.pi * frequency
    # Equal electric and magnetic loss tangents must give identical k.
    sigma = 0.3
    ratio = sigma / (omega * config.e0 * 4)
    m = _material(
        4,
        3,
        se=0 if magnetic else sigma,
        sm=sigma * config.m0 * 3 / (config.e0 * 4) if magnetic else 0,
    )
    common = omega / config.c * np.sqrt(12 / 2)
    alpha = common * np.sqrt(np.sqrt(1 + ratio**2) - 1)
    beta = common * np.sqrt(np.sqrt(1 + ratio**2) + 1)
    k = material_propagation(m, frequency)
    np.testing.assert_allclose(k.real, beta, rtol=1e-13)
    np.testing.assert_allclose(-k.imag, alpha, rtol=1e-13)
    result = spatial_resolution(_grid([m]), frequency[-1], "3D")
    assert result["N"] == pytest.approx(2 * np.pi / beta[-1] / 0.001)
    assert result["attenuation_cells"] == pytest.approx(1 / alpha[-1] / 0.001)
    assert result["deltavp"] is None
    assert result["phase_error_notes"]


def test_debye_spectrum_and_internal_lorentz_resonance():
    debye = DispersiveMaterial(3, "debye")
    debye.er, debye.se, debye.poles, debye.type = 2, 0.1, 1, "debye"
    debye.deltaer, debye.tau = [4], [1e-10]
    frequency = np.asarray([0.5e9, 1e9, 2e9])
    omega = 2 * np.pi * frequency
    er = 2 + 4 / (1 + (omega * 1e-10) ** 2)
    ei = -(4 * omega * 1e-10 / (1 + (omega * 1e-10) ** 2) + 0.1 / (omega * config.e0))
    magnitude = np.hypot(er, ei)
    beta = omega / config.c * np.sqrt((magnitude + er) / 2)
    alpha = omega / config.c * np.sqrt((magnitude - er) / 2)
    np.testing.assert_allclose(
        material_propagation(debye, frequency), beta - 1j * alpha, rtol=1e-13
    )

    lorentz = DispersiveMaterial(4, "lorentz")
    lorentz.er, lorentz.poles, lorentz.type = 2, 1, "lorentz"
    lorentz.deltaer, lorentz.tau, lorentz.alpha = [4], [1e9], [1e7]
    result = spatial_resolution(_grid([lorentz]), 3e9, "3D")
    assert 0.99e9 < result["sampling_frequency"] < 1.01e9
    assert result["deltavp"] is None  # No claim about the discrete recurrence.


def test_evanescent_medium_has_no_finite_phase_velocity():
    m = DispersiveMaterial(3, "plasma")
    m.er, m.type, m.poles = 1, "drude", 1
    m.tau, m.alpha = [2e9], [0]
    result = spatial_resolution(_grid([m]), 1e9, "3D")
    assert np.isinf(result["wavelength"])
    assert result["phase_velocity"] is None
    assert np.isfinite(result["attenuation_cells"])
    assert result["deltavp"] is None


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("family", ("TM", "TE"))
def test_invariant_direction_is_excluded_from_all_resolution_metrics(axis, family):
    spacing = [0.001] * 3
    mode = f"2D {family}{'xyz'[axis]}"
    grid = _grid([_material(4, 3)], spacing)
    reference = spatial_resolution(grid, 2e9, mode)
    setattr(grid, "d" + "xyz"[axis], 0.1)
    actual = spatial_resolution(grid, 2e9, mode)
    for key in ("N", "phase_velocity", "wavelength", "deltavp"):
        assert actual[key] == reference[key]


def test_diagonal_anisotropy_uses_bound_not_scalar_mean():
    grid = _grid([_material(100, 1, name="x"), _material(1, 100, name="y"), _material(name="z")])
    # Give distinct numeric IDs, as a built grid does.
    for index, material in enumerate(grid.materials):
        material.numID = index
    directional = create_directional_material(grid, grid.materials)
    result = spatial_resolution(grid, 1e9, "3D")
    assert result["material"] is directional
    assert result["N"] == pytest.approx(config.c / (1e9 * 100 * 0.001))
    assert result["phase_velocity"] is None
    assert result["sampling_kind"] == "anisotropic index-magnitude bound"


def test_nonfinite_material_response_is_actionable():
    with pytest.raises(ValueError, match="non-finite"):
        material_propagation(_material(er=float("nan")), np.asarray([1e9]))


def test_distributed_diagnostic_uses_remote_material_name(make_grid, monkeypatch):
    grid = make_grid(arrays=False)
    local = spatial_resolution(_grid([_material()]), 1e9, "3D")
    local.update(maxfreq=1e9, error="")
    remote = dict(local, N=2, material=SimpleNamespace(ID="remote_generated"))

    def gather(payload):
        report, failure = payload
        assert failure is None
        # Do not gather entire Material instances or depend on rank-local IDs.
        assert vars(report["material"]) == {"ID": "medium"}
        return [payload, (remote, None)]

    grid.comm = SimpleNamespace(allgather=gather)
    monkeypatch.setattr(grid, "_dispersion_analysis", lambda _: local)
    with pytest.raises(ValueError, match="remote_generated"):
        grid.dispersion_analysis(100)


@pytest.mark.parametrize("local_failure", (False, True))
def test_distributed_diagnostic_collects_failures_before_raising(
    make_grid, monkeypatch, local_failure
):
    grid = make_grid(arrays=False)
    result = spatial_resolution(_grid([_material()]), 1e9, "3D")
    result.update(maxfreq=1e9, error="")
    reached = []

    def analyse(_):
        if local_failure:
            raise ValueError("singular pole")
        return result

    def gather(payload):
        reached.append(True)
        return [payload, (None, "singular pole")]

    grid.comm = SimpleNamespace(allgather=gather)
    monkeypatch.setattr(grid, "_dispersion_analysis", analyse)
    with pytest.raises(ValueError, match="singular pole"):
        grid.dispersion_analysis(100)
    assert reached
