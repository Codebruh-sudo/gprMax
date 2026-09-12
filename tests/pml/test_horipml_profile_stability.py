"""A profile instability is present before either SIBC or CFL roundoff."""

import numpy as np
import pytest

import gprMax
from gprMax.updates.cpu_updates import CPUUpdates
from testing.validation.impedance_surface.investigate_pml_profile import (
    DL,
    ETA,
    WAVE_SPEED,
    amplification_matrix,
    analytic_diagnostic,
    coefficients,
    inverse_stretch,
)
from testing.validation.impedance_surface.validate_sibc_pml import advance, build_grid


def test_unshifted_product_has_a_growing_mode_in_the_continuum_limit():
    result = analytic_diagnostic()
    bad = [row for row in result["fourier_cases"] if row["profile"] == "duplicated_unshifted"]
    continuum = result["continuum_dominant_root_per_s"][0]
    assert continuum > 2e9
    assert all(row["spectral_radius"] > 1.0005 for row in bad)
    error = [abs(row["amplitude_growth_per_s"] / continuum - 1) for row in bad]
    assert all(later < earlier / 3 for earlier, later in zip(error, error[1:]))
    repaired = [row for row in result["fourier_cases"] if row["profile"] != "duplicated_unshifted"]
    assert max(row["spectral_radius"] for row in repaired) < 1 + 3e-14


def test_shift_equal_to_first_sigma_cancels_the_extra_pole():
    dt, sigma = 1.9e-12, 2.0
    z = np.exp(1j * np.linspace(0.0001, np.pi - 0.0001, 4096))
    double = [coefficients(0.0, 1.0, sigma, dt), coefficients(sigma, 1.0, sigma, dt)]
    single = [coefficients(0.0, 1.0, 2 * sigma, dt)]
    np.testing.assert_allclose(
        inverse_stretch(z, double), inverse_stretch(z, single), rtol=3e-12, atol=1e-15
    )


@pytest.mark.integration
@pytest.mark.parametrize("shift_scale", (0.0, 1.1))
def test_fourier_matrix_matches_native_cython_fields_and_pml_histories(tmp_path, shift_scale):
    """One complex Fourier step, recovered from two real native-grid runs.

    The observation point and its complete stencil lie inside a uniform
    constant-coefficient z+ slab. Outer boundary samples are not used.
    """
    sigma = 2.0
    scene = gprMax.Scene()
    for obj in (
        gprMax.Domain(p1=(12 * DL, 3 * DL, 12 * DL)),
        gprMax.Discretisation(p1=(DL,) * 3),
        gprMax.PMLThickness(thickness=(0, 0, 0, 0, 0, 4)),
        gprMax.TimeWindow(iterations=1),
        gprMax.TimeStepStabilityFactor(f=0.99),
        gprMax.OMPThreads(1),
    ):
        scene.add(obj)
    for alpha in (0.0, shift_scale * sigma):
        scene.add(
            gprMax.PMLCFS(
                alphascalingprofile="constant",
                alphamin=alpha,
                alphamax=alpha,
                alphascalingdirection="forward",
                kappascalingdirection="forward",
                sigmascalingdirection="forward",
                kappascalingprofile="constant",
                kappamin=1.0,
                kappamax=1.0,
                sigmascalingprofile="constant",
                sigmamin=sigma,
                sigmamax=sigma,
            )
        )
    if shift_scale == 0:
        with pytest.raises(ValueError, match="negative real total stretch"):
            build_grid(scene, tmp_path / "native")
        return
    grid = build_grid(scene, tmp_path / "native")
    (slab,) = grid.pmls["slabs"]
    # The formula diagnostic must agree with the coefficients built by gprMax.
    for index, alpha in enumerate((0.0, shift_scale * sigma)):
        actual = [getattr(slab, name)[index, 0] for name in ("ERA", "ERB", "ERE", "ERF")]
        np.testing.assert_allclose(actual, coefficients(alpha, 1.0, sigma, grid.dt), rtol=2e-14)
    kx, kz = 0.4 / DL, 0.5 / DL
    matrix = amplification_matrix(
        grid.dt,
        2 * np.sin(0.2) / DL,
        2 * np.sin(0.25) / DL,
        [0.0, shift_scale * sigma],
        [sigma, sigma],
    )
    state = np.asarray(
        [
            0.2 + 0.3j,
            -0.1 + 0.8j,
            0.7 - 0.2j,
            0.04 + 0.03j,
            -0.02 + 0.01j,
            0.01 - 0.03j,
            0.07 + 0.02j,
        ]
    )
    point = (6, 1, 10)
    phi_point = (6 - slab.xs, 1 - slab.ys, 10 - slab.zs)
    coordinates = np.indices(grid.Ex.shape)
    phase_e = np.exp(1j * (kx * coordinates[0] * DL + kz * coordinates[2] * DL))
    phase_hx = phase_e * np.exp(0.5j * kz * DL)
    phase_hz = phase_e * np.exp(0.5j * kx * DL)
    native = np.zeros(7, dtype=complex)
    p = np.indices(slab.EPhi2.shape[1:])
    phase_phi_e = np.exp(1j * (kx * (p[0] + slab.xs) * DL + kz * (p[2] + slab.zs) * DL))
    p = np.indices(slab.HPhi1.shape[1:])
    phase_phi_h = np.exp(1j * (kx * (p[0] + slab.xs) * DL + kz * (p[2] + slab.zs + 0.5) * DL))
    for quadrature, project in ((1.0, np.real), (1j, np.imag)):
        grid.reset_fields()
        grid.Ey[:] = project(state[0] * phase_e)
        grid.Hx[:] = project(state[1] * phase_hx) / ETA
        grid.Hz[:] = project(state[2] * phase_hz) / ETA
        for index in range(2):
            slab.HPhi1[index] = project(state[3 + index] * phase_phi_h) / (WAVE_SPEED * grid.dt)
            slab.EPhi2[index] = project(state[5 + index] * phase_phi_e) / (
                WAVE_SPEED * grid.dt * ETA
            )
        advance(CPUUpdates(grid))
        native[:3] += quadrature * np.asarray(
            [grid.Ey[point], ETA * grid.Hx[point], ETA * grid.Hz[point]]
        )
        for index in range(2):
            native[3 + index] += quadrature * WAVE_SPEED * grid.dt * slab.HPhi1[(index, *phi_point)]
            native[5 + index] += (
                quadrature * WAVE_SPEED * grid.dt * ETA * slab.EPhi2[(index, *phi_point)]
            )
    native[:3] /= [phase_e[point], phase_hx[point], phase_hz[point]]
    native[3:5] /= phase_hx[point]
    native[5:] /= phase_e[point]
    np.testing.assert_allclose(native, matrix @ state, rtol=2e-13, atol=1e-14)
