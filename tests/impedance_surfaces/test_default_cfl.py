"""Default timestep rounding and the rounded-coefficient cavity reproducer."""

from decimal import Decimal

import numpy as np
import pytest

from testing.validation.impedance_surface.default_cfl import (
    coefficient_cavity_spectrum,
    full_solver_run,
    timestep_measurement,
)
from testing.validation.impedance_surface.stability import Stepper, build_case


@pytest.mark.integration
def test_default_sibc_timestep_has_one_percent_margin_at_one_mm(tmp_path):
    grid, _ = build_case(tmp_path / "default", courant=None, precision=None)
    measured = timestep_measurement(grid)
    assert grid.Ex.dtype == np.float32
    assert (
        Decimal("0.01")
        <= Decimal(measured["relative_margin_below_cfl_from_c"])
        < Decimal("0.010001")
    )


@pytest.mark.integration
def test_historical_binary64_ulp_margin_does_not_change_float32_cavity_coefficients(tmp_path):
    spectra = []
    for index, factor in enumerate((None, np.nextafter(1.0, 0.0))):
        grid, occupied = build_case(
            tmp_path / str(index),
            geometry="cavity",
            resistance=1e6,
            courant=factor,
            precision=None,
            automatic_timestep=False,
        )
        spectra.append(coefficient_cavity_spectrum(Stepper(grid, occupied)))
    assert spectra[0] == spectra[1]
    assert spectra[0]["coefficient_matrix_spectral_radius"] > 1.0001


@pytest.mark.integration
def test_historical_full_solver_growth_matches_stored_coefficient_prediction(tmp_path):
    # Characterize a known endpoint limitation, rather than labelling a
    # successful diagnostic run as evidence of physical stability.
    result = full_solver_run(tmp_path / "historical_cavity", steps=60000, automatic_timestep=False)
    assert result["parameters"]["timestep_factor"] is None
    assert result["parameters"]["cpu_precision_argument"] is None
    assert Decimal(result["timestep"]["dt_over_cfl_from_c"]) < 1
    assert result["source_free"]["peak_state_norm_over_initial"] > 1e5
    predicted = result["coefficients"]["coefficient_matrix_spectral_radius"]
    actual = result["source_free"]["fitted_tail_growth_per_step"]
    assert abs(predicted - actual) < 2e-6


@pytest.mark.integration
def test_default_full_solver_cavity_is_bounded_by_automatic_timestep(tmp_path):
    result = full_solver_run(tmp_path / "protected_cavity", steps=200000)
    assert result["parameters"]["timestep_factor"] is None
    assert result["parameters"]["cpu_precision_argument"] is None
    assert result["parameters"]["automatic_sibc_timestep"]
    assert result["coefficients"]["coefficient_matrix_spectral_radius"] < 1 + 1e-12
    assert result["source_free"]["peak_state_norm_over_initial"] < 2
    assert result["source_free"]["final_state_norm_over_initial"] < 1


@pytest.mark.integration
def test_small_resolved_timestep_margin_removes_exponential_cavity_growth(tmp_path):
    result = full_solver_run(
        tmp_path / "margin", courant=0.999999, steps=60000, automatic_timestep=False
    )
    assert result["coefficients"]["coefficient_matrix_spectral_radius"] <= 1 + 1e-12
    assert result["source_free"]["peak_state_norm_over_initial"] < 100
    assert result["source_free"]["final_state_norm_over_initial"] < 1


@pytest.mark.integration
def test_default_copper_cavity_is_bounded_in_full_solver(tmp_path):
    result = full_solver_run(tmp_path / "copper", copper=True, steps=20000)
    assert result["source_free"]["finite"]
    assert result["source_free"]["peak_state_norm_over_initial"] < 2
