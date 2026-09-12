"""Physical acceptance tests for the validation-only infinite-SIBC limit."""

import numpy as np
import pytest

from testing.validation.impedance_surface.stability import Stepper, audit_geometry, build_case
from testing.validation.sibc_based_pmc.runtime import assert_zero_admittance, exact_pmc
from testing.validation.sibc_based_pmc.validate import (
    cavity_mode,
    image_case,
    plane_trace,
    reflection,
)


@pytest.mark.integration
@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("precision", ("double", "single"))
def test_exact_pmc_matches_all_six_fields_of_mirror_solution(tmp_path, axis, precision):
    result = image_case(tmp_path, axis, precision, steps=50)
    assert result["passed"], result


@pytest.mark.integration
@pytest.mark.parametrize(
    "indices,precision",
    [((1, 1, 0), "double"), ((1, 2, 1), "double"), ((2, 1, 2), "double"), ((1, 2, 1), "single")],
)
def test_rectangular_pmc_cavity_matches_discrete_resonant_mode(tmp_path, indices, precision):
    result, _ = cavity_mode(tmp_path, indices, precision=precision, steps=500)
    assert result["passed"], result


@pytest.mark.integration
def test_exact_limit_is_lossless_with_retained_voxel_energy(tmp_path):
    with exact_pmc():
        grid, occupied = build_case(tmp_path / "energy", geometry="cavity", resistance=50)
    assert_zero_admittance(grid)
    stepper = Stepper(grid, occupied)
    _, q = audit_geometry(stepper)
    metric = np.diag(np.r_[stepper.we, stepper.wh])
    metric[stepper.ne :, : stepper.ne] = 0.5 * stepper.wh[:, None] * q
    metric[: stepper.ne, stepper.ne :] = metric[stepper.ne :, : stepper.ne].T
    transition = stepper.amplification()
    # A complete lossless leapfrog step preserves the modified energy,
    # including all quarter-area electric and half-volume magnetic weights.
    assert np.max(abs(transition.T @ metric @ transition - metric)) < 2e-13


@pytest.mark.integration
def test_exact_pmc_reflects_at_voxel_face_and_blocks_transmission(tmp_path):
    incident, dt = plane_trace(tmp_path, 0, 2, 1, "reference")
    total, actual_dt = plane_trace(tmp_path, 0, 2, 1, "exact")
    assert actual_dt == dt and total.shape == incident.shape
    *_, metrics = reflection(incident, total, dt)
    assert metrics["max_complex_error"] < 2e-7, metrics
    assert abs(metrics["fitted_plane_offset_m"]) < 1e-9, metrics
    assert metrics["transmission_peak_ratio"] < 1e-12, metrics
