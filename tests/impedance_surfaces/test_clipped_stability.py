"""Energy, spectral, and long-time checks of the compiled E/H coupling."""

import numpy as np
import pytest
from scipy.linalg import eigvals

from testing.validation.impedance_surface.stability import (
    Stepper,
    audit_geometry,
    build_case,
    source_free_run,
)


@pytest.mark.integration
@pytest.mark.parametrize(
    "geometry,spacing",
    [
        ("box", (0.001,) * 3),
        ("stair", (0.001,) * 3),
        ("plate", (0.001,) * 3),
        ("cavity", (0.001,) * 3),
        ("stair", (0.0005, 0.001, 0.002)),
    ],
)
def test_clipped_curls_are_energy_adjoint_and_obey_cartesian_cfl(tmp_path, geometry, spacing):
    grid, occupied = build_case(
        tmp_path / "geometry",
        geometry=geometry,
        spacing=spacing,
        courant=1.0,
        automatic_timestep=False,
    )
    stepper = Stepper(grid, occupied)
    result, _ = audit_geometry(stepper)
    assert result["weighted_adjoint_relative_defect"] < 2e-14
    assert result["dt_times_max_frequency"] <= 2 + 2e-12
    if geometry == "cavity":
        # The isolated retained voxel attains the bound: a strict margin matters.
        assert result["dt_times_max_frequency"] > 2 - 1e-10
        assert 0.25 in stepper.system.edge_fraction


@pytest.mark.integration
@pytest.mark.parametrize("copper", (False, True))
def test_complete_field_and_surface_state_spectrum_has_no_growing_mode(tmp_path, copper):
    grid, occupied = build_case(tmp_path / "spectrum", copper=copper)
    stepper = Stepper(grid, occupied)
    matrix = stepper.amplification()
    assert np.max(np.abs(eigvals(matrix))) < 1 + 2e-10
    # Check that the dense diagnostic describes a general production step,
    # including simultaneous nonzero E, H, and ADE histories.
    state = np.random.default_rng(91).normal(size=stepper.size)
    stepper.set(state)
    stepper.step()
    np.testing.assert_allclose(stepper.get(), matrix @ state, rtol=2e-11, atol=2e-11)


@pytest.mark.integration
def test_above_cfl_negative_control_is_detected(tmp_path):
    grid, occupied = build_case(tmp_path / "unstable", courant=1.2, automatic_timestep=False)
    stepper = Stepper(grid, occupied)
    result, q = audit_geometry(stepper)
    assert result["positive_energy_margin"] < 0
    assert np.max(np.abs(eigvals(stepper.amplification()))) > 2
    result = source_free_run(stepper, q, steps=100)
    assert result["peak_state_norm_over_initial"] > 1e20


@pytest.mark.integration
@pytest.mark.parametrize("precision", ("single", "double"))
def test_high_resistance_quarter_cell_cavity_is_bounded_with_cfl_margin(tmp_path, precision):
    grid, occupied = build_case(
        tmp_path / "cavity", geometry="cavity", resistance=1e6, precision=precision, courant=0.99
    )
    stepper = Stepper(grid, occupied)
    geometry, q = audit_geometry(stepper)
    assert geometry["positive_energy_margin"] > 0.00999
    result = source_free_run(stepper, q, steps=20000)
    assert result["finite"]
    assert result["peak_state_norm_over_initial"] < 3
    assert result["peak_modified_field_energy_over_initial"] < 1.00001


@pytest.mark.integration
def test_resistive_full_update_satisfies_discrete_energy_dissipation(tmp_path):
    from gprMax import config

    grid, occupied = build_case(tmp_path / "energy")
    stepper = Stepper(grid, occupied)
    _, q = audit_geometry(stepper)
    n, ne = stepper.size, stepper.ne
    metric = np.diag(np.r_[stepper.we, stepper.wh])
    metric[ne:, :ne] = 0.5 * stepper.wh[:, None] * q
    metric[:ne, ne:] = metric[ne:, :ne].T
    transition = stepper.amplification()
    damping = np.zeros(ne)
    lookup = {}
    offset = 0
    for component, (idx, count) in enumerate(zip(stepper.indices[:3], stepper.sizes[:3])):
        lookup.update({(component, *p): offset + i for i, p in enumerate(zip(*idx))})
        offset += count
    s = stepper.system
    volume = grid.dx * grid.dy * grid.dz
    for edge in s.edge_info:
        ports = slice(edge[6], edge[6] + edge[7])
        damping[lookup[tuple(edge[:4])]] = (
            grid.dt
            / (config.sim_config.em_consts["e0"] * volume)
            * np.sum(s.port_area[ports] * s.port_inv_Z0[ports])
        )
    midpoint = 0.5 * (transition[:ne] + np.eye(n)[:ne])
    defect = (
        transition.T @ metric @ transition - metric + 2 * midpoint.T @ (damping[:, None] * midpoint)
    )
    assert np.max(np.abs(defect)) < 2e-13
