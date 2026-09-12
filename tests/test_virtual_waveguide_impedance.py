"""Physical split-guide checks for extruded surface-impedance walls."""

import copy

import numpy as np
import pytest

from gprMax.fdfd_eigenmode_solver.surface_impedance_operator import evaluate_surface_ade
from testing.validation.impedance_surface.virtual_waveguide import (
    active_guide,
    compare_guides,
    run_guide,
)


def test_exact_pmc_frequency_response_has_zero_admittance():
    response = evaluate_surface_ade(
        frequency_hz=22e9, dt=1e-12, F=np.empty((0, 0)), G=(), L=(), Z0=np.inf
    )
    assert response.admittance == 0j
    assert np.isposinf(response.impedance.real)


@pytest.mark.parametrize("resistance", [np.inf, 1000.0, "copper"])
@pytest.mark.integration
def test_extruded_impedance_virtual_guide_matches_monolithic(tmp_path, resistance):
    reference, _ = run_guide(tmp_path / "reference", virtual=False, resistance=resistance)
    actual, grid = run_guide(tmp_path / "split", virtual=True, resistance=resistance)
    scale = np.max(np.abs(reference), axis=-1, keepdims=True)
    scale = np.maximum(scale, np.max(scale) * 1e-12)
    assert np.max(np.abs(actual - reference) / scale) < 2e-8
    guide = grid.virtual_waveguides[0]
    assert guide.aux_grid.impedance_surfaces.edge_count > 0
    assert len(guide.aux_grid.impedance_surfaces.pml_edge_indices) > 0
    assert len(grid.impedance_surfaces.virtual_frozen_edges) > 0
    assert np.all(np.isfinite(guide.aux_grid.impedance_surfaces.state_y))


@pytest.mark.parametrize(
    "normal_axis,direction", [(0, "+"), (0, "-"), (1, "+"), (1, "-"), (2, "-")]
)
@pytest.mark.parametrize("formulation", ["HORIPML", "MRIPML"])
@pytest.mark.integration
def test_impedance_aperture_closure_is_rotation_and_direction_invariant(
    tmp_path, normal_axis, direction, formulation
):
    result = compare_guides(
        tmp_path,
        resistance=1000.0,
        normal_axis=normal_axis,
        direction=direction,
        formulation=formulation,
    )
    assert result["passed"], result


@pytest.mark.parametrize("resistance", [np.inf, 1000.0, "copper"])
@pytest.mark.integration
def test_active_surface_source_and_histories_terminate_with_low_reflection(tmp_path, resistance):
    result = active_guide(tmp_path / "active", resistance=resistance, steps=1200)
    assert result["passed"], result


@pytest.mark.integration
def test_virtual_surface_histories_reset_and_require_opaque_window_padding(tmp_path):
    _, grid = run_guide(tmp_path / "state", virtual=True, resistance="copper", steps=400)
    guide = grid.virtual_waveguides[0]
    system = guide.aux_grid.impedance_surfaces
    names = ("EPhi1", "EPhi2", "HPhi1", "HPhi2")
    assert np.any(system.state_y)
    assert any(
        np.any(getattr(slab, name)) for slab in guide.aux_grid.pmls["slabs"] for name in names
    )
    guide.reset_run_state()
    assert not np.any(system.state_y)
    assert not any(
        np.any(getattr(slab, name)) for slab in guide.aux_grid.pmls["slabs"] for name in names
    )
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        assert not np.any(getattr(guide.aux_grid, name))

    cropped = copy.copy(guide)
    cropped.v0 = 4  # The lower SIBC wall now coincides with the modal-window edge.
    with pytest.raises(ValueError, match="strictly inside"):
        cropped._validate_impedance_cross_section()
