"""Finite and dispersive walls must remain matched through longitudinal PML."""

import pytest

from testing.validation.impedance_surface.validate_sibc_pml import compare_packet


@pytest.mark.parametrize(
    "kind,axis,formulation,order",
    (
        ("resistance", 0, "HORIPML", 1),
        ("foster", 1, "MRIPML", 2),
    ),
)
def test_sibc_pml_pulse_matches_causally_longer_guide(tmp_path, kind, axis, formulation, order):
    result, _ = compare_packet(
        tmp_path, kind=kind, axis=axis, formulation=formulation, order=order, precision="double"
    )
    assert result["passed"], result
    assert result["pml_edge_count"] > 0
    assert result["relative_reflection_peak"] < 1e-4
    assert result["timestep_factor"] == pytest.approx(0.99)
    if kind == "foster":
        assert result["surface_state_peak"] > 1e-4
