"""Analytical profile rejection and actionable build-time diagnostics."""

from types import SimpleNamespace

import numpy as np
import pytest

import gprMax
from gprMax.pml import PML
from testing.validation.impedance_surface.validate_sibc_pml import build_grid, scene_for


def check(factors, formulation="HORIPML"):
    slab = SimpleNamespace(ID="zplus", profile_id="custom", formulation=formulation)
    profiles = [
        tuple(tuple(np.atleast_1d(v).astype(float) for v in values) for values in term)
        for term in factors
    ]
    PML._validate_stretch_profiles(slab, profiles)


@pytest.mark.parametrize("field", (0, 1))
def test_checks_both_staggered_profiles(field):
    safe = ([3.0, 3.0], [1.0, 1.0], [2.0, 2.0])
    bad = ([3.0, 0.0], [1.0, 1.0], [2.0, 2.0])
    term = [safe, safe]
    term[field] = bad
    with pytest.raises(ValueError, match=f"global {'EH'[field]} sample 1.*Reducing the timestep"):
        check((term, term))


@pytest.mark.parametrize(
    "alpha,sigma,kappa",
    (
        ([0.0, 2.2], [2.0, 2.0], [1.0, 1.0]),
        ([0.8, 0.8], [2.0, 2.0], [1.0, 1.0]),
        ([0.0, 0.0], [0.0, 2.0], [1.0, 1.0]),
        ([0.0, 0.0], [0.0, 0.0], [1.0, 1.0]),
    ),
)
def test_accepts_valid_and_disabled_factors(alpha, sigma, kappa):
    check([[(a, k, s)] * 2 for a, k, s in zip(alpha, kappa, sigma)])


def test_does_not_apply_product_condition_to_mripml():
    check([[(0.0, 0.5, 2.0)] * 2] * 2, formulation="MRIPML")


@pytest.mark.parametrize("scale", (1e-100, 1.0, 1e100))
def test_detects_shifted_negative_stretch_independent_of_units(scale):
    with pytest.raises(ValueError, match="negative real total stretch"):
        check([[(0.1 * scale, 1.0, 2.0 * scale)] * 2] * 2)


@pytest.mark.parametrize(
    "formulation,order,duplicated",
    (
        ("HORIPML", 1, False),
        ("HORIPML", 2, False),
        ("MRIPML", 2, False),
        ("HORIPML", 2, True),
    ),
)
def test_native_graded_profile_build(tmp_path, formulation, order, duplicated):
    scene = scene_for(
        48,
        kind="pec",
        formulation=formulation,
        pml_cells=8,
        order=order,
        duplicated_unshifted=duplicated,
    )
    if duplicated:
        with pytest.raises(ValueError, match="negative real total stretch"):
            build_grid(scene, tmp_path / "bad")
    else:
        build_grid(scene, tmp_path / "good")


def test_zero_sigma_is_not_replaced_by_automatic_sigma(tmp_path):
    scene = scene_for(48, kind="pec", pml_cells=8)
    for _ in range(2):
        scene.add(
            gprMax.PMLCFS(
                alphascalingprofile="constant",
                alphascalingdirection="forward",
                alphamin=0.0,
                alphamax=0.0,
                kappascalingprofile="constant",
                kappascalingdirection="forward",
                kappamin=1.0,
                kappamax=1.0,
                sigmascalingprofile="quartic",
                sigmascalingdirection="forward",
                sigmamin=0.0,
                sigmamax=0.0,
            )
        )
    grid = build_grid(scene, tmp_path / "zero")
    for slab in grid.pmls["slabs"]:
        assert all(cfs.sigma.max == 0 for cfs in slab.CFS)
        assert not np.any(slab.ERF)
        assert not np.any(slab.HRF)


def test_global_sample_is_checked_before_local_partition():
    # All valid local slices must share the global profile's rejection.
    good = (np.array([2.0, 2.0, 0.0]), np.ones(3), np.ones(3))
    slab = SimpleNamespace(
        ID="rank_0_zplus",
        profile_id="partitioned",
        formulation="HORIPML",
        profile_offset=0,
        thickness=1,
        profile_thickness=3,
    )
    with pytest.raises(ValueError, match="global E sample 2"):
        PML._validate_stretch_profiles(slab, [(good, good), (good, good)])


@pytest.mark.unit
def test_internal_terminal_e_sample_is_not_omitted(make_pml_grid, make_cfs):
    factors = [
        make_cfs(alpha={"min": 1.0, "max": 1.0}, sigma={"max": 5.0, "scalingprofile": "linear"})
        for _ in range(2)
    ]
    grid = make_pml_grid(nx=20, cfs=factors)
    boundary = PML(grid, "x0", "xminus", 0, 4, 0, 11, 0, 11)
    boundary.calculate_update_coeffs(1.0, 1.0)
    internal = PML(grid, "embedded", "xplus", 5, 9, 0, 11, 0, 11, internal=True)
    with pytest.raises(ValueError, match="global E sample 4"):
        internal.calculate_update_coeffs(1.0, 1.0)


def test_named_virtual_profile_error_identifies_port_and_remedy(tmp_path):
    from testing.validation.impedance_surface.validate_2d import scene_2d

    scene = scene_2d(active=True, steps=100)
    scene.add(gprMax.PMLFormulation(formulation="HORIPML", id="bad_load"))
    for _ in range(2):
        scene.add(
            gprMax.PMLCFS(
                profile_id="bad_load",
                alphascalingprofile="constant",
                alphascalingdirection="forward",
                alphamin=0.0,
                alphamax=0.0,
                kappascalingprofile="constant",
                kappascalingdirection="forward",
                kappamin=1.0,
                kappamax=1.0,
                sigmascalingprofile="quartic",
                sigmascalingdirection="forward",
                sigmamin=0.0,
                sigmamax=None,
            )
        )
    scene.add(
        gprMax.VirtualWaveguide(
            port=1, length_cells=24, pml_cells=8, source_clearance_cells=4, pml_profile="bad_load"
        )
    )
    with pytest.raises(
        ValueError, match="Virtual waveguide port 1.*bad_load.*negative real total stretch"
    ):
        build_grid(scene, tmp_path / "bad_virtual")
