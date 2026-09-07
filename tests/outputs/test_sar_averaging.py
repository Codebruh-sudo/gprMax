# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

"""Mass-based 1 g/10 g SAR spatial-averaging tests."""

import numpy as np
import pytest

from gprMax.sar_averaging import (
    INVALID,
    VALID,
    _bounds_for_cube,
    _centered_shells_touch_tissue,
    _density_fingerprint,
    _spatial_average_sar_python,
    apply_spatial_average_plan,
    build_spatial_average_plan,
    spatial_average_sar,
)


@pytest.mark.parametrize(
    "orientation,axis,expected_lower,expected_upper",
    (
        (0, 0, 0.5, 2.5),
        (1, 0, -0.5, 1.5),
        (2, 1, 0.5, 2.5),
        (3, 1, -0.5, 1.5),
        (4, 2, 0.5, 2.5),
        (5, 2, -0.5, 1.5),
    ),
)
def test_face_orientation_uses_iec_negative_then_positive_order(
    orientation, axis, expected_lower, expected_upper
):
    lower, upper = _bounds_for_cube(np.ones(3), 2.0, orientation, np.ones(3))

    assert lower[axis] == pytest.approx(expected_lower)
    assert upper[axis] == pytest.approx(expected_upper)


def test_centered_cube_checks_every_intermediate_shell_face():
    tissue = np.ones((7, 7, 7), dtype=bool)
    tissue[2, 2:5, 2:5] = False

    assert not _centered_shells_touch_tissue(tissue, np.ones(3), (3, 3, 3), 4.0, 0)
    tissue[2, 3, 3] = True
    assert _centered_shells_touch_tissue(tissue, np.ones(3), (3, 3, 3), 4.0, 0)


@pytest.mark.parametrize("target_mass", (0.001, 0.01))
def test_uniform_tissue_preserves_constant_local_sar(target_mass):
    density = np.full((14, 14, 14), 1000.0)
    local_sar = np.full(density.shape, 2.5)

    result = spatial_average_sar(
        density,
        local_sar,
        (0.002, 0.002, 0.002),
        target_mass,
    )

    assert result.peak_sar == pytest.approx(2.5)
    np.testing.assert_allclose(result.sar[np.isfinite(result.sar)], 2.5)
    np.testing.assert_allclose(
        result.averaging_mass[np.isfinite(result.averaging_mass)],
        target_mass,
        rtol=2e-9,
    )


def test_fractional_cells_use_piecewise_constant_density_and_mass_weighted_sar():
    density = np.full((10, 10, 10), 1000.0)
    density[5:, :, :] = 2000.0
    local_sar = np.ones_like(density)
    local_sar[5:, :, :] = 3.0

    result = spatial_average_sar(density, local_sar, (0.001, 0.001, 0.001), 0.001)

    cell = (4, 5, 5)
    side = result.averaging_volume[cell] ** (1 / 3)
    upper_x = (cell[0] + 0.5) * 0.001 + 0.5 * side
    high_length = max(0.0, upper_x - 0.005)
    low_length = side - high_length
    expected = (1000 * low_length * 1 + 2000 * high_length * 3) / (
        1000 * low_length + 2000 * high_length
    )
    assert result.sar[cell] == pytest.approx(expected, rel=2e-9)


def test_background_cells_are_not_given_spatial_sar():
    density = np.full((12, 12, 12), np.nan)
    density[2:10, 2:10, 2:10] = 1000.0
    local_sar = np.zeros_like(density)
    local_sar[np.isfinite(density)] = 4.0

    result = spatial_average_sar(density, local_sar, (0.002, 0.002, 0.002), 0.001)

    assert np.all(result.status[~np.isfinite(density)] == INVALID)
    assert np.all(np.isnan(result.sar[~np.isfinite(density)]))
    assert result.peak_sar == pytest.approx(4.0)


def test_insufficient_tissue_mass_produces_no_average():
    density = np.full((4, 4, 4), 1000.0)
    local_sar = np.ones_like(density)

    result = spatial_average_sar(density, local_sar, (0.001, 0.001, 0.001), 0.001)

    assert np.all(np.isnan(result.sar))
    assert np.isnan(result.peak_sar)
    assert not np.any(result.status == VALID)


def test_invalid_density_is_rejected():
    density = np.full((3, 3, 3), 1000.0)
    density[1, 1, 1] = 0
    with pytest.raises(ValueError, match="density must be positive"):
        spatial_average_sar(density, np.ones_like(density), (0.001,) * 3, 0.001)


def test_compiled_spatial_plan_is_deterministic_across_thread_counts():
    density = np.full((18, 17, 16), np.nan)
    density[2:16, 2:15, 2:14] = 950.0
    density[9:16, 2:15, 2:14] = 1300.0
    local_sar = np.zeros_like(density)
    local_sar[np.isfinite(density)] = np.linspace(0.5, 3.5, np.count_nonzero(np.isfinite(density)))

    serial = spatial_average_sar(density, local_sar, (0.0015, 0.0015, 0.0015), 0.001, nthreads=1)
    parallel = spatial_average_sar(density, local_sar, (0.0015, 0.0015, 0.0015), 0.001, nthreads=4)

    np.testing.assert_array_equal(parallel.status, serial.status)
    np.testing.assert_array_equal(parallel.orientation, serial.orientation)
    np.testing.assert_allclose(parallel.sar, serial.sar, equal_nan=True)
    np.testing.assert_allclose(parallel.averaging_mass, serial.averaging_mass, equal_nan=True)


def test_compiled_spatial_plan_matches_python_reference():
    density = np.full((12, 11, 10), np.nan)
    density[1:11, 1:10, 1:9] = 980.0
    density[6:11, 2:9, 2:8] = 1250.0
    local_sar = np.zeros_like(density)
    local_sar[np.isfinite(density)] = np.linspace(0.25, 4.0, np.count_nonzero(np.isfinite(density)))
    arguments = (
        density,
        local_sar,
        (0.0015, 0.0015, 0.0015),
        0.001,
    )

    reference = _spatial_average_sar_python(*arguments)
    compiled = spatial_average_sar(*arguments, nthreads=4)

    np.testing.assert_array_equal(compiled.status, reference.status)
    np.testing.assert_array_equal(compiled.orientation, reference.orientation)
    np.testing.assert_allclose(compiled.sar, reference.sar, equal_nan=True)
    np.testing.assert_allclose(compiled.averaging_mass, reference.averaging_mass, equal_nan=True)
    np.testing.assert_allclose(
        compiled.averaging_volume, reference.averaging_volume, equal_nan=True
    )
    assert compiled.peak_sar == pytest.approx(reference.peak_sar)
    assert compiled.peak_cell == reference.peak_cell


def test_reusable_plan_rejects_changed_density_with_unchanged_tissue_mask():
    density = np.full((8, 8, 8), 1000.0)
    local_sar = np.full(density.shape, 2.5)
    plan = build_spatial_average_plan(density, (0.001,) * 3, 0.0001)
    original = apply_spatial_average_plan(plan, local_sar, density)
    assert original.peak_sar == pytest.approx(2.5)

    density *= 2
    with pytest.raises(ValueError, match="density values differ"):
        apply_spatial_average_plan(plan, local_sar, density)


def test_reusable_plan_spacing_does_not_alias_the_callers_array():
    density = np.full((8, 8, 8), 1000.0)
    local_sar = np.full(density.shape, 2.5)
    spacing = np.full(3, 0.001)
    plan = build_spatial_average_plan(density, spacing, 0.0001)
    original = apply_spatial_average_plan(plan, local_sar, density)

    spacing *= 2
    np.testing.assert_array_equal(plan.spacing, np.full(3, 0.001))
    repeated = apply_spatial_average_plan(plan, local_sar, density)
    np.testing.assert_array_equal(repeated.sar, original.sar)
    assert not plan.spacing.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        plan.spacing[0] = 0.002


@pytest.mark.parametrize("change", ("one_ulp", "reordered"))
def test_reusable_plan_checks_density_values_not_only_global_summaries(change):
    density = np.full((8, 8, 8), 1000.0)
    density[3, 3, 3] = 1200.0
    plan = build_spatial_average_plan(density, (0.001,) * 3, 0.0001)
    changed = density.copy()
    if change == "one_ulp":
        changed[3, 3, 3] = np.nextafter(changed[3, 3, 3], np.inf)
    else:
        changed[3, 3, 3], changed[4, 4, 4] = changed[4, 4, 4], changed[3, 3, 3]

    with pytest.raises(ValueError, match="density values differ"):
        apply_spatial_average_plan(plan, np.ones_like(density), changed)


@pytest.mark.parametrize(
    "representation", ("float32", "fortran", "strided", "big_endian", "nan_payload")
)
def test_reusable_plan_accepts_equivalent_density_representations(representation):
    density = np.full((8, 8, 8), 1000.0)
    density[0] = np.nan
    density[3, 3, 3] = 1200.0
    original_bytes = density.tobytes()
    local_sar = np.full(density.shape, 2.5)
    plan = build_spatial_average_plan(density, (0.001,) * 3, 0.0001)
    original = apply_spatial_average_plan(plan, local_sar, density)
    if representation == "float32":
        equivalent = density.astype(np.float32)
    elif representation == "fortran":
        equivalent = np.asfortranarray(density)
    elif representation == "strided":
        storage = np.empty((16, 8, 8))
        equivalent = storage[::2]
        equivalent[:] = density
        assert not equivalent.flags.c_contiguous
    elif representation == "big_endian":
        equivalent = density.astype(">f8")
    else:
        equivalent = density.copy()
        equivalent.view(np.uint64)[0] = 0x7FF8000000000001
        assert np.all(np.isnan(equivalent[0]))

    equivalent_bytes = equivalent.tobytes()
    repeated = apply_spatial_average_plan(plan, local_sar, equivalent)
    for name in ("sar", "status", "averaging_mass", "averaging_volume", "orientation"):
        np.testing.assert_array_equal(getattr(repeated, name), getattr(original, name))
    assert repeated.peak_sar == original.peak_sar
    assert repeated.peak_cell == original.peak_cell
    assert equivalent.tobytes() == equivalent_bytes
    assert density.tobytes() == original_bytes


def test_reusable_plan_accepts_readonly_inputs_and_multiple_sar_fields():
    density = np.full((8, 8, 8), 1000.0)
    density.setflags(write=False)
    spacing = np.full(3, 0.001)
    spacing.setflags(write=False)
    plan = build_spatial_average_plan(density, spacing, 0.0001)
    for value in (2.5, 7.0, 2.5):
        local_sar = np.full(density.shape, value)
        local_sar.setflags(write=False)
        result = apply_spatial_average_plan(plan, local_sar, density)
        assert result.peak_sar == pytest.approx(value)
        np.testing.assert_allclose(result.sar[np.isfinite(result.sar)], value)


@pytest.mark.parametrize("replacement", (0.0, -1.0, np.nan, np.inf, -np.inf))
def test_reusable_plan_still_rejects_invalid_or_changed_tissue(replacement):
    density = np.full((8, 8, 8), 1000.0)
    plan = build_spatial_average_plan(density, (0.001,) * 3, 0.0001)
    density[3, 3, 3] = replacement
    with pytest.raises(ValueError, match="tissue density|density tissue membership"):
        apply_spatial_average_plan(plan, np.ones_like(density), density)


def test_density_fingerprint_covers_chunk_boundaries_and_final_chunk():
    density = np.full((3, 131072), 1000.0)
    original = _density_fingerprint(density)
    assert len(original) == 32
    for position in (0, 131072, density.size - 1):
        density.flat[position] = 1001.0
        assert _density_fingerprint(density) != original
        density.flat[position] = 1000.0
    assert _density_fingerprint(density) == original
