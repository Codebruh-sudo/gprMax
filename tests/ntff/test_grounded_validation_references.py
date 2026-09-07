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

import h5py
import numpy as np
import pytest

from testing.validation.planar_layered_ntff import validate_grounded_dipoles as dipoles
from testing.validation.planar_layered_ntff import validate_grounded_slab_reflection as slab


def _case(name):
    return next(case for case in dipoles.CASES if case.name == name)


# Fractions of a cell from the source's grid-index anchor. Keep these explicit
# so a shared coordinate-helper error cannot also move the test's reference.
OFFSETS = {
    ("electric", "x"): (0.5, 0, 0),
    ("electric", "y"): (0, 0.5, 0),
    ("electric", "z"): (0, 0, 0.5),
    ("magnetic", "x"): (0, 0.5, 0.5),
    ("magnetic", "y"): (0.5, 0, 0.5),
    ("magnetic", "z"): (0.5, 0.5, 0),
}


@pytest.mark.parametrize("kind,axis", OFFSETS)
@pytest.mark.parametrize("dl", (0.0015, 0.00075))
def test_source_centres_follow_electric_and_magnetic_staggering(monkeypatch, kind, axis, dl):
    monkeypatch.setattr(dipoles, "DL", dl)
    anchor = np.array(dipoles.SOURCE_ANCHOR)
    expected = anchor + dl * np.array(OFFSETS[kind, axis])
    actual = dipoles._physical_source_position(dipoles.Case("test", kind, axis, False))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(dipoles.SOURCE_ANCHOR, anchor)


@pytest.mark.parametrize("kind,axis", OFFSETS)
def test_source_anchor_is_snapped_before_staggering(monkeypatch, kind, axis):
    monkeypatch.setattr(dipoles, "DL", 0.003)
    # The nominal z=46.5 mm lies halfway between nodes at 45 and 48 mm.
    # The input discretisation rounds half values down to node 15 (45 mm),
    # then the component
    # offset is applied. Rounding the already-staggered point is different.
    expected = np.array((0.060, 0.060, 0.045)) + 0.003 * np.array(OFFSETS[kind, axis])
    actual = dipoles._physical_source_position(dipoles.Case("test", kind, axis, False))
    np.testing.assert_array_equal(actual, expected)


def _image_fields(case, theta, phi, frequency):
    """Direct point moment plus its PEC image; no TE/TM reflection helper."""
    direction = np.stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)), axis=-1
    )
    basis_theta = np.stack(
        (np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)), axis=-1
    )
    basis_phi = np.stack((-np.sin(phi), np.cos(phi), np.zeros_like(phi)), axis=-1)
    location = np.array(dipoles.SOURCE_ANCHOR) + dipoles.DL * np.array(
        OFFSETS[case.source_kind, case.polarisation]
    )
    image_location = location.copy()
    image_location[2] = 2 * dipoles.GROUND - location[2]
    moment = np.eye(3)["xyz".index(case.polarisation)]
    # Electric moments normal to PEC have equal images; tangential moments
    # have opposite images. Magnetic moments have the dual image signs.
    signs = (-1, -1, 1) if case.source_kind == "electric" else (1, 1, -1)
    vector = np.zeros(direction.shape, dtype=complex)
    for centre, dipole in ((location, moment), (image_location, moment * signs)):
        if case.source_kind == "electric":
            shape = dipole - direction * (direction @ dipole)[..., None]
        else:
            shape = np.cross(direction, dipole)
        phase = np.exp(
            2j * np.pi * frequency / dipoles.c * ((centre - dipoles.ORIGIN) @ direction.T)
        )
        vector += phase[:, None] * shape
    return np.sum(vector * basis_theta, axis=-1), np.sum(vector * basis_phi, axis=-1)


@pytest.mark.parametrize("kind,axis", OFFSETS)
def test_bare_pec_complex_fields_match_independent_image_dipoles(kind, axis):
    case = dipoles.Case("test", kind, axis, False)
    theta = np.deg2rad((0, 10, 30, 55, 80))
    phi = np.deg2rad((17, 39, 143, 218, 307))
    for frequency in dipoles.FREQUENCIES:
        actual = dipoles.analytical_fields(case, theta, phi, frequency)
        expected = _image_fields(case, theta, phi, frequency)
        np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=3e-15)


@pytest.mark.parametrize("kind", ("electric", "magnetic"))
@pytest.mark.parametrize("offset_in_dt", (0, 0.5))
def test_absolute_factor_uses_sampled_drive_time_and_spatial_scale(tmp_path, kind, offset_in_dt):
    dt, scale = 2e-12, 0.0015 if kind == "electric" else 1.0
    case = dipoles.Case("test", kind, "x", False)
    with h5py.File(tmp_path / "source.h5", "w") as f:
        excitation = f.create_group("excitation")
        excitation.create_dataset("samples", data=(0.0, 2.0, 0.0))
        excitation.attrs.update(
            SampleInterval=dt, TimeSampleOffset=offset_in_dt * dt, SpatialScale=scale
        )
        actual = dipoles._source_field_factor(case, excitation)
    # One nonzero sample has a closed-form DFT, independent of the driver's
    # summation. No field fit or resampled continuous waveform is involved.
    omega = 2 * np.pi * dipoles.FREQUENCIES
    moment = 2 * dt * scale * np.exp(-1j * omega * (1 + offset_in_dt) * dt)
    expected = 1j * omega / dipoles.c / (4 * np.pi) * moment
    if kind == "electric":
        expected *= -np.sqrt(dipoles.mu_0 / dipoles.epsilon_0)
    np.testing.assert_allclose(actual, expected, rtol=3e-15, atol=0)


@pytest.mark.parametrize("gain", (1.02, np.exp(0.1j)))
def test_absolute_check_rejects_errors_hidden_by_pattern_fit(tmp_path, monkeypatch, gain):
    case = _case("magnetic_tangential_bare")
    path = tmp_path / "fields.h5"
    dt = 2e-12
    with h5py.File(path, "w") as f:
        source = f.create_group("srcs/src1/excitation")
        source.create_dataset("samples", data=(1.0, 0.0))
        source.attrs.update(SampleInterval=dt, TimeSampleOffset=0.0, SpatialScale=1.0)
        for name, angle in (("e_plane", 0.0), ("h_plane", np.pi / 2)):
            theta = np.deg2rad(dipoles.THETA)
            phi = np.full_like(theta, angle)
            predicted = np.array(
                [_image_fields(case, theta, phi, frequency) for frequency in dipoles.FREQUENCIES]
            )
            factor = 1j * (2 * np.pi * dipoles.FREQUENCIES / dipoles.c) * dt / (4 * np.pi)
            values = gain * factor[:, None, None] * predicted
            group = f.create_group("ntff/surface/frequency/spectrum/far_field/" + name)
            group.create_dataset("fields/Etheta", data=values[:, 0, :])
            group.create_dataset("fields/Ephi", data=values[:, 1, :])
            group.create_dataset("fields/directivity", data=np.ones((3, theta.size)))
            group.create_dataset("maximum_directivity", data=np.ones(3))
    # Isolate the amplitude/phase metric: directivity integration is not the
    # subject of this synthetic check.
    monkeypatch.setattr(dipoles, "_analytical_directivity", lambda *args: 1.0)
    metrics, _ = dipoles.compare_case(case, path)
    for values in metrics["frequencies"].values():
        assert values["vector_field_maximum_error_peak_normalised"] < 2e-14
        assert values["absolute_vector_field_relative_l2_error"] == pytest.approx(
            abs(gain - 1), rel=2e-13
        )
    assert not dipoles._acceptance({case.name: metrics})["passed"]


def test_bare_pec_reflection_coefficients_are_minus_one():
    theta = np.deg2rad((0, 20, 45, 80))
    tm, te = dipoles._grounded_reflection(theta, 2e9, coated=False)
    np.testing.assert_array_equal(tm, -np.ones(theta.size))
    np.testing.assert_array_equal(te, -np.ones(theta.size))


def test_bare_pec_dipole_oracle_recovers_image_theory_power():
    theta = np.deg2rad(np.asarray((10, 30, 55, 80), dtype=float))
    phi = np.zeros_like(theta)
    frequency = 2e9
    wavenumber = 2 * np.pi * frequency / dipoles.c

    electric_normal = _case("electric_normal_bare")
    etheta, ephi = dipoles.analytical_fields(electric_normal, theta, phi, frequency)
    height = dipoles.SOURCE_ANCHOR[2] + dipoles.DL / 2 - dipoles.GROUND
    expected = 4 * np.sin(theta) ** 2 * np.cos(wavenumber * height * np.cos(theta)) ** 2
    np.testing.assert_allclose(np.abs(etheta) ** 2, expected, rtol=2e-14, atol=2e-15)
    np.testing.assert_allclose(ephi, 0, atol=2e-15)

    magnetic_tangential = _case("magnetic_tangential_bare")
    etheta, ephi = dipoles.analytical_fields(magnetic_tangential, theta, phi, frequency)
    height = dipoles.SOURCE_ANCHOR[2] + dipoles.DL / 2 - dipoles.GROUND
    expected = 4 * np.cos(theta) ** 2 * np.cos(wavenumber * height * np.cos(theta)) ** 2
    np.testing.assert_allclose(np.abs(ephi) ** 2, expected, rtol=2e-14, atol=2e-15)
    np.testing.assert_allclose(etheta, 0, atol=2e-15)


def test_lossless_grounded_slab_reflection_has_unit_magnitude():
    frequencies = np.linspace(slab.FREQUENCY_MIN, slab.FREQUENCY_MAX, 101)
    reflection = slab._analytical_reflection(frequencies)
    np.testing.assert_allclose(np.abs(reflection), 1, rtol=2e-15, atol=2e-15)
