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

"""Compare SIBC symmetry cuts with fully modeled mirror geometries."""

import numpy as np
import pytest

import gprMax
import gprMax.impedance_surfaces as implementation


pytestmark = pytest.mark.integration
DL = 0.001
N = 8


def _scene(axis, maximum, kind, *, full=False, touch=True, declare=True, normal_source=False, exterior=None):
    size = np.full(3, N)
    if full:
        size[axis] *= 2
    scene = gprMax.Scene()
    for obj in (
        gprMax.Domain(p1=tuple(size * DL)),
        gprMax.Discretisation(p1=(DL, DL, DL)),
        gprMax.TimeWindow(iterations=160),
        gprMax.PMLThickness(thickness=0),
        gprMax.OMPThreads(1),
        gprMax.SurfaceImpedance(
            id="wall", preset="copper", fit_frequency_range=(8e9, 12e9), fit_order=4,
        ),
        gprMax.Waveform(wave_type="ricker", amp=1, freq=8e9, id="pulse"),
        gprMax.Waveform(
            wave_type="ricker", amp=1 if (kind == "pec") == normal_source else -1, freq=8e9, id="image",
        ),
    ):
        scene.add(obj)
    if not full and declare:
        scene.add(gprMax.SymmetryBoundary(face="xyz"[axis] + ("max" if maximum else "0"), type=kind))
    if exterior is not None:
        from testing.benchmarking.benchmark_impedance_box import add_exterior

        # Full and half domains have different x extents but the same material.
        add_exterior(scene, exterior, N*DL)
        if full:
            scene.add(gprMax.Box(p1=(0, 0, 0), p2=tuple(size*DL), material_id=f"benchmark_{exterior}"))
    lower, upper = np.full(3, 3), np.full(3, 5)
    if touch:
        lower[axis], upper[axis] = ((N - 2, N + 2) if full else (N - 2, N) if maximum else (0, 2))
    scene.add(gprMax.Box(p1=tuple(lower * DL), p2=tuple(upper * DL), material_id="wall"))
    source = np.full(3, 4)
    source[axis] = (N - 3 if maximum else 3) if touch else 1
    if full and not maximum:
        source[axis] += N
    polarisation = "xyz"[axis if normal_source else (axis + 1) % 3]
    scene.add(gprMax.HertzianDipole(tuple(source * DL), polarisation, "pulse"))
    if full:
        image = source.copy()
        image[axis] = 2 * N - source[axis] - int(normal_source)
        scene.add(gprMax.HertzianDipole(tuple(image * DL), polarisation, "image"))
    return scene


def _run(scene, path, monkeypatch, *, solve=True):
    captured = {}
    original = implementation.compile_impedance_surfaces

    def capture(grid):
        captured["grid"] = grid
        return original(grid)

    with monkeypatch.context() as patch:
        patch.setattr(implementation, "compile_impedance_surfaces", capture)
        gprMax.run(
            scenes=[scene], outputfile=path, geometry_only=not solve,
            cpu_precision="double", hide_progress_bars=True,
        )
    return captured["grid"]


@pytest.mark.parametrize("axis", range(3), ids=("x", "y", "z"))
@pytest.mark.parametrize("maximum", (False, True), ids=("min", "max"))
@pytest.mark.parametrize("kind", ("pec", "pmc"))
@pytest.mark.parametrize("normal_source", (False, True), ids=("tangential-source", "normal-source"))
def test_symmetry_cut_matches_full_mirrored_dynamic_sibc(
    axis, maximum, kind, normal_source, tmp_path, monkeypatch,
):
    half = _run(_scene(axis, maximum, kind, normal_source=normal_source), tmp_path / "half", monkeypatch)
    full = _run(_scene(axis, maximum, kind, full=True, normal_source=normal_source), tmp_path / "full", monkeypatch)
    assert np.any(half.impedance_surfaces.state_y != 0)
    for component, name in enumerate(("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")):
        actual = implementation._component_valid_view(getattr(half, name), component, half)
        expected = implementation._component_valid_view(getattr(full, name), component, full)
        selection = [slice(None)] * 3
        start = 0 if maximum else N
        selection[axis] = slice(start, start + actual.shape[axis])
        expected = expected[tuple(selection)]
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10 * max(1, np.max(np.abs(expected))))

    system = half.impedance_surfaces
    on_plane = (system.edge_info[:, 0] != axis) & (system.edge_info[:, 1 + axis] == (N if maximum else 0))
    if kind == "pec":
        assert not np.any(on_plane)
    else:
        assert np.any(on_plane)
        np.testing.assert_array_equal(system.edge_fraction[on_plane], 0.25)
        for row in system.edge_info[on_plane]:
            ports = slice(row[6], row[6] + row[7])
            assert row[7] == 1
            assert system.port_normal[ports][0, 0] != axis
            np.testing.assert_allclose(system.port_area[ports], DL**2 / 2)
            # The local current/state is the same as in the full domain;
            # only its physical surface integration area is halved.
            full_coord = row[:4].copy()
            if not maximum:
                full_coord[axis + 1] += N
            other = full.impedance_surfaces
            index = np.flatnonzero(np.all(other.edge_info[:, :4] == full_coord, axis=1))[0]
            full_port = other.edge_info[index, 6]
            model, state_start = system.port_info[row[6]]
            order = system.model_info[model, 0]
            full_start = other.port_info[full_port, 1]
            np.testing.assert_allclose(
                system.state_y[state_start:state_start + order], other.state_y[full_start:full_start + order],
                rtol=1e-10, atol=1e-12,
            )


@pytest.mark.parametrize("exterior", ("debye", "lorentz", "drude"))
@pytest.mark.parametrize("kind", ("pec", "pmc"))
def test_dispersive_symmetry_contact_matches_full_domain(exterior, kind, tmp_path, monkeypatch):
    half = _run(_scene(0, False, kind, exterior=exterior), tmp_path / "half", monkeypatch)
    full = _run(_scene(0, False, kind, exterior=exterior, full=True), tmp_path / "full", monkeypatch)
    assert np.any(half.impedance_surfaces.state_p)
    for component, name in enumerate(("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")):
        actual = implementation._component_valid_view(getattr(half, name), component, half)
        expected = implementation._component_valid_view(getattr(full, name), component, full)
        expected = expected[N:N+actual.shape[0]]
        np.testing.assert_allclose(actual, expected, rtol=1e-10,
                                   atol=1e-10*max(1, np.max(np.abs(expected))))


@pytest.mark.parametrize("kind", ("pec", "pmc"))
def test_symmetry_can_coexist_with_interior_sibc(kind, tmp_path, monkeypatch):
    grid = _run(_scene(0, False, kind, touch=False), tmp_path / "interior", monkeypatch, solve=False)
    assert grid.impedance_surfaces.edge_count > 0


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("maximum", (False, True))
def test_contact_with_undeclared_domain_face_still_fails(axis, maximum, tmp_path, monkeypatch):
    scene = _scene(axis, maximum, "pmc", declare=False)
    with pytest.raises(ValueError, match="retained cell.*non-symmetry side"):
        _run(scene, tmp_path / "unsupported", monkeypatch, solve=False)


def _corner_scene(kinds, maximum, *, full=False):
    scene = gprMax.Scene()
    size = np.array((2 * N, 2 * N, N) if full else (N, N, N))
    for obj in (
        gprMax.Domain(p1=tuple(size * DL)),
        gprMax.Discretisation(p1=(DL, DL, DL)),
        gprMax.TimeWindow(iterations=160),
        gprMax.PMLThickness(thickness=0),
        gprMax.OMPThreads(1),
        gprMax.SurfaceImpedance(id="wall", preset="copper", fit_frequency_range=(8e9, 12e9), fit_order=4),
        gprMax.Waveform(wave_type="ricker", amp=1, freq=8e9, id="positive"),
        gprMax.Waveform(wave_type="ricker", amp=-1, freq=8e9, id="negative"),
    ):
        scene.add(obj)
    lower, upper, source = np.full(3, 3), np.full(3, 5), np.full(3, 4)
    for axis, kind in enumerate(kinds):
        lower[axis], upper[axis] = ((N - 2, N + 2) if full else (N - 2, N) if maximum else (0, 2))
        source[axis] = N - 3 if maximum else (N + 3 if full else 3)
        if not full:
            scene.add(gprMax.SymmetryBoundary(face="xy"[axis] + ("max" if maximum else "0"), type=kind))
    scene.add(gprMax.Box(p1=tuple(lower * DL), p2=tuple(upper * DL), material_id="wall"))
    for image_bits in range(4 if full else 1):
        point = source.copy()
        sign = 1
        for axis, kind in enumerate(kinds):
            if image_bits & (1 << axis):
                point[axis] = 2 * N - source[axis]
                sign *= -1 if kind == "pec" else 1
        scene.add(gprMax.HertzianDipole(tuple(point * DL), "z", "positive" if sign > 0 else "negative"))
    return scene


@pytest.mark.parametrize("kinds", (("pec", "pec"), ("pmc", "pmc"), ("pec", "pmc"), ("pmc", "pec")))
@pytest.mark.parametrize("maximum", (False, True))
def test_two_intersecting_symmetry_planes_match_full_domain(kinds, maximum, tmp_path, monkeypatch):
    half = _run(_corner_scene(kinds, maximum), tmp_path / "quarter", monkeypatch)
    full = _run(_corner_scene(kinds, maximum, full=True), tmp_path / "full", monkeypatch)
    for component, name in enumerate(("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")):
        actual = implementation._component_valid_view(getattr(half, name), component, half)
        expected = implementation._component_valid_view(getattr(full, name), component, full)
        start = 0 if maximum else N
        expected = expected[start:start + actual.shape[0], start:start + actual.shape[1], :]
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10 * max(1, np.max(np.abs(expected))))
