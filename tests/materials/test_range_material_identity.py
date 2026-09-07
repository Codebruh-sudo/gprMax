"""Exact constitutive identity for non-dispersive stochastic material bins."""

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from gprMax.materials import DispersiveMaterial, Material, RangeMaterial


def _range(name="range", **ranges):
    limits = dict(er=(4.0, 4.0), se=(0.0, 0.0), mr=(1.0, 1.0), sm=(0.0, 0.0))
    limits.update(ranges)
    return RangeMaterial(name, *(limits[property_name] for property_name in limits))


@pytest.mark.parametrize(
    "property_name,lower", [("er", 4.0), ("se", 0.0), ("mr", 1.0), ("sm", 0.0)]
)
def test_sub_four_decimal_bins_remain_distinct(property_name, lower):
    grid = SimpleNamespace(materials=[])
    model = _range(**{property_name: (lower, lower + 4e-5)})
    model.calculate_properties(4, grid)

    assert len(set(model.matID)) == 4
    assert len({material.ID for material in grid.materials}) == 4
    actual = [getattr(grid.materials[index], property_name) for index in model.matID]
    expected = lower + np.array([5e-6, 1.5e-5, 2.5e-5, 3.5e-5])
    # The independently specified decimal midpoints can differ by one ULP
    # from midpoints formed from floating-point bin endpoints.
    assert_allclose(actual, expected, rtol=4e-16, atol=1e-20)


def test_even_adjacent_float_properties_are_not_merged():
    grid = SimpleNamespace(materials=[])
    for er in (4.0, np.nextafter(4.0, np.inf)):
        _range(er=(er, er)).calculate_properties(1, grid)
    assert len(grid.materials) == 2
    assert grid.materials[0].er != grid.materials[1].er
    assert grid.materials[0].ID != grid.materials[1].ID


def test_identical_bins_and_overlapping_ranges_share_only_exact_matches():
    grid = SimpleNamespace(materials=[])
    first = _range(se=(0.0, 4e-5))
    first.calculate_properties(4, grid)
    same = _range("second", se=(0.0, 4e-5))
    same.calculate_properties(4, grid)
    se = grid.materials[first.matID[2]].se
    constant = _range("constant", se=(se, se))
    constant.calculate_properties(5, grid)

    assert same.matID == first.matID
    assert constant.matID == [first.matID[2]] * 5
    assert len(grid.materials) == 4


@pytest.mark.parametrize("material_class", [Material, DispersiveMaterial])
@pytest.mark.parametrize("name_kind", ["legacy", "current"])
def test_user_material_names_cannot_supply_a_range_bin(material_class, name_kind):
    probe = SimpleNamespace(materials=[])
    _range().calculate_properties(1, probe)
    name = "|4.0000+0.0000+1.0000+0.0000|" if name_kind == "legacy" else probe.materials[0].ID
    user = material_class(0, name)
    # Matching electromagnetic constants are still not enough to reuse a
    # user material: it may have dispersion, density or other semantics.
    user.er = 4.0
    user.mass_density = 1000.0
    if material_class is DispersiveMaterial:
        user.type = "debye"
        user.poles = 1
        user.deltaer = [2.0]
        user.tau = [1e-10]
    collision = Material(1, name + "_1")
    grid = SimpleNamespace(materials=[user, collision])
    model = _range()
    model.calculate_properties(2, grid)

    assert model.matID == [2, 2]
    assert len({m.ID for m in grid.materials}) == 3
    assert type(grid.materials[2]) is Material
    assert grid.materials[2].er == 4.0
    assert grid.materials[2].mass_density is None
    assert user.mass_density == 1000.0
    assert user.ID == name


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("er", 5.0),
        ("se", 1e-8),
        ("mr", 2.0),
        ("sm", 1e-8),
        ("mass_density", 1000.0),
        ("averagable", False),
        ("type", "changed"),
    ],
)
def test_modified_generated_material_is_not_reused(attribute, value):
    grid = SimpleNamespace(materials=[])
    first = _range()
    first.calculate_properties(1, grid)
    setattr(grid.materials[0], attribute, value)
    second = _range("second")
    second.calculate_properties(1, grid)

    assert second.matID == [1]
    assert grid.materials[1].ID != grid.materials[0].ID
    assert (
        grid.materials[1].er,
        grid.materials[1].se,
        grid.materials[1].mr,
        grid.materials[1].sm,
    ) == (4.0, 0.0, 1.0, 0.0)


def test_generated_names_do_not_depend_on_range_order_or_name():
    catalogues = []
    for values in ((4.0, 8.0), (8.0, 4.0)):
        grid = SimpleNamespace(materials=[])
        for index, er in enumerate(values):
            _range(f"range{index}", er=(er, er)).calculate_properties(1, grid)
        catalogues.append({m.er: m.ID for m in grid.materials})
        assert all(m.is_compound_material() for m in grid.materials)
    assert catalogues[0] == catalogues[1]


def test_reuse_lookup_is_local_to_each_grid():
    first_grid = SimpleNamespace(materials=[])
    _range().calculate_properties(1, first_grid)
    second_grid = SimpleNamespace(materials=[Material(0, "unrelated")])
    model = _range()
    model.calculate_properties(1, second_grid)
    assert model.matID == [1]
    assert second_grid.materials[1] is not first_grid.materials[0]
