"""Shared mixing definitions must not share a mutable per-volume bin map."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from gprMax.materials import (
    CrimMixture,
    DispersiveMaterial,
    ListMaterial,
    Material,
    PeplinskiSoil,
    RangeMaterial,
    create_built_in_materials,
)
from gprMax.user_objects.cmds_geometry.add_surface_roughness import AddSurfaceRoughness
from gprMax.user_objects.cmds_geometry.fractal_box import FractalBox

pytestmark = pytest.mark.unit
KINDS = ("range", "list", "peplinski", "crim")


def _grid(factory):
    grid = factory(nx=32)
    grid.materials = []
    create_built_in_materials(grid)
    for i in range(8):
        material = Material(len(grid.materials), f"dielectric{i}")
        material.er = 2 + i
        grid.materials.append(material)
    water = DispersiveMaterial(len(grid.materials), "water")
    water.type = "debye"
    water.poles = 1
    water.er = 4.9
    water.deltaer = [75.2]
    water.tau = [8.5e-12]
    grid.materials.append(water)
    return grid


def _model(kind, name="mix"):
    if kind == "range":
        return RangeMaterial(name, (4, 8), (0, 4e-5), (1, 1), (0, 0))
    if kind == "list":
        return ListMaterial(name, [f"dielectric{i}" for i in range(8)])
    if kind == "peplinski":
        return PeplinskiSoil(name, 0.5, 0.5, 2.0, 2.66, (0.01, 0.25))
    return CrimMixture(name, "dielectric3", 0.6, "water", 0.02, 0.35, 1e6, 3e9)


def _properties(material):
    return (
        material.er,
        material.se,
        material.mr,
        material.sm,
        tuple(getattr(material, "deltaer", ())),
        tuple(getattr(material, "tau", ())),
    )


def _bin_properties(grid, ids):
    return [_properties(grid.materials[i]) for i in ids]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("counts", [(4, 8), (8, 4), (4, 4)])
def test_recalculation_replaces_mapping_without_mutating_previous_result(
    fractal_grid, kind, counts
):
    grid = _grid(fractal_grid)
    model = _model(kind)
    model.calculate_properties(counts[0], grid)
    previous = model.matID
    previous_ids = previous.copy()
    model.calculate_properties(counts[1], grid)

    reference_grid = _grid(fractal_grid)
    reference = _model(kind)
    reference.calculate_properties(counts[1], reference_grid)
    assert len(model.matID) == counts[1]
    assert previous == previous_ids
    assert _bin_properties(grid, model.matID) == _bin_properties(reference_grid, reference.matID)


@pytest.mark.parametrize("kind", KINDS)
def test_mapping_uses_ids_from_current_grid(fractal_grid, kind):
    first, second = _grid(fractal_grid), _grid(fractal_grid)
    first.materials.append(Material(len(first.materials), "extra"))
    model = _model(kind)
    model.calculate_properties(4, first)
    model.calculate_properties(8, second)
    reference_grid = _grid(fractal_grid)
    reference = _model(kind)
    reference.calculate_properties(8, reference_grid)
    assert model.matID == reference.matID
    assert _bin_properties(second, model.matID) == _bin_properties(reference_grid, reference.matID)


def _build(grid, kind, counts, rough, averaging, shared):
    boxes = []
    for index, (start, count) in enumerate(zip((2, 20), counts)):
        name = "mix" if shared else f"mix{index}"
        if not shared or index == 0:
            grid.mixingmodels.append(_model(kind, name))
        box = FractalBox(
            p1=(start * 0.001, 0.004, 0.004),
            p2=((start + 8) * 0.001, 0.012, 0.012),
            frac_dim=1.5,
            weighting=(1, 1, 1),
            n_materials=count,
            mixing_model_id=name,
            id=f"box{index}",
            seed=1,
            averaging=averaging,
        )
        box.build(grid)
        if rough:
            AddSurfaceRoughness(
                p1=(start * 0.001, 0.004, 0.012),
                p2=((start + 8) * 0.001, 0.012, 0.012),
                frac_dim=1.5,
                weighting=(1, 1),
                limits=(0.010, 0.014),
                fractal_box_id=f"box{index}",
                seed=2,
            ).build(grid)
        boxes.append(box)
    # Match Scene's two-pass lifecycle: prepare every volume before stamping any.
    for box in boxes:
        box.build(grid)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("counts", [(4, 8), (8, 4), (4, 4)])
@pytest.mark.parametrize("rough", [False, True])
@pytest.mark.parametrize("averaging", ["n", "y"])
def test_shared_model_matches_independent_definitions(fractal_grid, kind, counts, rough, averaging):
    shared, separate = _grid(fractal_grid), _grid(fractal_grid)
    _build(shared, kind, counts, rough, averaging, shared=True)
    _build(separate, kind, counts, rough, averaging, shared=False)
    for volume, reference in zip(shared.fractalvolumes, separate.fractalvolumes):
        assert len(volume.material_ids) == volume.nbins
        assert _bin_properties(shared, volume.material_ids) == _bin_properties(
            separate, reference.material_ids
        )
    # Compare physical properties rather than generated names or numeric IDs.
    for name in ("solid", "ID"):
        actual, expected = getattr(shared, name), getattr(separate, name)
        for actual_id, expected_id in np.unique(
            np.column_stack((actual.ravel(), expected.ravel())), axis=0
        ):
            assert _properties(shared.materials[actual_id]) == _properties(
                separate.materials[expected_id]
            )
    assert_array_equal(shared.rigidE, separate.rigidE)
    assert_array_equal(shared.rigidH, separate.rigidH)
