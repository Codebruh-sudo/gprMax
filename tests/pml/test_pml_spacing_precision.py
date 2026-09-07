"""Check spacing precision in every compiled CPU PML update variant."""

from importlib import import_module

import numpy as np
import pytest
from numpy.testing import assert_array_equal

pytestmark = pytest.mark.unit
FORMULATIONS = ("HORIPML", "MRIPML")
POLARITIES = ("electric", "magnetic")
DIRECTIONS = ("xminus", "xplus", "yminus", "yplus", "zminus", "zplus")


def _update(formulation, polarity, order, direction, dtype, spacing):
    """One correction with zero target fields/history and fixed curl inputs.

    Holding all coefficients fixed isolates the spacing argument from profile
    construction. The resulting target fields and histories must scale as 1/d.
    Bounds lie inside the array for both forward and backward differences.
    """
    shape = (7, 7, 7)
    rng = np.random.default_rng(20260907)
    fields = [np.zeros(shape, dtype=dtype) for _ in range(6)]
    input_components = range(3, 6) if polarity == "electric" else range(3)
    for index in input_components:
        fields[index][...] = rng.standard_normal(shape)
    histories = [np.zeros((order, 2, 2, 2), dtype=dtype) for _ in range(2)]
    coefficients = [np.full((order, 2), value, dtype=dtype) for value in (0.8, 0.3, 0.7, 0.2)]
    material_coefficients = np.ones((1, 5), dtype=dtype)
    material_ids = np.zeros((6, *shape), dtype=np.uint32)
    module = import_module(f"gprMax.cython.pml_updates_{polarity}_{formulation}")
    kernel = getattr(module, f"order{order}_{direction}")
    kernel(
        2,
        4,
        2,
        4,
        2,
        4,
        1,
        material_coefficients,
        material_ids,
        *fields,
        *histories,
        *coefficients,
        spacing,
    )
    targets = fields[:3] if polarity == "electric" else fields[3:]
    return [*targets, *histories]


@pytest.mark.parametrize("formulation", FORMULATIONS)
@pytest.mark.parametrize("polarity", POLARITIES)
@pytest.mark.parametrize("order", (1, 2))
@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_spacing_uses_field_precision(formulation, polarity, order, direction, dtype):
    # Distinct in double precision but deliberately indistinguishable in float32.
    first_spacing = 0.001
    second_spacing = first_spacing * (1 + 1e-8)
    assert np.float32(first_spacing) == np.float32(second_spacing)
    first = _update(formulation, polarity, order, direction, dtype, first_spacing)
    second = _update(formulation, polarity, order, direction, dtype, second_spacing)
    assert max(np.max(np.abs(values)) for values in first) > 0
    assert all(np.max(np.abs(values)) > 0 for values in first[3:])
    for actual, changed in zip(first, second):
        assert np.isfinite(actual).all() and np.isfinite(changed).all()
        if dtype == np.float32:
            # Preserve the existing single-precision spacing conversion.
            assert_array_equal(actual, changed)
        else:
            scale = max(np.max(np.abs(actual)), 1e-30)
            # Zero histories make this correction homogeneous in 1/d, for both
            # formulations/orders. This expectation does not repeat their algebra.
            expected = actual * (first_spacing / second_spacing)
            assert np.max(np.abs(changed - expected)) / scale < 2e-14
