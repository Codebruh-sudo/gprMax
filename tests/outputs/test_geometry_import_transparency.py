"""Transparent component imports and wide global material-ID regressions."""

import h5py
import numpy as np
import pytest

from gprMax.cython.geometry_primitives import (
    build_edge_x,
    build_edge_y,
    build_edge_z,
    build_magnetic_edge_x,
    build_magnetic_edge_y,
    build_magnetic_edge_z,
    build_voxels_from_array,
)
from gprMax.geometry_outputs.geometry_objects_read import ReadGeometryObject
from gprMax.geometry_tags import GeometryTagMap, GeometryTagRegistry

pytestmark = pytest.mark.unit


def _write_file(path, data, component_ids=None, rigid_value=0):
    with h5py.File(path, "w") as output:
        output.attrs["dx_dy_dz"] = (0.001,) * 3
        output["data"] = data
        if component_ids is not None:
            output["ID"] = component_ids
            output["rigidE"] = np.full((12, *data.shape), rigid_value, np.int8)
            output["rigidH"] = np.full((6, *data.shape), rigid_value, np.int8)
            output["tag_data"] = np.ones(data.shape, np.uint8)
            output["tag_names"] = np.array(["untagged", "imported"], dtype=h5py.string_dtype())
    return path


def _initialise_prior_geometry(grid, rigid_value):
    grid.solid[:] = 7
    grid.ID[:] = 7
    grid.rigidE[:] = rigid_value
    grid.rigidH[:] = rigid_value
    registry = GeometryTagRegistry()
    registry.register_many(["prior", "imported"])
    registry.freeze()
    grid.geometry_tag_map = GeometryTagMap(tuple(grid.size), registry)
    grid.geometry_tag_map.data[:] = registry.id_for("prior")


def _read_complete(path, grid, start=(1, 2, 3), **kwargs):
    with ReadGeometryObject(path, grid, np.array(start, np.int32), np.array([9], np.int32), **kwargs) as reader:
        reader.read_data()
        reader.read_ID()
        reader.read_rigidE()
        reader.read_rigidH()
        reader.read_tags()


@pytest.mark.parametrize("component", range(6), ids=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"))
@pytest.mark.parametrize("rigid_value", [0, 1])
@pytest.mark.parametrize("write_cell", [False, True])
def test_partial_component_transparency_preserves_independent_rigid_claims(
    tmp_path, make_view_grid, component, rigid_value, write_cell
):
    """An edge's component mask, not /data, determines all of its rigid bits.

    Use the independent explicit-edge builders to identify the owning cells;
    this checks both neighbour claims and all 18 E/H rigidity planes.
    """
    grid = make_view_grid(nx=8, ny=9, nz=10)
    _initialise_prior_geometry(grid, 1 - rigid_value)
    expected = make_view_grid(nx=8, ny=9, nz=10)
    _initialise_prior_geometry(expected, 1 - rigid_value)
    data = np.full((4, 4, 4), -1, np.int16)
    if write_cell:
        data[0, 0, 0] = 0
        expected.solid[1, 2, 3] = 9
        expected.geometry_tag_map.data[1, 2, 3] = 2
    ids = np.full((6, 5, 5, 5), -1, np.int16)
    ids[component, 2, 2, 2] = 0
    position = (3, 4, 5)
    expected.ID[(component, *position)] = 9

    marker = make_view_grid(nx=8, ny=9, nz=10)
    marker.rigidE[:] = 0
    marker.rigidH[:] = 0
    if component < 3:
        (build_edge_x, build_edge_y, build_edge_z)[component](*position, 9, marker.rigidE, marker.rigidH, marker.ID)
    else:
        (build_magnetic_edge_x, build_magnetic_edge_y, build_magnetic_edge_z)[component - 3](
            *position, 9, marker.rigidH, marker.ID
        )
    for family in ("E", "H"):
        getattr(expected, f"rigid{family}")[getattr(marker, f"rigid{family}").astype(bool)] = rigid_value

    path = _write_file(tmp_path / "partial.h5", data, ids, rigid_value)
    _read_complete(path, grid)
    for name in ("solid", "ID", "rigidE", "rigidH"):
        np.testing.assert_array_equal(getattr(grid, name), getattr(expected, name), err_msg=name)
    np.testing.assert_array_equal(grid.geometry_tag_map.data, expected.geometry_tag_map.data)


@pytest.mark.parametrize("rigid_value", [0, 1])
def test_all_transparent_file_is_an_exact_noop(tmp_path, make_view_grid, rigid_value):
    grid = make_view_grid(nx=8, ny=9, nz=10)
    _initialise_prior_geometry(grid, 1 - rigid_value)
    before = {name: np.asarray(getattr(grid, name)).copy() for name in ("solid", "ID", "rigidE", "rigidH")}
    tags = grid.geometry_tag_map.data.copy()
    path = _write_file(
        tmp_path / "transparent.h5",
        np.full((4, 4, 4), -1, np.int16),
        np.full((6, 5, 5, 5), -1, np.int16),
        rigid_value,
    )
    _read_complete(path, grid)
    for name, array in before.items():
        np.testing.assert_array_equal(getattr(grid, name), array, err_msg=name)
    np.testing.assert_array_equal(grid.geometry_tag_map.data, tags)


@pytest.mark.parametrize("global_id", [32767, 32768, 40000, 65535, 65536])
@pytest.mark.parametrize("averaging", [False, True])
def test_compact_index_maps_to_wide_global_id_and_builds_voxel(tmp_path, make_view_grid, global_id, averaging):
    grid = make_view_grid()
    _initialise_prior_geometry(grid, 1)
    data = np.array([-1, 1], np.int16).reshape(2, 1, 1)
    path = _write_file(tmp_path / "compact.h5", data)
    with ReadGeometryObject(path, grid, np.ones(3, np.int32), np.array([0, global_id], np.int32)) as reader:
        mapped = reader.get_data()
        np.testing.assert_array_equal(mapped[:, 0, 0], [-1, global_id])
        assert mapped.dtype == np.int32
        assert mapped.flags.c_contiguous
        build_voxels_from_array(
            *reader.get_local_data_start(),
            0,
            averaging,
            np.zeros(global_id + 1, np.uint8),
            np.ones(global_id + 1, np.uint8),
            mapped,
            grid.solid,
            grid.rigidE,
            grid.rigidH,
            grid.ID,
        )
    assert grid.solid[1, 1, 1] == 7  # transparent cell is untouched
    assert grid.solid[2, 1, 1] == global_id
    assert np.all(grid.rigidE[:, 1, 1, 1] == 1)
    assert np.all(grid.rigidH[:, 1, 1, 1] == 1)
    if not averaging:
        np.testing.assert_array_equal(grid.ID[:, 2, 1, 1], np.full(6, global_id))


@pytest.mark.parametrize("method", ["get_data", "read_data", "read_ID"])
def test_unsigned_positive_file_index_is_not_reinterpreted_as_negative(tmp_path, make_view_grid, method):
    grid = make_view_grid()
    data = np.full((2, 2, 2), 40000, np.uint16)
    ids = np.full((6, 3, 3, 3), 40000, np.uint16)
    path = _write_file(tmp_path / "unsigned.h5", data, ids)
    mapping = np.full(40001, 65536, np.int32)
    with ReadGeometryObject(path, grid, np.ones(3, np.int32), mapping) as reader:
        result = getattr(reader, method)()
    if method == "get_data":
        assert np.all(result == 65536)
    else:
        target = grid.solid if method == "read_data" else grid.ID
        assert np.all(target[(..., slice(1, 3), slice(1, 3), slice(1, 3))] == 65536)


@pytest.mark.parametrize("method", ["get_data", "read_data", "read_ID"])
@pytest.mark.parametrize(
    "value,dtype,message",
    [
        (-2, np.int16, "must be -1"),
        (2, np.int16, "references material index 2"),
        (65535, np.uint16, "references material index 65535"),
        (0.5, np.float64, "must be integers"),
    ],
)
def test_invalid_file_indices_raise_before_conversion(tmp_path, make_view_grid, method, value, dtype, message):
    path = _write_file(
        tmp_path / "invalid.h5",
        np.full((2, 2, 2), value, dtype),
        np.full((6, 3, 3, 3), value, dtype),
    )
    grid = make_view_grid()
    before = grid.solid.copy(), grid.ID.copy()
    with ReadGeometryObject(path, grid, np.ones(3, np.int32), np.array([0, 1], np.int32)) as reader:
        with pytest.raises(ValueError, match=message):
            getattr(reader, method)()
    np.testing.assert_array_equal(grid.solid, before[0])
    np.testing.assert_array_equal(grid.ID, before[1])


@pytest.mark.parametrize("method", ["get_data", "read_data", "read_ID"])
def test_empty_material_map_accepts_only_transparent_data(tmp_path, make_view_grid, method):
    path = _write_file(
        tmp_path / "empty.h5",
        np.full((2, 2, 2), -1, np.int16),
        np.full((6, 3, 3, 3), -1, np.int16),
    )
    grid = make_view_grid()
    before = grid.solid.copy(), grid.ID.copy()
    with ReadGeometryObject(path, grid, np.ones(3, np.int32), np.array([], np.int32)) as reader:
        result = getattr(reader, method)()
    if method == "get_data":
        assert np.all(result == -1)
    np.testing.assert_array_equal(grid.solid, before[0])
    np.testing.assert_array_equal(grid.ID, before[1])


@pytest.mark.parametrize("mapping", [np.array([-1]), np.array([2**31], np.int64), np.array([1.5]), np.array([[1]])])
def test_invalid_global_id_maps_raise_without_narrowing(tmp_path, make_view_grid, mapping):
    path = _write_file(tmp_path / "map.h5", np.zeros((1, 1, 1), np.int16))
    with pytest.raises(ValueError, match="Geometry material ID map"):
        ReadGeometryObject(path, make_view_grid(), np.ones(3, np.int32), mapping)


@pytest.mark.parametrize("axis", range(3), ids=("x", "y", "z"))
@pytest.mark.parametrize("source_size,target_size", [(1, 2), (2, 1)])
def test_component_transparency_follows_2d_canonical_edge(tmp_path, make_view_grid, axis, source_size, target_size):
    shape = [4, 4, 4]
    shape[axis] = source_size
    data = np.full(shape, -1, np.int16)
    ids = np.full((6, *(size + 1 for size in shape)), -1, np.int16)
    canonical = int(source_size == 2)
    # Distinct masks on the discarded boundary and canonical edge ensure
    # cell resizing cannot accidentally be substituted for ID resizing.
    selected = [slice(None)] * 4
    selected[axis + 1] = canonical
    ids[tuple(selected)] = 0
    ids[(0, *([0] * 3))] = -1
    path = _write_file(tmp_path / "source.h5", data, ids)
    expanded_shape = shape.copy()
    expanded_shape[axis] = target_size
    expected_data = np.full(expanded_shape, -1, np.int16)
    expected_ids = np.repeat(np.take(ids, [canonical], axis=axis + 1), target_size + 1, axis=axis + 1)
    explicit = _write_file(tmp_path / "explicit.h5", expected_data, expected_ids)
    grid, expected = make_view_grid(), make_view_grid()
    _initialise_prior_geometry(grid, 1)
    _initialise_prior_geometry(expected, 1)
    _read_complete(path, grid, start=(1, 1, 1), invariant_axis=axis, target_invariant_size=target_size)
    _read_complete(explicit, expected, start=(1, 1, 1))
    for name in ("solid", "ID", "rigidE", "rigidH"):
        np.testing.assert_array_equal(getattr(grid, name), getattr(expected, name), err_msg=name)
    np.testing.assert_array_equal(grid.geometry_tag_map.data, expected.geometry_tag_map.data)


@pytest.mark.parametrize("axis", range(3), ids=("x", "y", "z"))
def test_transparency_includes_nonleading_mpi_negative_halo(tmp_path, make_view_grid, make_mpi_grid, axis):
    """A real COMM_SELF exercises the nonleading rank's reader slices.

    Rank-local index zero is a negative interface halo here; compare it and
    every owned value against the same physical region of a serial import.
    """
    shape = [4, 4, 4]
    shape[axis] = 8
    indices = np.indices(shape)
    data = np.where(indices.sum(axis=0) % 3, -1, 0).astype(np.int16)
    edge_shape = (6, *(size + 1 for size in shape))
    ids = np.where(np.indices(edge_shape).sum(axis=0) % 3, -1, 0).astype(np.int16)
    path = _write_file(tmp_path / "halo.h5", data, ids)
    serial = make_view_grid(nx=16, ny=16, nz=16)
    _initialise_prior_geometry(serial, 1)
    start = np.ones(3, np.int32)
    start[axis] = 2
    _read_complete(path, serial, start=start)
    local = make_view_grid(nx=6, ny=6, nz=6)
    _initialise_prior_geometry(local, 1)
    origin = np.zeros(3, np.int32)
    origin[axis] = 4
    halo = np.zeros(3, np.int32)
    halo[axis] = 1
    grid = make_mpi_grid(
        size=(6, 6, 6),
        negative_halo_offset=halo,
        origin=origin,
        arrays={name: getattr(local, name) for name in ("solid", "ID", "rigidE", "rigidH", "geometry_tag_map")},
    )
    _read_complete(path, grid, start=start - origin)
    for name in ("solid", "ID", "rigidE", "rigidH"):
        size = 7 if name == "ID" else 6
        region = tuple(slice(int(offset), int(offset) + size) for offset in origin)
        np.testing.assert_array_equal(getattr(grid, name), getattr(serial, name)[(..., *region)], err_msg=name)
    region = tuple(slice(int(offset), int(offset) + 6) for offset in origin)
    np.testing.assert_array_equal(grid.geometry_tag_map.data, serial.geometry_tag_map.data[region])
