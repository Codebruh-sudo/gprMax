"""Coordinate-faithful coarse snapshot sampling on native Yee fields."""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from gprMax import config
from gprMax.mode2d import mode2d_geometry
from gprMax.snapshots import Snapshot, YEE_OFFSETS

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("step", [(1, 1, 1), (2, 2, 2), (3, 3, 3), (2, 3, 4), (3, 2, 1)])
@pytest.mark.parametrize("mode", ["3D", "2D TMx", "2D TMy", "2D TMz", "2D TEx", "2D TEy", "2D TEz"])
def test_affine_yee_fields_match_written_coarse_cell_centres(
    make_view_grid, monkeypatch, tmp_path, dtype, step, mode
):
    monkeypatch.setitem(config.sim_config.dtypes, "float_or_double", dtype)
    monkeypatch.setattr(config.get_model_config(), "mode", mode)
    grid = make_view_grid(nx=14, ny=13, nz=12, fill=False)
    grid.dl = np.asarray((0.001, 0.002, 0.003))
    start, stop, step = np.asarray((1, 2, 1)), np.asarray((10, 11, 10)), np.asarray(step)
    geometry = mode2d_geometry(mode)
    active = tuple(YEE_OFFSETS)
    if geometry is not None:
        axis = geometry.invariant_axis
        start[axis], stop[axis], step[axis] = geometry.live_index, geometry.live_index + 1, 1
        active = geometry.active_electric + geometry.active_magnetic
    indices = np.indices(tuple(grid.size + 1))
    for component, offsets in YEE_OFFSETS.items():
        values = sum((indices[a] + offsets[a] / 2) * grid.dl[a] * (a + 1) for a in range(3))
        if geometry is not None:
            mask = indices[geometry.invariant_axis] == geometry.live_index
            values = np.where(mask, values, 0)
        setattr(grid, component, np.ascontiguousarray(values, dtype=dtype))
    snap = Snapshot(
        *start,
        *stop,
        *step,
        4,
        str(tmp_path / "affine"),
        ".h5",
        {name: name in active for name in YEE_OFFSETS},
        grid
    )
    snap.initialise_snapfields()
    snap.store()
    bar = SimpleNamespace(update=lambda **kwargs: None)
    snap.write_hdf5(bar)
    with h5py.File(snap.filename) as output:
        centre = np.indices(tuple(output.attrs["nx_ny_nz"])) + 0.5
        expected = sum(
            (output.attrs["origin"][a] + centre[a] * output.attrs["dx_dy_dz"][a]) * (a + 1)
            for a in range(3)
        )
        for component in active:
            np.testing.assert_allclose(output[component][...], expected, rtol=2e-7, atol=1e-9)
    snap.filename = snap.filename.with_suffix(".vtkhdf")
    snap.write_vtk(bar)
    with h5py.File(snap.filename) as output:
        root = output["VTKHDF"]
        np.testing.assert_allclose(root.attrs["Origin"], snap._physical_origin())
        np.testing.assert_allclose(root.attrs["Spacing"], step * grid.dl)
        for component in active:
            np.testing.assert_allclose(
                root["CellData"][component][...], expected.transpose(2, 1, 0), rtol=2e-7, atol=1e-9
            )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_stride_one_preserves_original_arithmetic_bitwise(make_view_grid, monkeypatch, dtype):
    monkeypatch.setitem(config.sim_config.dtypes, "float_or_double", dtype)
    grid = make_view_grid(nx=6, ny=7, nz=8, fill=False)
    rng = np.random.default_rng(929)
    for name in YEE_OFFSETS:
        setattr(grid, name, rng.normal(size=tuple(grid.size + 1)).astype(dtype))
    snap = Snapshot(
        1, 1, 1, 5, 6, 7, 1, 1, 1, 0, "unused", ".h5", dict.fromkeys(YEE_OFFSETS, True), grid
    )
    snap.initialise_snapfields()
    snap.store()
    stencils = {
        "Ex": ((0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1)),
        "Ey": ((0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1)),
        "Ez": ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)),
        "Hx": ((0, 0, 0), (1, 0, 0)),
        "Hy": ((0, 0, 0), (0, 1, 0)),
        "Hz": ((0, 0, 0), (0, 0, 1)),
    }
    for name, stencil in stencils.items():
        slices = [
            getattr(grid, name)[
                tuple(slice(1 + v, 1 + v + n) for v, n in zip(offset, snap.grid_view.size))
            ]
            for offset in stencil
        ]
        expected = slices[0].copy()
        for values in slices[1:]:
            expected += values
        expected /= len(slices)
        assert snap.snapfields[name].tobytes() == expected.tobytes()


@pytest.mark.parametrize("stop,step", [((10, 10, 10), (3, 3, 3)), ((9, 10, 10), (4, 2, 2))])
def test_unsupported_partial_final_cell_fails_during_construction(make_view_grid, stop, step):
    grid = make_view_grid(nx=10, ny=10, nz=10)
    with pytest.raises(ValueError, match="outside native Yee support"):
        Snapshot(0, 0, 0, *stop, *step, 0, "unused", ".h5", dict.fromkeys(YEE_OFFSETS, True), grid)


def test_valid_nondividing_interior_roi_retains_requested_shape_and_spacing(make_view_grid):
    grid = make_view_grid(nx=12, ny=12, nz=12)
    snap = Snapshot(
        0, 0, 0, 10, 10, 10, 3, 3, 3, 0, "unused", ".h5", dict.fromkeys(YEE_OFFSETS, True), grid
    )
    np.testing.assert_array_equal(snap.grid_view.stop, (10, 10, 10))
    np.testing.assert_array_equal(snap.grid_view.size, (4, 4, 4))
    np.testing.assert_array_equal(snap.grid_view.step, (3, 3, 3))


def test_cropped_native_roi_inputs_share_grid_storage_with_selected_output(
    make_view_grid, monkeypatch
):
    grid = make_view_grid(nx=80, ny=90, nz=100, fill=False)
    snap = Snapshot(
        2,
        3,
        4,
        74,
        83,
        92,
        8,
        7,
        6,
        0,
        "unused",
        ".h5",
        {name: name == "Ez" for name in YEE_OFFSETS},
        grid,
    )
    snap.initialise_snapfields()
    calls = []

    def inspect(*args):
        for name, native in zip(YEE_OFFSETS, args[10:16]):
            assert np.shares_memory(native, getattr(grid, name))
            assert native.strides[-1] == native.dtype.itemsize
            assert not native.flags.c_contiguous
        calls.append(args)

    monkeypatch.setattr("gprMax.snapshots.calculate_snapshot_fields", inspect)
    snap.store()
    assert len(calls) == 1
    assert snap.snapfields["Ex"].shape == (1, 1, 1)
