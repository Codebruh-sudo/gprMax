"""Symbolic coordinates belong to declarations, not their first built grid."""

from copy import deepcopy

import h5py
import numpy as np
import pytest

import gprMax
from gprMax.model import Model

pytestmark = pytest.mark.integration


def _scene(dl, mode, axis, objects):
    extent = [0.04] * 3
    extent[axis] = float("inf")
    scene = gprMax.Scene()
    for obj in (
        gprMax.DomainMode(mode),
        gprMax.Domain(tuple(extent)),
        gprMax.Discretisation((dl,) * 3),
        gprMax.TimeWindow(iterations=32),
        gprMax.OMPThreads(1),
        gprMax.PMLThickness(0),
        gprMax.Waveform(wave_type="ricker", amp=1, freq=5e9, id="pulse"),
        *objects,
    ):
        scene.add(obj)
    return scene


def _run(scene, path):
    gprMax.run(scenes=[scene], outputfile=path, cpu_precision="double", log_level=50, hide_progress_bars=True)


@pytest.mark.parametrize("axis", range(3), ids=list("xyz"))
@pytest.mark.parametrize("mode", ["TM", "TE"])
@pytest.mark.parametrize("family", ["hertzian", "magnetic", "hard", "resistive"])
def test_reused_source_and_receiver_match_fresh_grid(tmp_path, monkeypatch, axis, mode, family):
    point = [0.02] * 3
    point[axis] = float("inf")
    magnetic = family == "magnetic"
    polarisation = "xyz"[axis if (mode == "TE") == magnetic else (axis + 1) % 3]
    component = ("H" if magnetic else "E") + polarisation
    options = dict(p1=tuple(point), polarisation=polarisation, waveform_id="pulse")
    source = (
        gprMax.MagneticDipole(**options)
        if magnetic
        else gprMax.HertzianDipole(**options)
        if family == "hertzian"
        else gprMax.VoltageSource(**options, resistance=0 if family == "hard" else 50)
    )
    outputs = [component]
    receiver = gprMax.Rx(tuple(point), outputs=outputs)
    fresh = deepcopy((source, receiver))
    original = Model.build
    grids = []

    def capture(model):
        original(model)
        grids.append(model.G)

    monkeypatch.setattr(Model, "build", capture)
    for name, dl, objects in (
        ("coarse", 0.002, (source, receiver)),
        ("reused", 0.001, (source, receiver)),
        ("fresh", 0.001, fresh),
    ):
        _run(_scene(dl, mode, axis, objects), tmp_path / name)
        assert source.point == receiver.point == tuple(point)
        assert receiver.outputs == outputs == [component]
    with h5py.File(tmp_path / "reused.h5") as reused, h5py.File(tmp_path / "fresh.h5") as control:
        actual = reused[f"rxs/rx1/{component}"][:]
        np.testing.assert_array_equal(actual, control[f"rxs/rx1/{component}"][:])
        assert np.count_nonzero(actual) > 1
        assert reused["rxs/rx1"].attrs["Name"] == control["rxs/rx1"].attrs["Name"]
        np.testing.assert_array_equal(reused["rxs/rx1"].attrs["Position"], control["rxs/rx1"].attrs["Position"])
    assert grids[1].rxs[0].coord[axis] == (1 if mode == "TE" else 0)


@pytest.mark.parametrize("mode", ["TM", "TE"])
@pytest.mark.parametrize("kind", ["snapshot", "view", "geometry"])
def test_output_bounds_remain_symbolic_across_grid_changes(tmp_path, mode, kind):
    lower, upper = (0.01, 0.01, -float("inf")), (0.03, 0.03, float("inf"))
    if kind == "snapshot":
        output = gprMax.Snapshot(lower, upper, (0.002,) * 3, "frame", iterations=2, fileext=".h5")
    elif kind == "view":
        output = gprMax.GeometryView(lower, upper, (0.002,) * 3, "n", "view")
    else:
        output = gprMax.GeometryObjectsWrite(lower, upper, "geometry")
    for index, dl in enumerate((0.002, 0.001)):
        if kind != "geometry":
            # A normal view's stride must fit the single TM invariant cell.
            output.dl = (dl,) * 3
        _run(_scene(dl, mode, 2, [output]), tmp_path / f"model{index}")
        assert output.lower_bound == lower
        assert output.upper_bound == upper


def test_receiver_build_does_not_sort_callers_output_list(tmp_path):
    outputs = ["Hz", "Ex"]
    receiver = gprMax.Rx((0.02, 0.02, float("inf")), outputs=outputs)
    _run(_scene(0.002, "TE", 2, [receiver]), tmp_path / "model")
    assert outputs == receiver.outputs == ["Hz", "Ex"]
