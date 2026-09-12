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

"""Actual-device equivalence of bounded and legacy full-ID PML dispatches."""

from math import prod
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import gprMax
from gprMax.updates.opencl_updates import OpenCLUpdates

pytestmark = [pytest.mark.integration, pytest.mark.gpu, pytest.mark.slow]

COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
PHIS = ("EPhi1", "EPhi2", "HPhi1", "HPhi2")
FACES = ("x0", "xmax", "y0", "ymax", "z0", "zmax")
DIRECTIONS = {"xminus", "xplus", "yminus", "yplus", "zminus", "zplus"}


@pytest.fixture
def opencl_runtime(opencl_device):
    """Reuse one context/queue for all paired solves in a parameter case."""
    import pyopencl as cl

    devices = [device for platform in cl.get_platforms() for device in platform.get_devices()]
    device = devices[opencl_device]
    context = cl.Context(devices=[device])
    queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)
    yield SimpleNamespace(dev=device, ctx=context, queue=queue)
    queue.finish()


def _scene(kind, formulation, order):
    scene = gprMax.Scene()
    scene.add(gprMax.OMPThreads(n=1))
    scene.add(gprMax.Discretisation(p1=(0.001,) * 3))
    is_2d = kind in ("TM", "TE")
    if is_2d:
        scene.add(gprMax.DomainMode(mode=kind))
    scene.add(gprMax.Domain(p1=(0.016, 0.019, float("inf") if is_2d else 0.023)))
    scene.add(gprMax.TimeWindow(iterations=4))
    thickness = (1, 2, 0, 2, 1, 0) if is_2d else (1, 2, 1, 2, 1, 2)
    scene.add(gprMax.PMLThickness(thickness=thickness if kind == "native" or is_2d else 0))
    scene.add(gprMax.PMLFormulation(formulation=formulation))
    for pole in range(order):
        scene.add(
            gprMax.PMLCFS(
                alphascalingprofile="constant",
                alphascalingdirection="forward",
                alphamin=0.2 if pole else 0.001,
                alphamax=0.2 if pole else 0.001,
                kappascalingprofile="linear",
                kappascalingdirection="forward",
                kappamin=1,
                kappamax=1.2 + 0.3 * pole,
                sigmascalingprofile="linear",
                sigmascalingdirection="forward",
                sigmamin=0.01,
                sigmamax=0.1 * (pole + 1),
            )
        )
    if kind == "internal":
        # Disjoint non-cubic boxes exercise unequal Phi1/Phi2 pitches and
        # the extra electric terminal plane for every slab direction.
        anchors = ((3, 3, 3), (9, 3, 3), (3, 9, 3), (9, 9, 3), (3, 3, 12), (9, 3, 12))
        for face, anchor in zip(FACES, anchors):
            scene.add(
                gprMax.PMLSlab(
                    p1=tuple(value * 0.001 for value in anchor),
                    p2=tuple((value + extent) * 0.001 for value, extent in zip(anchor, (2, 3, 4))),
                    maximum_face=face,
                    id=f"internal_{face}",
                )
            )
    elif kind == "replacement":
        scene.add(
            gprMax.PMLSlab(
                p1=(0, 0, 0), p2=(0.002, 0.019, 0.023), maximum_face="x0", id="replacement"
            )
        )
    scene.add(gprMax.Rx(p1=(0.008, 0.008, float("inf") if is_2d else 0.010)))
    return scene


def _run_seeded(
    tmp_path, monkeypatch, runtime, device, precision, formulation, order, kind, *, legacy
):
    """Run the actual solver; only seeds and the legacy range omission differ."""
    result = {"initial_phi": {}, "state": {}, "slabs": {}, "dispatches": 0}
    original_init = OpenCLUpdates.__init__
    original_finalise = OpenCLUpdates.finalise

    def initialise(updates, grid, shared=None):
        original_init(updates, grid, shared=runtime)
        rng = np.random.default_rng(92841)
        active = {"TM": {"Ez", "Hx", "Hy"}, "TE": {"Ex", "Ey", "Hz"}}.get(kind, set(COMPONENTS))
        for name in COMPONENTS:
            array = getattr(grid, f"{name}_dev")
            values = rng.uniform(-0.1, 0.1, array.shape).astype(array.dtype)
            if name not in active:
                values.fill(0)
            array.set(values, queue=runtime.queue)

        for pml in grid.pmls["slabs"]:
            assert len(pml.CFS) == order
            assert pml.formulation == formulation
            terminal = pml._updates_terminal_e_plane()
            assert terminal == (kind == "internal")
            assert pml.ERA.shape[1] == pml.thickness + int(terminal)
            assert pml.HRA.shape[1] == pml.thickness
            result["slabs"][pml.ID] = (pml.direction, terminal, pml.thickness)
            for name in PHIS:
                array = getattr(pml, f"{name}_dev")
                values = rng.uniform(-0.01, 0.01, array.shape).astype(array.dtype)
                assert np.all(values != 0)
                array.set(values, queue=runtime.queue)
                result["initial_phi"][f"phi/{pml.ID}/{name}"] = values

            for phase, prefix in (("electric", "E"), ("magnetic", "H")):
                spatial_size = max(
                    prod(getattr(pml, f"{prefix}Phi{number}").shape[1:]) for number in (1, 2)
                )
                expected = slice(0, spatial_size)
                assert getattr(pml, f"_{phase}_update_range") == expected
                assert spatial_size < grid.ID_dev.size
                kernel = getattr(pml, f"update_{phase}_dev")

                def dispatch(*args, _kernel=kernel, _expected=expected, **kwargs):
                    # The bounded side must use the new production launch
                    # limit. Restoring legacy only removes that keyword.
                    assert kwargs["range"] == _expected
                    if legacy:
                        kwargs.pop("range")
                    result["dispatches"] += 1
                    return _kernel(*args, **kwargs)

                setattr(pml, f"update_{phase}_dev", dispatch)

    def finalise(updates):
        original_finalise(updates)
        grid = updates.grid
        result["iterations"] = grid.iterations
        for name in COMPONENTS:
            result["state"][f"field/{name}"] = getattr(grid, f"{name}_dev").get()
        for pml in grid.pmls["slabs"]:
            for name in PHIS:
                result["state"][f"phi/{pml.ID}/{name}"] = getattr(pml, f"{name}_dev").get()

    output_path = tmp_path / f"{kind}_{'legacy' if legacy else 'bounded'}"
    with monkeypatch.context() as patch:
        patch.setattr(OpenCLUpdates, "__init__", initialise)
        patch.setattr(OpenCLUpdates, "finalise", finalise)
        gprMax.run(
            scenes=[_scene(kind, formulation, order)],
            outputfile=output_path,
            hide_progress_bars=True,
            opencl=[device],
            gpu_precision=precision,
        )
    with h5py.File(str(output_path) + ".h5", "r") as output:
        for name, dataset in output["rxs/rx1"].items():
            result["state"][f"receiver/{name}"] = dataset[:]
    return result


def _assert_pair(bounded, legacy, kind, order):
    assert bounded["slabs"] == legacy["slabs"]
    expected_directions = DIRECTIONS
    if kind in ("TM", "TE"):
        expected_directions = {"xminus", "xplus", "yminus", "yplus"}
    elif kind == "replacement":
        expected_directions = {"xminus"}
    assert {info[0] for info in bounded["slabs"].values()} == expected_directions
    assert (
        bounded["dispatches"]
        == legacy["dispatches"]
        == 2 * len(bounded["slabs"]) * bounded["iterations"]
    )
    assert bounded["state"].keys() == legacy["state"].keys()
    for name, values in bounded["state"].items():
        assert np.isfinite(values).all(), name
        reference = legacy["state"][name]
        np.testing.assert_array_equal(values, reference, err_msg=f"{kind}: {name}")
        assert values.dtype == reference.dtype
        assert values.tobytes() == reference.tobytes(), f"{kind}: {name} differs at the byte level"
    assert any(
        np.any(values != 0)
        for name, values in bounded["state"].items()
        if name.startswith("receiver/")
    )
    for name, before in bounded["initial_phi"].items():
        after = bounded["state"][name]
        assert before.shape[0] == order
        for pole in range(order):
            assert np.any(after[pole] != before[pole]), f"Unexercised {name} pole {pole}"
        _, slab_id, component = name.split("/")
        direction, terminal, thickness = bounded["slabs"][slab_id]
        if terminal and component.startswith("E"):
            axis = "xyz".index(direction[0]) + 1
            assert np.any(
                np.take(after, thickness, axis=axis) != np.take(before, thickness, axis=axis)
            ), name


@pytest.mark.parametrize("precision", ("single", "double"))
@pytest.mark.parametrize("formulation", ("HORIPML", "MRIPML"))
@pytest.mark.parametrize("order", (1, 2))
def test_opencl_pml_range_3d_matches_legacy(
    tmp_path, monkeypatch, opencl_runtime, opencl_device, precision, formulation, order
):
    for kind in ("native", "internal", "replacement"):
        options = (
            tmp_path,
            monkeypatch,
            opencl_runtime,
            opencl_device,
            precision,
            formulation,
            order,
            kind,
        )
        bounded = _run_seeded(*options, legacy=False)
        legacy = _run_seeded(*options, legacy=True)
        _assert_pair(bounded, legacy, kind, order)


@pytest.mark.parametrize("precision", ("single", "double"))
@pytest.mark.parametrize("formulation", ("HORIPML", "MRIPML"))
@pytest.mark.parametrize("order", (1, 2))
@pytest.mark.parametrize("mode", ("TM", "TE"))
def test_opencl_pml_range_2d_matches_legacy(
    tmp_path, monkeypatch, opencl_runtime, opencl_device, precision, formulation, order, mode
):
    options = (
        tmp_path,
        monkeypatch,
        opencl_runtime,
        opencl_device,
        precision,
        formulation,
        order,
        mode,
    )
    bounded = _run_seeded(*options, legacy=False)
    legacy = _run_seeded(*options, legacy=True)
    _assert_pair(bounded, legacy, mode, order)
