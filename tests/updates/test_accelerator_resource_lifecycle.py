"""Snapshot transfer cost and context ownership across successful/failed runs."""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import gprMax
from gprMax import config
from gprMax.snapshots import Snapshot, update_snapshot_max_dims
from gprMax.updates.cuda_updates import CUDAUpdates
from gprMax.updates.opencl_updates import OpenCLUpdates


@pytest.mark.unit
@pytest.mark.parametrize("updates_class", [CUDAUpdates, OpenCLUpdates])
def test_retained_snapshot_histories_are_downloaded_once(monkeypatch, updates_class):
    monkeypatch.setattr(
        config, "get_model_config", lambda: SimpleNamespace(device={"snapsgpu2cpu": False})
    )
    updater = updates_class.__new__(updates_class)
    updater.grid = SimpleNamespace(
        rxs=[],
        transmissionlines=[],
        magneticfrillsources=[],
        snapshots=[SimpleNamespace(nx=2, ny=3, nz=4, snapfields={}) for _ in range(4)],
    )
    copied = []
    for index, component in enumerate(("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")):

        def get(index=index):
            copied.append(index)
            return np.arange(96).reshape(4, 2, 3, 4) + index

        setattr(updater, f"snap{component}_dev", SimpleNamespace(get=get))
    updater.finalise()
    assert copied == list(range(6))
    for index, snap in enumerate(updater.grid.snapshots):
        np.testing.assert_array_equal(
            snap.snapfields["Ex"], np.arange(96).reshape(4, 2, 3, 4)[index]
        )


@pytest.mark.unit
def test_cuda_cleanup_is_idempotent_and_respects_shared_contexts():
    calls = []
    context = SimpleNamespace(
        pop=lambda: calls.append("pop"), detach=lambda: calls.append("detach")
    )
    parent, child = CUDAUpdates.__new__(CUDAUpdates), CUDAUpdates.__new__(CUDAUpdates)
    parent.ctx = child.ctx = context
    parent._owns_context, child._owns_context = True, False
    child.cleanup()
    assert calls == []
    assert child.ctx is context
    parent.cleanup()
    parent.cleanup()
    assert calls == ["pop", "detach"]
    assert parent.ctx is None


@pytest.mark.unit
def test_updater_snapshot_shape_is_independent_of_another_grid():
    parent = CUDAUpdates.__new__(CUDAUpdates)
    parent.grid = SimpleNamespace(snapshots=[SimpleNamespace(nx=3, ny=4, nz=5)])
    assert parent.snapshot_shape == (3, 4, 5)
    update_snapshot_max_dims([SimpleNamespace(nx=8, ny=2, nz=2)])
    assert parent.snapshot_shape == (3, 4, 5)


def _scene(shape=(8, 2, 2), count=1):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(0.001,) * 3))
    scene.add(gprMax.Domain(p1=(0.018,) * 3))
    scene.add(gprMax.PMLThickness(thickness=2))
    scene.add(gprMax.TimeWindow(iterations=90))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=8e9, id="pulse"))
    scene.add(gprMax.HertzianDipole(p1=(0.009,) * 3, polarisation="z", waveform_id="pulse"))
    for index in range(count):
        scene.add(
            gprMax.Snapshot(
                p1=(0.003,) * 3,
                p2=tuple(0.003 + 0.001 * value for value in shape),
                dl=(0.001,) * 3,
                iterations=50 + 5 * index,
                filename=f"snap{index}",
                fileext=".h5",
            )
        )
    return scene


@pytest.mark.gpu
@pytest.mark.parametrize("stage", ["_set_macros", "update_magnetic", "finalise"])
def test_cuda_context_is_released_after_failure(tmp_path, monkeypatch, gpu_device, stage):
    import pycuda.driver as driver

    before = driver.Context.get_current()

    def fail(*args):
        raise RuntimeError("controlled lifecycle failure")

    with monkeypatch.context() as patch:
        patch.setattr(CUDAUpdates, stage, fail)
        with pytest.raises(RuntimeError, match="controlled lifecycle failure"):
            gprMax.run(
                scenes=[_scene()],
                gpu=[gpu_device],
                outputfile=tmp_path / "failed",
                hide_progress_bars=True,
                log_level=30,
            )
    assert driver.Context.get_current() == before
    gprMax.run(
        scenes=[_scene()],
        gpu=[gpu_device],
        outputfile=tmp_path / "recovery",
        hide_progress_bars=True,
        log_level=30,
    )
    assert driver.Context.get_current() == before
    assert (tmp_path / "recovery_snaps/snap0.h5").is_file()


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["cuda", "opencl"])
def test_sequential_snapshot_allocations_and_transfers(tmp_path, monkeypatch, request, backend):
    if backend == "cuda":
        device = request.getfixturevalue("gpu_device")
        import pycuda.gpuarray as gpuarray

        array_class, updates_class = gpuarray.GPUArray, CUDAUpdates
        options = {"gpu": [device]}
    else:
        device = request.getfixturevalue("opencl_device")
        import pyopencl.array as clarray

        array_class, updates_class = clarray.Array, OpenCLUpdates
        options = {"opencl": [device]}

    records = []
    original_finalise, original_get = updates_class.finalise, array_class.get
    targets, transfers = set(), []

    def get(array, *args, **kwargs):
        if id(array) in targets:
            transfers.append(array.nbytes)
        return original_get(array, *args, **kwargs)

    def finalise(updater):
        arrays = [
            getattr(updater, f"snap{component}_dev")
            for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        ]
        targets.update(id(array) for array in arrays)
        transfers.clear()
        original_finalise(updater)
        records.append(
            (arrays[0].shape, len(transfers), sum(transfers), sum(array.nbytes for array in arrays))
        )
        targets.clear()

    monkeypatch.setattr(array_class, "get", get)
    monkeypatch.setattr(updates_class, "finalise", finalise)
    for index, shape in enumerate(((8, 2, 2), (2, 8, 2), (2, 2, 8), (4, 4, 4))):
        count = 4 if index == 3 else 1
        gprMax.run(
            scenes=[_scene(shape, count)],
            outputfile=tmp_path / f"device{index}",
            hide_progress_bars=True,
            log_level=30,
            **options,
        )
        allocated, copies, transferred, size = records[-1]
        assert allocated == (count, *shape)
        assert copies == 6
        assert transferred == size
    gprMax.run(
        scenes=[_scene((4, 4, 4), 4)],
        cpu_precision="single",
        outputfile=tmp_path / "cpu",
        hide_progress_bars=True,
        log_level=30,
    )
    for index in range(4):
        with h5py.File(tmp_path / f"device3_snaps/snap{index}.h5") as actual, h5py.File(
            tmp_path / f"cpu_snaps/snap{index}.h5"
        ) as reference:
            for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
                values = reference[component][...]
                # The symmetry-zero Hz component contains only roundoff.
                # Scale absolute tolerance by the physical E or H vector,
                # not by that component's numerical noise.
                peak = max(
                    float(np.max(np.abs(reference[component[0] + axis][...]))) for axis in "xyz"
                )
                np.testing.assert_allclose(
                    actual[component][...], values, rtol=1e-5, atol=5e-6 * max(peak, 1e-20)
                )
