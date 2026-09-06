"""Snapshot launch bounds must provide one writer per allocated-grid index."""

import numpy as np
import pytest

from gprMax.cuda_opencl.knl_snapshots import store_snapshot
from gprMax.snapshots import Snapshot
from tests.updates.test_gpu_snapshot_indexing import _FakeSnap, _make_opencl_updates

pytestmark = pytest.mark.unit


def test_shared_snapshot_body_guards_before_decoding_without_modulo_or_return():
    body = store_snapshot["func"].substitute(
        REAL="float", NX_SNAPS=7, NY_SNAPS=9, NZ_SNAPS=5, CUDA_IDX=""
    )
    assert "if ((size_t)i < snapshot_volume)" in body
    assert body.index("if ((size_t)i < snapshot_volume)") < body.index("size_t rem_snaps")
    assert "size_t rem_snaps = (size_t)i;" in body
    assert "% snapshot_volume" not in body
    assert "return;" not in body


def test_opencl_snapshot_range_uses_maximum_pitches_for_uneven_snapshots(monkeypatch):
    first, second = _FakeSnap(0), _FakeSnap(0)
    first.nx, first.ny, first.nz = 7, 3, 2
    second.nx, second.ny, second.nz = 2, 9, 5
    updates, calls = _make_opencl_updates(monkeypatch, [first, second])
    updates.store_snapshots(0)
    assert len(calls) == 2
    for args, kwargs in calls:
        assert kwargs["range"] == slice(0, 7 * 9 * 5)
        # Individual snapshot volume is insufficient with max-shaped pitches.
        assert kwargs["range"].stop != int(np.prod(args[4:7]))


def test_opencl_snapshot_range_is_per_updater_not_mutable_class_maxima(monkeypatch):
    first, second = _FakeSnap(0), _FakeSnap(0)
    first.nx, first.ny, first.nz = 7, 9, 5
    second.nx, second.ny, second.nz = 2, 3, 4
    large, large_calls = _make_opencl_updates(monkeypatch, [first])
    small, small_calls = _make_opencl_updates(monkeypatch, [second])
    assert large.snapshot_shape == (7, 9, 5)
    for name in ("nx_max", "ny_max", "nz_max"):
        monkeypatch.setattr(Snapshot, name, 999)
    small.store_snapshots(0)
    large.store_snapshots(0)
    assert small_calls[0][1]["range"] == slice(0, 24)
    assert large_calls[0][1]["range"] == slice(0, 315)
