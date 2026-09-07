"""Reduced-mode MPI payload contracts and collective error handling."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from gprMax.grid.mpi_grid import MPIGrid
from gprMax.sar import SARLocalPayload, SARMonitor, edge_offsets_for_mode

pytestmark = pytest.mark.unit


def _payload(mode, rank, density=True):
    cell = np.asarray([[3, 3, 3]], dtype=np.int32)
    coordinates, dft = {}, {}
    for component, offsets in edge_offsets_for_mode(mode).items():
        edges = np.unique((cell[:, None, :] + offsets).reshape(-1, 3), axis=0)
        owned = np.arange(len(edges)) % 2 == rank
        coordinates[component] = edges[owned]
        dft[component] = np.full((2, owned.sum()), 2 + 1j, dtype=np.complex128)
    count = int(rank == 0)
    return SARLocalPayload(
        cell[:count],
        np.ones(count, dtype=np.uint8),
        np.ones(count, dtype=np.uint32),
        np.full(count if density else 0, 1000.0),
        np.zeros((2, count)),
        rank,
        coordinates,
        dft,
    )


@pytest.mark.parametrize(
    "mode", ["3D", "2D TMx", "2D TMy", "2D TMz", "2D TEx", "2D TEy", "2D TEz"]
)
@pytest.mark.parametrize("density", [True, False], ids=["sar", "radiometry"])
def test_merge_and_collocation_use_only_active_components(monkeypatch, mode, density):
    monitor = SARMonitor.__new__(SARMonitor)
    monitor.edge_offsets = edge_offsets_for_mode(mode)
    monitor.frequencies = np.asarray([1e9, 2e9])
    monitor.real_dtype = np.dtype(np.float64)
    monitor.grid = SimpleNamespace()
    monkeypatch.setattr(
        "gprMax.sar._material_loss_conductivity", lambda *args, **kwargs: np.ones((2, 1))
    )
    merged = monitor.merge_local_payloads([_payload(mode, rank, density) for rank in range(2)])
    assert tuple(merged.edge_coordinates) == tuple(monitor.edge_offsets)
    result = monitor._collocate_mpi_payload(merged, (8, 8, 8))
    np.testing.assert_allclose(result.absorbed_power_density, 2.5 * len(monitor.edge_offsets))
    assert result.density.shape == ((1,) if density else (0,))
    assert result.excluded_pml_cell_count == 1


@pytest.mark.parametrize("failure", ["keys", "dft_keys", "dft_shape", "coordinates", "metadata"])
def test_merge_rejects_inconsistent_rank_payloads(failure):
    left, right = [_payload("2D TMz", rank) for rank in range(2)]
    if failure == "keys":
        right = replace(right, edge_coordinates={"Ex": right.edge_coordinates["Ez"]})
    elif failure == "dft_keys":
        right = replace(right, edge_dft={})
    elif failure == "dft_shape":
        right = replace(right, edge_dft={"Ez": np.zeros((3, 2))})
    elif failure == "coordinates":
        right = replace(right, edge_coordinates={"Ez": np.zeros((2, 2), dtype=np.int32)})
    else:
        left = replace(left, density=np.empty(2))
    with pytest.raises(RuntimeError, match="inconsistent"):
        SARMonitor.merge_local_payloads([left, right])


def test_gather_reports_missing_monitor_collectively():
    grid = SimpleNamespace(
        sar_monitors=[], comm=SimpleNamespace(allgather=lambda value: [value, ("missing",)])
    )
    with pytest.raises(RuntimeError, match="inconsistent SAR monitors"):
        MPIGrid.gather_sar_payloads(grid)


def test_gather_reports_local_preparation_failure_before_payload_gather():
    calls = []

    def allgather(value):
        calls.append(value)
        return [value, value]

    def fail():
        raise ValueError("incomplete sample count")

    monitor = SimpleNamespace(mpi_signature=lambda: ("same",), local_payload=fail)
    grid = SimpleNamespace(sar_monitors=[monitor], comm=SimpleNamespace(allgather=allgather))
    with pytest.raises(RuntimeError, match="incomplete sample count"):
        MPIGrid.gather_sar_payloads(grid)
    assert len(calls) == 2


def test_gather_reports_remote_preparation_failure_before_payload_gather():
    def allgather(value):
        return [value, "ValueError: remote failure"] if value is None else [value, value]

    monitor = SimpleNamespace(mpi_signature=lambda: ("same",), local_payload=lambda: object())
    grid = SimpleNamespace(sar_monitors=[monitor], comm=SimpleNamespace(allgather=allgather))
    with pytest.raises(RuntimeError, match="remote failure"):
        MPIGrid.gather_sar_payloads(grid)
