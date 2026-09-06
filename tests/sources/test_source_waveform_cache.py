"""Window/configuration-aware cache reuse for all four sampled source families."""

from copy import copy
from types import SimpleNamespace

import numpy as np
import pytest

from gprMax import config
from gprMax.sources import HertzianDipole, MagneticDipole, TransmissionLine, VoltageSource
from gprMax.waveforms import Waveform

pytestmark = pytest.mark.unit

FAMILIES = (
    (VoltageSource, "voltagesources"),
    (HertzianDipole, "hertziandipoles"),
    (MagneticDipole, "magneticdipoles"),
    (TransmissionLine, "transmissionlines"),
)
NAMES = ("waveformvalues_halfdt", "waveformvalues_wholedt")
WINDOWS = {
    "start_only": ((0.375, 2), (0, 2)),
    "stop_only": ((0, 1.5), (0, 2)),
    "both": ((0.375, 1.5), (0, 2)),
    "same_custom": ((0.375, 1.5), (0.375, 1.5)),
    "full": ((0, 2), (0, 2)),
}


def _grid(dtype):
    config.sim_config.dtypes["float_or_double"] = dtype
    waveform = Waveform()
    waveform.ID, waveform.type, waveform.freq = "w", "gaussian", 0.75
    duplicate = copy(waveform)
    duplicate.ID = "independent"
    return SimpleNamespace(
        dt=0.125,
        iterations=16,
        timewindow=2,
        waveforms=[waveform, duplicate],
        voltagesources=[],
        hertziandipoles=[],
        magneticdipoles=[],
        transmissionlines=[],
    )


def _source(cls, grid, window, waveform="w"):
    source = cls(grid.iterations, grid.dt) if cls is TransmissionLine else cls()
    source.start, source.stop = window
    source.waveformID = waveform
    return source


def _assert_independent(source, cls, grid, collection):
    reference = _source(cls, grid, (source.start, source.stop), "independent")
    # Fresh local cache models construction on a different MPI partition.
    local = copy(grid)
    setattr(local, collection, [])
    reference.calculate_waveform_values(local)
    for name in NAMES:
        actual = getattr(source, name)
        if actual is not None:
            expected = getattr(reference, name)
            assert actual.dtype == expected.dtype
            assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("family", FAMILIES, ids=lambda pair: pair[0].__name__)
@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["single", "double"])
@pytest.mark.parametrize("reverse", [False, True], ids=["custom_first", "full_first"])
@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("separate_ids", [False, True], ids=["shared_id", "separate_ids"])
def test_histories_match_independent_cache_in_both_orders(family, dtype, reverse, window, separate_ids):
    cls, collection = family
    grid = _grid(dtype)
    windows = WINDOWS[window]
    objects = [
        _source(cls, grid, bounds, "independent" if separate_ids and i else "w") for i, bounds in enumerate(windows)
    ]
    for source in reversed(objects) if reverse else objects:
        source.calculate_waveform_values(grid)
        getattr(grid, collection).append(source)
        _assert_independent(source, cls, grid, collection)
    if not separate_ids and windows[0] == windows[1]:
        for name in NAMES:
            assert getattr(objects[0], name) is getattr(objects[1], name)
    else:
        for name in NAMES:
            if getattr(objects[0], name) is not None:
                assert getattr(objects[0], name) is not getattr(objects[1], name)


@pytest.mark.parametrize("family", FAMILIES, ids=lambda pair: pair[0].__name__)
@pytest.mark.parametrize("change", ["dt", "iterations", "dtype", "amplitude", "frequency", "resampled"])
def test_stale_configuration_or_resampled_history_is_not_a_donor(family, change):
    cls, collection = family
    grid = _grid(np.float64)
    donor = _source(cls, grid, (0, 2))
    donor.calculate_waveform_values(grid)
    getattr(grid, collection).append(donor)
    if change == "dt":
        grid.dt *= 0.8
    elif change == "iterations":
        grid.iterations += 2
    elif change == "dtype":
        config.sim_config.dtypes["float_or_double"] = np.float32
    elif change == "amplitude":
        for waveform in grid.waveforms:
            waveform.amp *= 0.5
    elif change == "frequency":
        for waveform in grid.waveforms:
            waveform.freq *= 1.5
    else:
        # Normal Study resampling replaces histories, including scaled drives.
        for name in NAMES:
            if getattr(donor, name) is not None:
                setattr(donor, name, getattr(donor, name) * 0.5)
    new = _source(cls, grid, (0, 2))
    new.calculate_waveform_values(grid)
    _assert_independent(new, cls, grid, collection)
    for name in NAMES:
        if getattr(new, name) is not None:
            assert getattr(new, name) is not getattr(donor, name)


@pytest.mark.parametrize("family", FAMILIES, ids=lambda pair: pair[0].__name__)
def test_self_and_uninitialised_sources_cannot_supply_a_cache(family):
    cls, collection = family
    grid = _grid(np.float64)
    source = _source(cls, grid, (0, 2))
    setattr(grid, collection, [source, _source(cls, grid, (0, 2))])
    source.calculate_waveform_values(grid)
    _assert_independent(source, cls, grid, collection)
