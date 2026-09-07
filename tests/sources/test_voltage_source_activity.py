"""Shared accelerator hard-source activity packing and template contracts."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from gprMax import config
from gprMax.cuda_opencl.knl_source_updates import update_voltage_source
from gprMax.sources import HertzianDipole, MagneticDipole, VoltageSource, htod_src_arrays

pytestmark = pytest.mark.unit


def _backend(monkeypatch, backend, dtype):
    config.sim_config.general["solver"] = backend
    config.sim_config.dtypes["float_or_double"] = dtype
    if backend == "metal":

        class Device:
            def newBufferWithBytes_length_options_(self, data, nbytes, options):
                assert len(data) == nbytes
                assert options == 0
                return data

        monkeypatch.setattr(
            config, "get_model_config", lambda: SimpleNamespace(device={"dev": Device()})
        )
    else:
        package = "pycuda" if backend == "cuda" else "pyopencl"
        name = "gpuarray" if backend == "cuda" else "array"
        module = ModuleType(f"{package}.{name}")
        module.to_gpu = lambda values: values.copy()
        module.to_device = lambda queue, values: values.copy()
        parent = ModuleType(package)
        setattr(parent, name, module)
        monkeypatch.setitem(sys.modules, package, parent)
        monkeypatch.setitem(sys.modules, f"{package}.{name}", module)


def _host(value, dtype):
    return np.frombuffer(value, dtype=dtype) if isinstance(value, bytes) else value.ravel()


@pytest.mark.parametrize("backend", ["cuda", "opencl", "metal"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["single", "double"])
@pytest.mark.parametrize("polarisation", ["x", "y", "z"])
def test_voltage_tail_preserves_rows_and_exact_inclusive_host_times(
    monkeypatch, backend, dtype, polarisation
):
    _backend(monkeypatch, backend, dtype)
    grid = SimpleNamespace(iterations=10, dt=0.1)
    exact = 3 * grid.dt
    windows = [
        (0, 1),
        (exact, 0.7),
        (np.nextafter(exact, np.inf), 0.7),
        (0, np.nextafter(exact, -np.inf)),
        (exact, exact),
        (0.31, 0.32),
        (2, 3),
        (0, 2),
    ]
    sources = []
    for index, (start, stop) in enumerate(windows):
        source = VoltageSource()
        source.coord[:] = (index + 1, index + 2, index + 3)
        source.polarisation = polarisation
        source.start, source.stop = start, stop
        source.resistance = 0 if index % 2 == 0 else 50
        source.waveformvalues_wholedt = np.arange(11, dtype=dtype) + 20 * index
        source.waveformvalues_halfdt = source.waveformvalues_wholedt + 0.5
        sources.append(source)
    info, resistance, waveforms = htod_src_arrays(sources, grid)
    info, resistance, waveforms = (
        _host(info, np.int32),
        _host(resistance, dtype),
        _host(waveforms, dtype),
    )
    assert info.size == 6 * len(sources)
    coords = info[: 4 * len(sources)].reshape(-1, 4)
    activity = info[4 * len(sources) :].reshape(-1, 2)
    for index, source in enumerate(sources):
        np.testing.assert_array_equal(coords[index], [*source.coord, "xyz".index(polarisation)])
        first, last = activity[index]
        for iteration in range(grid.iterations + 1):
            assert (first <= iteration <= last) == (
                source.start <= iteration * grid.dt <= source.stop
            )
        assert resistance[index] == source.resistance
        expected = (
            source.waveformvalues_halfdt if source.resistance else source.waveformvalues_wholedt
        )
        np.testing.assert_array_equal(waveforms.reshape(-1, 11)[index], expected)
    # A later study upload must use the current window, not a cached interval.
    sources[0].start, sources[0].stop = exact, 0.5
    updated, _, _ = htod_src_arrays(sources, grid)
    np.testing.assert_array_equal(_host(updated, np.int32)[4 * len(sources) :][:2], [3, 5])


@pytest.mark.parametrize("backend", ["cuda", "opencl", "metal"])
@pytest.mark.parametrize("family", [HertzianDipole, MagneticDipole])
def test_non_voltage_source_layout_is_unchanged(monkeypatch, backend, family):
    _backend(monkeypatch, backend, np.float64)
    grid = SimpleNamespace(iterations=3)
    sources = []
    for index in range(3):
        source = family()
        source.coord[:], source.polarisation = (index, index + 1, index + 2), "xyz"[index]
        source.dl = 0.001 * (index + 1)
        source.waveformvalues_halfdt = np.arange(4) + 10 * index + 0.5
        source.waveformvalues_wholedt = np.arange(4) + 10 * index
        sources.append(source)
    info, other, waveforms = htod_src_arrays(sources, grid)
    np.testing.assert_array_equal(
        _host(info, np.int32).reshape(-1, 4), [[i, i + 1, i + 2, i] for i in range(3)]
    )
    np.testing.assert_array_equal(
        _host(other, np.float64),
        [source.dl if family is HertzianDipole else 0 for source in sources],
    )
    expected = [
        source.waveformvalues_halfdt if family is HertzianDipole else source.waveformvalues_wholedt
        for source in sources
    ]
    np.testing.assert_array_equal(_host(waveforms, np.float64).reshape(-1, 4), expected)


@pytest.mark.parametrize("backend", ["cuda", "opencl", "metal"])
@pytest.mark.parametrize("real", ["float", "double"])
def test_every_shared_template_gates_only_hard_assignment(backend, real):
    args = update_voltage_source[f"args_{backend}"].substitute(REAL=real)
    body = update_voltage_source["func"].substitute(REAL=real, CUDA_IDX="")
    assert args.count("srcinfo1") == 1
    assert "int activity_offset = 4 * NVOLTSRC + 2 * source;" in body
    assert "int active = iteration >= first_active && iteration <= last_active;" in body
    assert body.count("else if (active)") == 3
    assert body.count("if (resistance != 0)") == 3
    assert "return;" not in body  # An inactive source must not skip later sources.
    for component, spacing in zip(("Ex", "Ey", "Ez"), ("dx", "dy", "dz")):
        assert (
            f"{component}[IDX3D_FIELDS(x,y,z)] = -1 * srcwaveforms[IDX2D_SRCWAVES(source,iteration)] / {spacing};"
            in body
        )
