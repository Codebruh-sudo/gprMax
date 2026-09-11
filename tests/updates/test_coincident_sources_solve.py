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

"""Coincident conventional sources follow the CPU's ordered update semantics.

Tiny no-PML scenes isolate source accumulation and assignment ordering. These
are backend-parity checks, not a physical interpretation of overlapping feeds.
"""

import h5py
import numpy as np
import pytest
from scipy.constants import c

import gprMax

pytestmark = pytest.mark.integration
SOLVERS = [
    pytest.param(
        backend,
        precision,
        id=f"{backend}-{precision}",
        marks=pytest.mark.gpu if backend != "cpu" else (),
    )
    for backend in ("cpu", "cuda", "opencl")
    for precision in ("single", "double")
] + [pytest.param("metal", "single", id="metal-single", marks=pytest.mark.gpu)]
COMPONENTS = tuple(kind + axis for kind in "EH" for axis in "xyz")
DL = 0.001
ITERATIONS = 80
DT = DL / (c * np.sqrt(3))
EARLY = (0, 30.25 * DT)
LATE = (35.25 * DT, 74.75 * DT)
OVERLAP = (15.25 * DT, 48.75 * DT)
SOURCE_POINT = (0.006,) * 3
SOURCE_TYPES = {
    "electric": gprMax.HertzianDipole,
    "magnetic": gprMax.MagneticDipole,
    "voltage": gprMax.VoltageSource,
    "line": gprMax.TransmissionLine,
}


def _source(family, amplitude=1, window=None, resistance=None):
    return dict(family=family, amplitude=amplitude, window=window, resistance=resistance)


def _scene(specifications, polarisation):
    scene = gprMax.Scene()
    for obj in (
        gprMax.Domain(p1=(0.012,) * 3),
        gprMax.Discretisation(p1=(DL,) * 3),
        gprMax.PMLThickness(thickness=0),
        gprMax.OMPThreads(n=1),
        gprMax.TimeWindow(iterations=ITERATIONS),
        gprMax.Rx(p1=SOURCE_POINT, id="edge"),
        gprMax.Rx(p1=(0.008, 0.007, 0.009), id="remote"),
    ):
        scene.add(obj)
    sources = []
    for index, spec in enumerate(specifications):
        waveform = f"wave{index}"
        scene.add(
            gprMax.Waveform(wave_type="gaussian", amp=spec["amplitude"], freq=2e10, id=waveform)
        )
        options = {}
        if spec["window"] is not None:
            options.update(zip(("start", "stop"), spec["window"]))
        if spec["resistance"] is not None:
            options["resistance"] = spec["resistance"]
        if spec["family"] == "voltage":
            options["id"] = f"voltage{index}"
        source = SOURCE_TYPES[spec["family"]](
            p1=spec.get("position", SOURCE_POINT),
            polarisation=polarisation,
            waveform_id=waveform,
            **options,
        )
        sources.append(source)
        scene.add(source)
    return scene, sources


def _run(path, specifications, polarisation, precision, backend="cpu", device=None):
    scene, sources = _scene(specifications, polarisation)
    options = (
        {"cpu_precision": precision}
        if backend == "cpu"
        else {"gpu" if backend == "cuda" else "opencl": [device], "gpu_precision": precision}
    )
    if backend == "metal":
        options = {"metal": True, "gpu_precision": precision}
    gprMax.run(scenes=[scene], outputfile=path, hide_progress_bars=True, log_level=40, **options)
    with h5py.File(path.with_suffix(".h5")) as output:
        fields = {
            f"{receiver.attrs['Name']}/{component}": receiver[component][...]
            for receiver in output["rxs"].values()
            for component in COMPONENTS
        }
        dt = float(output.attrs["dt"])
        lines = {
            f"{name}/{component}": line[component][...]
            for name, line in output.get("tls", {}).items()
            for component in ("Vtotal", "Itotal", "Vinc", "Iinc")
        }
    return dict(fields=fields, lines=lines, dt=dt, sources=sources)


@pytest.fixture(scope="module")
def cpu_reference(tmp_path_factory):
    """Reuse immutable CPU output references, not gprMax geometry or waveforms."""
    directory = tmp_path_factory.mktemp("coincident_cpu")
    cache = {}

    def reference(specifications, polarisation, precision):
        key = (repr(specifications), polarisation, precision)
        if key not in cache:
            cache[key] = _run(
                directory / f"reference{len(cache)}", specifications, polarisation, precision
            )
        return cache[key]

    return reference


def _device(request, backend):
    if backend == "cpu":
        return None
    if backend == "metal":
        metal = pytest.importorskip("Metal", reason="Apple Metal/PyObjC is unavailable")
        if metal.MTLCreateSystemDefaultDevice() is None:
            pytest.skip("No Apple Metal device is available")
        return None
    return request.getfixturevalue("gpu_device" if backend == "cuda" else "opencl_device")


def _compare_fields(actual, expected, scale_reference, precision):
    """Check every component, using only same-unit scales for zero components."""
    assert actual.keys() == expected.keys() == scale_reference.keys()
    tolerance = 5e-5 if precision == "single" else 2e-12
    for name, reference in expected.items():
        kind = name.rsplit("/", 1)[1][0]
        scale = max(
            float(np.max(np.abs(values)))
            for key, values in scale_reference.items()
            if key.rsplit("/", 1)[1].startswith(kind)
        )
        assert scale > 0, kind
        assert np.isfinite(reference).all(), name
        assert np.isfinite(actual[name]).all(), name
        np.testing.assert_allclose(
            actual[name], reference, rtol=tolerance, atol=tolerance * scale, err_msg=name
        )


@pytest.mark.parametrize("backend,precision", SOLVERS)
@pytest.mark.parametrize("family", ["electric", "magnetic"])
@pytest.mark.parametrize("polarisation", ["x", "y", "z"])
@pytest.mark.parametrize("case", ["duplicate", "cancel", "disjoint_windows"])
def test_coincident_dipoles_superpose(
    tmp_path, request, cpu_reference, backend, family, polarisation, precision, case
):
    device = _device(request, backend)
    single = cpu_reference([_source(family)], polarisation, precision)
    if case == "disjoint_windows":
        first = cpu_reference([_source(family, window=EARLY)], polarisation, precision)
        second = cpu_reference([_source(family, window=LATE)], polarisation, precision)
        expected = {
            name: first["fields"][name] + second["fields"][name] for name in single["fields"]
        }
        specifications = [_source(family, window=EARLY), _source(family, window=LATE)]
    else:
        factor = 2 if case == "duplicate" else 0
        expected = {name: factor * values for name, values in single["fields"].items()}
        specifications = [
            _source(family),
            _source(family, amplitude=1 if case == "duplicate" else -1),
        ]
    actual = _run(tmp_path / "actual", specifications, polarisation, precision, backend, device)
    assert actual["dt"] == single["dt"]
    # Nonzero reference at a remote receiver prevents an inert source or bad
    # receiver placement from satisfying either cancellation or parity checks.
    component = ("E" if family == "electric" else "H") + polarisation
    assert np.linalg.norm(single["fields"][f"remote/{component}"]) > 0
    _compare_fields(actual["fields"], expected, single["fields"], precision)


def _voltage_specs(case):
    hard = _source("voltage", 0.7, resistance=0)
    windowed_hard = _source("voltage", -0.4, OVERLAP, resistance=0)
    soft = _source("voltage", 0.3, OVERLAP, resistance=50)
    return {
        "hard_then_hard": [hard, windowed_hard],
        "hard_reversed": [windowed_hard, hard],
        "hard_then_soft": [hard, soft],
        "soft_then_hard": [soft, hard],
        "soft_pair": [
            _source("voltage", 0.7, resistance=50),
            _source("voltage", -0.4, OVERLAP, resistance=75),
        ],
        "hard_disjoint": [
            _source("voltage", 0.7, EARLY, resistance=0),
            _source("voltage", -0.4, LATE, resistance=0),
        ],
        "mixed_families": [
            hard,
            _source("electric", 1e-3),
            _source("electric", 0.25e-3, OVERLAP),
            _source("magnetic", 1e-5),
            _source("magnetic", 0.25e-5, OVERLAP),
        ],
    }[case]


def _check_hard_assignments(result, polarisation, case):
    """The last active hard source writes -V/dl at its electric edge."""
    edge = result["fields"][f"edge/E{polarisation}"]
    samples = np.arange(edge.size)
    iterations = samples - 1  # Additive updates precede this stored E sample.
    expected, prescribed = np.zeros_like(edge), np.zeros(edge.size, dtype=bool)
    soft_active = np.zeros(edge.size, dtype=bool)
    for source in result["sources"]:
        if not isinstance(source, gprMax.VoltageSource):
            continue
        internal = source._source
        if internal.resistance == 0:
            time = samples * result["dt"]
            active = (time >= internal.start) & (time <= internal.stop)
            expected[active] = -internal.waveformvalues_wholedt[samples[active]] / DL
            prescribed |= active
        else:
            time = iterations * result["dt"]
            active = (iterations >= 0) & (time >= internal.start) & (time <= internal.stop)
            soft_active |= active
    if case == "hard_then_soft":
        # A later soft source adds its correction; it must not be erased by
        # a racing hard writer. Outside its window the hard value is exact.
        assert np.linalg.norm(edge[soft_active] - expected[soft_active]) > 1e-3
        prescribed &= ~soft_active
    assert np.count_nonzero(prescribed) > 0
    np.testing.assert_allclose(edge[prescribed], expected[prescribed], rtol=2e-6, atol=1e-6)
    if case == "hard_disjoint":
        assert np.max(np.abs(edge[(iterations >= 0) & ~prescribed])) > 1e-3


@pytest.mark.parametrize("backend,precision", SOLVERS)
@pytest.mark.parametrize("polarisation", ["x", "y", "z"])
@pytest.mark.parametrize(
    "case",
    [
        "hard_then_hard",
        "hard_reversed",
        "hard_then_soft",
        "soft_then_hard",
        "soft_pair",
        "hard_disjoint",
        "mixed_families",
    ],
)
def test_coincident_voltage_sources_follow_cpu_order(
    tmp_path, request, cpu_reference, backend, polarisation, precision, case
):
    device = _device(request, backend)
    specifications = _voltage_specs(case)
    reference = cpu_reference(specifications, polarisation, precision)
    actual = (
        reference
        if backend == "cpu"
        else _run(tmp_path / "actual", specifications, polarisation, precision, backend, device)
    )
    assert actual["dt"] == reference["dt"]
    _compare_fields(actual["fields"], reference["fields"], reference["fields"], precision)
    if case == "soft_pair":
        # Keep BOTH resistive loads in each control: removing an inactive
        # source would change the geometry and invalidate superposition.
        first_specs = [
            dict(spec, amplitude=spec["amplitude"] if index == 0 else 0)
            for index, spec in enumerate(specifications)
        ]
        second_specs = [
            dict(spec, amplitude=spec["amplitude"] if index == 1 else 0)
            for index, spec in enumerate(specifications)
        ]
        first = cpu_reference(first_specs, polarisation, precision)
        second = cpu_reference(second_specs, polarisation, precision)
        expected = {
            name: first["fields"][name] + second["fields"][name] for name in reference["fields"]
        }
        _compare_fields(actual["fields"], expected, reference["fields"], precision)
    elif case != "mixed_families":
        _check_hard_assignments(actual, polarisation, case)


def _line_specs(case):
    lines = [_source("line", 0.7, resistance=50), _source("line", -0.4, OVERLAP, resistance=75)]
    if case == "reversed":
        return lines[::-1]
    if case == "mixed_families":
        # CPU family order is voltage -> TL -> Hertzian, independently of
        # scene insertion. Both TL states must advance even on the same edge.
        return [
            _source("electric", 1e-3),
            *lines,
            _source("voltage", 0.2, resistance=0),
            _source("magnetic", 1e-5),
        ]
    return lines


@pytest.mark.parametrize("backend,precision", SOLVERS)
@pytest.mark.parametrize("polarisation", ["x", "y", "z"])
@pytest.mark.parametrize("case", ["forward", "reversed", "mixed_families"])
def test_coincident_transmission_lines_follow_cpu_order(
    tmp_path, request, cpu_reference, backend, polarisation, precision, case
):
    device = _device(request, backend)
    specifications = _line_specs(case)
    reference = cpu_reference(specifications, polarisation, precision)
    actual = (
        reference
        if backend == "cpu"
        else _run(tmp_path / "actual", specifications, polarisation, precision, backend, device)
    )
    assert actual["dt"] == reference["dt"]
    _compare_fields(actual["fields"], reference["fields"], reference["fields"], precision)
    assert actual["lines"].keys() == reference["lines"].keys()
    assert len(reference["lines"]) == 8
    tolerance = 5e-5 if precision == "single" else 2e-12
    for name, expected in reference["lines"].items():
        scale = float(np.max(np.abs(expected)))
        assert scale > 0, name
        assert np.isfinite(actual["lines"][name]).all(), name
        np.testing.assert_allclose(
            actual["lines"][name], expected, rtol=tolerance, atol=tolerance * scale, err_msg=name
        )


@pytest.mark.parametrize(
    "backend,precision",
    [
        parameter
        for parameter in SOLVERS
        if parameter.values[1] == "double" or parameter.values[0] == "metal"
    ],
)
def test_study_moves_dipoles_together_then_apart(
    tmp_path, request, cpu_reference, backend, precision
):
    device = _device(request, backend)
    positions = (
        ((0.004, 0.006, 0.006), (0.008, 0.006, 0.006)),
        (SOURCE_POINT, SOURCE_POINT),
        ((0.008, 0.006, 0.006), (0.004, 0.006, 0.006)),
    )
    specifications = [_source("electric", 1), _source("electric", -0.4)]

    def placed(pair):
        return [dict(spec, position=position) for spec, position in zip(specifications, pair)]

    scene, sources = _scene(placed(positions[0]), "z")
    study = gprMax.GPRStudy(
        [
            gprMax.StudyCase(
                f"position{index}",
                [
                    gprMax.ObjectState(source, position=position)
                    for source, position in zip(sources, pair)
                ],
            )
            for index, pair in enumerate(positions)
        ]
    )
    if backend == "cpu":
        options = {"cpu_precision": precision}
    elif backend == "metal":
        options = {"metal": True, "gpu_precision": precision}
    else:
        options = {"gpu" if backend == "cuda" else "opencl": [device], "gpu_precision": precision}
    gprMax.run(
        scenes=[scene],
        study=study,
        outputfile=tmp_path / "study",
        hide_progress_bars=True,
        log_level=40,
        **options,
    )
    for index, pair in enumerate(positions, start=1):
        fresh = _run(tmp_path / f"fresh{index}", placed(pair), "z", precision, backend, device)
        cpu = fresh if backend == "cpu" else cpu_reference(placed(pair), "z", precision)
        with h5py.File(tmp_path / f"study{index}.h5") as output:
            assert bool(output["study"].attrs["GeometryReused"]) == (index > 1)
            reused = {
                f"{receiver.attrs['Name']}/{component}": receiver[component][...]
                for receiver in output["rxs"].values()
                for component in COMPONENTS
            }
        for name, expected in fresh["fields"].items():
            # The same-backend comparison is deliberately exact: positions,
            # drives, and geometry are identical, only the reuse path differs.
            np.testing.assert_array_equal(reused[name], expected, err_msg=f"case{index}/{name}")
        _compare_fields(reused, cpu["fields"], cpu["fields"], precision)
