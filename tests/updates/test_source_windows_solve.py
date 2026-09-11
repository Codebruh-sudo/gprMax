"""Real-device parity for shared waveform caches and hard-voltage windows.

These small no-PML models compare the same discretisation, not analytical
accuracy. Metal shares the checked template but requires separate hardware.
"""

import h5py
import numpy as np
import pytest

import gprMax

pytestmark = pytest.mark.integration

FAMILIES = ("HertzianDipole", "MagneticDipole", "VoltageSource", "TransmissionLine")
COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
START, STOP, END = 5e-11, 9e-11, 1.8e-10


def _base():
    scene = gprMax.Scene()
    for obj in (
        gprMax.Domain(p1=(0.024, 0.022, 0.020)),
        gprMax.Discretisation(p1=(0.001,) * 3),
        gprMax.TimeWindow(time=END),
        gprMax.PMLThickness(thickness=0),
        gprMax.OMPThreads(n=1),
    ):
        scene.add(obj)
    return scene


def _read(path):
    result = {}
    with h5py.File(str(path) + ".h5", "r") as output:
        dt = float(output.attrs["dt"])
        for receiver in output["rxs"].values():
            name = receiver.attrs["Name"]
            for component in COMPONENTS:
                if component in receiver:
                    result[f"rx/{name}/{component}"] = receiver[component][:]
        if "ports" in output:
            for port_id, port in output["ports"].items():
                for name in ("Vtotal", "Itotal", "Vgenerator"):
                    if name in port:
                        result[f"port/{port_id}/{name}"] = port[name][:]
    return result, dt


def _options(backend, device, precision):
    return (
        {"cpu_precision": precision}
        if backend == "cpu"
        else {"gpu" if backend == "cuda" else "opencl": [device], "gpu_precision": precision}
    )


def _run(path, scene, backend, device, precision, **kwargs):
    gprMax.run(
        scenes=[scene],
        outputfile=path,
        hide_progress_bars=True,
        log_level=40,
        **_options(backend, device, precision),
        **kwargs,
    )
    return _read(path)


def _equal(left, right):
    assert left.keys() == right.keys()
    for name in left:
        assert left[name].dtype == right[name].dtype
        assert left[name].tobytes() == right[name].tobytes(), name


def _parity(cpu, device, precision):
    assert cpu.keys() == device.keys()
    tolerance = 5e-5 if precision == "single" else 2e-12
    for name, reference in cpu.items():
        assert np.isfinite(device[name]).all(), name
        if name.startswith("rx/"):
            family = name.rsplit("/", 1)[1][0]
            scale = max(
                float(np.max(np.abs(values)))
                for key, values in cpu.items()
                if key.startswith("rx/") and key.rsplit("/", 1)[1].startswith(family)
            )
        else:
            scale = float(np.max(np.abs(reference)))
        np.testing.assert_allclose(
            device[name], reference, rtol=tolerance, atol=tolerance * max(scale, 1e-12), err_msg=name
        )


def _cache_scene(family, reverse=False, independent=False):
    scene = _base()
    amplitude = 1e-3 if family in ("HertzianDipole", "MagneticDipole") else 1
    for name in ("shared", "independent"):
        scene.add(gprMax.Waveform(wave_type="gaussian", amp=amplitude, freq=2e10, id=name))
    source_class = getattr(gprMax, family)
    extra = {"resistance": 50} if family in ("VoltageSource", "TransmissionLine") else {}
    sources = [
        source_class(
            p1=(0.008, 0.011, 0.010), polarisation="z", waveform_id="shared", start=START, stop=1.3e-10, **extra
        ),
        source_class(
            p1=(0.016, 0.011, 0.010), polarisation="z", waveform_id="independent" if independent else "shared", **extra
        ),
    ]
    for source in reversed(sources) if reverse else sources:
        scene.add(source)
    scene.add(gprMax.Rx(p1=(0.012, 0.012, 0.010), id="between"))
    scene.add(gprMax.Rx(p1=(0.017, 0.013, 0.011), id="remote"))
    return scene


def _cache_case(tmp_path, backend, device, family, precision):
    results = {}
    for mode, reverse, independent in (
        ("custom_first", False, False),
        ("full_first", True, False),
        ("independent", False, True),
    ):
        for solver, selected in (("cpu", None), (backend, device)):
            results[solver, mode], _ = _run(
                tmp_path / f"{solver}_{mode}", _cache_scene(family, reverse, independent), solver, selected, precision
            )
        _parity(results["cpu", mode], results[backend, mode], precision)
    for solver in ("cpu", backend):
        # Receiver fields must be byte-identical regardless of source order.
        # Automatic port IDs follow source insertion, so compare those only
        # against the same-order independent-ID control.
        fields = lambda mode: {key: values for key, values in results[solver, mode].items() if key.startswith("rx/")}
        _equal(fields("custom_first"), fields("independent"))
        _equal(fields("full_first"), fields("independent"))
        _equal(results[solver, "custom_first"], results[solver, "independent"])


@pytest.mark.gpu
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("precision", ["single", "double"])
def test_cuda_shared_waveform_windows(tmp_path, gpu_device, family, precision):
    _cache_case(tmp_path, "cuda", gpu_device, family, precision)


@pytest.mark.gpu
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("precision", ["single", "double"])
def test_opencl_shared_waveform_windows(tmp_path, opencl_device, family, precision):
    _cache_case(tmp_path, "opencl", opencl_device, family, precision)


def _hard_scene(polarisation, variant, *, start=START, stop=STOP):
    scene = _base()
    scene.add(gprMax.Waveform(wave_type="gaussian", amp=1e-3, freq=2e10, id="incident"))
    scene.add(gprMax.HertzianDipole(p1=(0.010, 0.009, 0.008), polarisation=polarisation, waveform_id="incident"))
    scene.add(gprMax.Waveform(wave_type="gaussian", amp=0 if variant == "zero" else 1, freq=2e10, id="voltage"))
    window = {} if variant == "full" else {"start": start, "stop": stop}
    voltage = gprMax.VoltageSource(
        p1=(0.012, 0.011, 0.010),
        polarisation=polarisation,
        resistance=50 if variant == "soft" else 0,
        waveform_id="voltage",
        **window,
    )
    scene.add(voltage)
    scene.add(gprMax.Rx(p1=(0.012, 0.011, 0.010), id="edge"))
    scene.add(gprMax.Rx(p1=(0.014, 0.013, 0.011), id="remote"))
    return scene, voltage


def _check_release(traces, dt, polarisation, variant, start=START, stop=STOP):
    edge = traces[f"rx/edge/E{polarisation}"]
    # Hard sources prescribe the stored electric time itself; additive
    # voltage-source updates still precede the stored E sample by one step.
    updated_at = (np.arange(edge.size) - (variant == "soft")) * dt
    if variant != "full":
        before = (updated_at >= 0) & (updated_at < start)
        after = updated_at > stop
        assert np.max(np.abs(edge[before])) > 1e-5
        assert np.max(np.abs(edge[after])) > 1e-5
        if variant == "zero":
            active = (updated_at >= start) & (updated_at <= stop)
            assert np.count_nonzero(edge[active]) == 0


def _hard_case(tmp_path, backend, device, polarisation, precision):
    for variant in ("window", "zero", "full", "soft"):
        cpu, dt = _run(tmp_path / f"cpu_{variant}", _hard_scene(polarisation, variant)[0], "cpu", None, precision)
        actual, device_dt = _run(
            tmp_path / f"{backend}_{variant}", _hard_scene(polarisation, variant)[0], backend, device, precision
        )
        assert dt == device_dt
        _parity(cpu, actual, precision)
        for traces in (cpu, actual):
            _check_release(traces, dt, polarisation, variant)


@pytest.mark.gpu
@pytest.mark.parametrize("polarisation", ["x", "y", "z"])
@pytest.mark.parametrize("precision", ["single", "double"])
def test_cuda_hard_voltage_activity(tmp_path, gpu_device, polarisation, precision):
    _hard_case(tmp_path, "cuda", gpu_device, polarisation, precision)


@pytest.mark.gpu
@pytest.mark.parametrize("polarisation", ["x", "y", "z"])
@pytest.mark.parametrize("precision", ["single", "double"])
def test_opencl_hard_voltage_activity(tmp_path, opencl_device, polarisation, precision):
    _hard_case(tmp_path, "opencl", opencl_device, polarisation, precision)


def _study_case(tmp_path, backend, device, precision):
    windows = [((START, 1.3e-10), (0, END)), ((2e-11, 7e-11), (START, 1.4e-10)), ((7e-11, 1.1e-10), (0, END))]

    def scene_for(bounds):
        scene = _base()
        scene.add(gprMax.Waveform(wave_type="gaussian", amp=1e-3, freq=2e10, id="shared"))
        sources = []
        for x, (start, stop) in zip((0.008, 0.016), bounds):
            source = gprMax.HertzianDipole(
                p1=(x, 0.011, 0.010), polarisation="z", waveform_id="shared", start=start, stop=stop
            )
            sources.append(source)
            scene.add(source)
        scene.add(gprMax.Rx(p1=(0.012, 0.012, 0.010), id="between"))
        scene.add(gprMax.Rx(p1=(0.017, 0.013, 0.011), id="remote"))
        return scene, sources

    for solver, selected in (("cpu", None), (backend, device)):
        scene, sources = scene_for(windows[0])
        study = gprMax.GPRStudy(
            [
                gprMax.StudyCase(
                    f"window{index}",
                    [
                        gprMax.ObjectState(source, start=start, stop=stop)
                        for source, (start, stop) in zip(sources, bounds)
                    ],
                )
                for index, bounds in enumerate(windows, start=1)
            ]
        )
        gprMax.run(
            scenes=[scene],
            study=study,
            outputfile=tmp_path / f"{solver}_study",
            hide_progress_bars=True,
            log_level=40,
            **_options(solver, selected, precision),
        )
        for index, bounds in enumerate(windows, start=1):
            reused, _ = _read(tmp_path / f"{solver}_study{index}")
            fresh, _ = _run(
                tmp_path / f"{solver}_fresh{index}",
                scene_for(bounds)[0],
                solver,
                selected,
                precision,
            )
            _equal(reused, fresh)
            with h5py.File(str(tmp_path / f"{solver}_study{index}") + ".h5", "r") as output:
                assert bool(output["study"].attrs["GeometryReused"]) == (index > 1)
    for index in range(1, len(windows) + 1):
        _parity(_read(tmp_path / f"cpu_study{index}")[0], _read(tmp_path / f"{backend}_study{index}")[0], precision)


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["single", "double"])
def test_cuda_shared_waveform_study_resamples_windows(tmp_path, gpu_device, precision):
    _study_case(tmp_path, "cuda", gpu_device, precision)


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["single", "double"])
def test_opencl_shared_waveform_study_resamples_windows(tmp_path, opencl_device, precision):
    _study_case(tmp_path, "opencl", opencl_device, precision)


def _hard_geometry_fixed_case(tmp_path, backend, device, precision):
    # Hard sources are deliberately unsupported in GPRStudy/PortStudy. Their
    # supported legacy geometry_fixed repeat must still reset device state.
    for solver, selected in (("cpu", None), (backend, device)):
        scene, _ = _hard_scene("z", "zero")
        gprMax.run(
            scenes=[scene],
            n=2,
            geometry_fixed=True,
            outputfile=tmp_path / f"{solver}_reused",
            hide_progress_bars=True,
            log_level=40,
            **_options(solver, selected, precision),
        )
        fresh, _ = _run(tmp_path / f"{solver}_fresh", _hard_scene("z", "zero")[0], solver, selected, precision)
        for index in (1, 2):
            reused, dt = _read(tmp_path / f"{solver}_reused{index}")
            _equal(reused, fresh)
            _check_release(reused, dt, "z", "zero")
    _parity(_read(tmp_path / "cpu_reused2")[0], _read(tmp_path / f"{backend}_reused2")[0], precision)


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["single", "double"])
def test_cuda_hard_voltage_geometry_fixed(tmp_path, gpu_device, precision):
    _hard_geometry_fixed_case(tmp_path, "cuda", gpu_device, precision)


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["single", "double"])
def test_opencl_hard_voltage_geometry_fixed(tmp_path, opencl_device, precision):
    _hard_geometry_fixed_case(tmp_path, "opencl", opencl_device, precision)
