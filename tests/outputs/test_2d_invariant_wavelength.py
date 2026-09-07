"""Changing an invariant thickness must not change spatial output validity."""

from itertools import product

import h5py
import numpy as np
import pytest

import gprMax


def _scene(family, axis, thickness, spectrum_limit=10):
    spacing = [0.001] * 3
    spacing[axis] = thickness
    domain = [0.032] * 3
    domain[axis] = float("inf")
    lower, upper = [0.0] * 3, domain.copy()
    source, receiver = [0.012] * 3, [0.020] * 3
    tag_upper = [0.021] * 3
    for point in (lower, source, receiver, tag_upper):
        point[axis] = float("inf")
    pml = [4] * 6
    pml[axis] = pml[axis + 3] = 0

    # Hertzian I*dl/volume cancels the invariant length for TM. The magnetic
    # dipole takes a moment/volume, so scale its moment with thickness for TE
    # to keep the same excitation per unit invariant length. Use that same
    # target amplitude when normalising the absorption output.
    amplitude = 1.0 if family == "TM" else thickness
    source_class = gprMax.HertzianDipole if family == "TM" else gprMax.MagneticDipole
    scene = gprMax.Scene()
    for item in (
        gprMax.Discretisation(p1=spacing),
        gprMax.DomainMode(mode=family),
        gprMax.Domain(p1=domain),
        gprMax.TimeWindow(time=2e-9),
        gprMax.PMLThickness(thickness=pml),
        gprMax.OMPThreads(n=1),
        gprMax.Material(er=2, se=0.2, mr=1, sm=0, id="lossy"),
        gprMax.MaterialDensity(density=1000, material_ids="lossy"),
        gprMax.Box(p1=lower, p2=upper, material_id="lossy"),
        gprMax.Box(p1=receiver, p2=tag_upper, material_id="lossy", tag="sample"),
        gprMax.Waveform(wave_type="ricker", amp=amplitude, freq=1e9, id="pulse"),
        source_class(p1=source, polarisation="xyz"[axis], waveform_id="pulse"),
        gprMax.Rx(p1=receiver, id="receiver"),
    ):
        scene.add(item)
    outputs = []
    for cls in (gprMax.SAR, gprMax.Radiometry):
        output = cls(
            frequencies=(0.75e9, 1e9, 1.25e9),
            tags="sample",
            waveform_id="pulse",
            target_amplitude=amplitude,
            spectrum_limit=spectrum_limit,
        )
        scene.add(output)
        outputs.append(output)
    return scene, outputs


def _run_pair(tmp_path, family, axis, *, options, rtol, spectrum_limit=10):
    records = []
    for thickness in (0.001, 0.1):
        scene, outputs = _scene(family, axis, thickness, spectrum_limit)
        path = tmp_path / f"{family}{axis}_{int(thickness * 1000)}mm"
        gprMax.run(
            scenes=[scene], outputfile=path, hide_progress_bars=True, log_level=40, **options
        )
        record = {}
        with h5py.File(path.with_suffix(".h5")) as data:
            record["dt"] = data.attrs["dt"]
            for component in ("Ex", "Ey", "Ez"):
                record[component] = data[f"rxs/rx1/{component}"][...]
            for group_name in ("sar/sar1", "radiometry/radiometry1"):
                group = data[group_name]
                assert np.all(group["valid"][...])
                assert np.all(group["mesh_valid"][...])
                record[group_name] = group["absorbed_power_density"][...]
                assert np.all(np.isfinite(record[group_name]))
                assert np.max(record[group_name]) > 0
                record[group_name + "_cells"] = group["cells_per_wavelength"][...]
                assert np.all(record[group_name + "_cells"] > 100)
                record[group_name + "_integral"] = group[
                    "tags/sample/absorbed_power_per_length"
                ][...]
            record["sar"] = data["sar/sar1/sar"][...]
        records.append(record)
        # Also check the in-memory result: a file-only fix would be incomplete.
        for output in outputs:
            assert np.all(output.result.valid)
            assert np.all(output.result.mesh_valid)
            if "gpu" in options:
                assert output._monitor.collection_backend == "cuda_device"
            elif "opencl" in options:
                assert output._monitor.collection_backend == "opencl_device"
    reference, actual = records
    assert actual["dt"] == reference["dt"]
    for key, expected in reference.items():
        scale = float(np.max(np.abs(expected)))
        np.testing.assert_allclose(actual[key], expected, rtol=rtol, atol=rtol * scale)


@pytest.mark.integration
@pytest.mark.parametrize("family,axis", tuple(product(("TM", "TE"), range(3))))
@pytest.mark.parametrize("spectrum_limit", (10, "nyquist"))
def test_cpu_2d_invariant_thickness_preserves_fields_and_output_validity(
    tmp_path, family, axis, spectrum_limit
):
    _run_pair(
        tmp_path, family, axis, options={"cpu_precision": "double"},
        rtol=2e-12, spectrum_limit=spectrum_limit,
    )


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize("family,axis", tuple(product(("TM", "TE"), range(3))))
@pytest.mark.parametrize("backend", ("cuda", "opencl"))
def test_device_2d_invariant_thickness_preserves_output_validity(
    tmp_path, request, family, axis, backend
):
    device = request.getfixturevalue("gpu_device" if backend == "cuda" else "opencl_device")
    options = {
        "gpu" if backend == "cuda" else "opencl": [device],
        "gpu_precision": "single",
    }
    _run_pair(tmp_path, family, axis, options=options, rtol=3e-5)
