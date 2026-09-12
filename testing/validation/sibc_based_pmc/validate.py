"""Exact SIBC-based PMC: image equivalence, reflection, cavity modes, stability.

Run from the repository root with ``python -m
testing.validation.sibc_based_pmc.validate``. An exact zero-admittance adapter
is used only inside this diagnostic; resistance=inf is not a public API yet.
"""

import argparse
import csv
import json
import platform
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import epsilon_0, mu_0

import gprMax
from gprMax.cython.fields_updates_normal import update_electric, update_magnetic
from gprMax.impedance_surfaces import _component_valid_view
from gprMax.solvers import Solver
from testing.validation.impedance_surface.default_cfl import full_solver_run
from testing.validation.impedance_surface.stability import (
    Stepper,
    audit_geometry,
    build_case,
    source_free_run,
)

from .runtime import assert_zero_admittance, base_scene, build_grid, exact_pmc

NAMES = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
ETA = np.sqrt(mu_0 / epsilon_0)
VELOCITY = 1 / np.sqrt(mu_0 * epsilon_0)


def fields(grid):
    return [getattr(grid, name) for name in NAMES]


def advance(grid):
    arrays = fields(grid)
    update_magnetic(grid.nx, grid.ny, grid.nz, -1, 1, grid.updatecoeffsH, grid.ID, *arrays)
    update_electric(grid.nx, grid.ny, grid.nz, -1, 1, grid.updatecoeffsE, grid.ID, *arrays)
    if grid.impedance_surfaces is not None:
        grid.impedance_surfaces.update(grid)


def image_case(cache, axis, precision, steps=200):
    """Compare every retained E/H component with an independently mirrored grid.

    Mirror signs: E_normal and H_tangent odd; E_tangent and H_normal even.
    Random three-dimensional initial fields excite both polarizations and
    spatial variations parallel to the wall, not just a normal TEM wave.
    """
    cells, spacing = np.array([8, 8, 8]), np.array([0.001, 0.0013, 0.0017])
    cells[axis] = 12
    wall = cells[axis] // 2
    reference = base_scene(cells, spacing)
    scene = base_scene(cells, spacing)
    for model in (reference, scene):
        for normal in "xyz":
            for side in ("0", "max"):
                model.add(gprMax.SymmetryBoundary(face=normal + side, type="pec"))
    scene.add(gprMax.SurfaceImpedance(id="wall", resistance=50))
    lower = np.zeros(3)
    lower[axis] = wall * spacing[axis]
    scene.add(
        gprMax.Box(p1=tuple(lower), p2=tuple(cells * spacing), material_id="wall", averaging="n")
    )
    full = build_grid(reference, cache / f"image_full_{axis}_{precision}", precision)
    with exact_pmc():
        half = build_grid(scene, cache / f"image_pmc_{axis}_{precision}", precision)
    system = assert_zero_admittance(half)
    assert full.dt == half.dt
    rng = np.random.default_rng(731 + axis)
    masks = []
    for component, (f, h) in enumerate(zip(fields(full), fields(half))):
        view = _component_valid_view(f, component, full)
        indices = np.indices(view.shape)
        half_offset = (component == axis) if component < 3 else (component - 3 != axis)
        coordinate = indices[axis] + 0.5 * half_offset
        retained = coordinate <= wall
        odd = (component == axis) if component < 3 else (component - 3 != axis)
        view[:] = rng.standard_normal(view.shape) / (ETA if component >= 3 else 1)
        # Compatible PEC outer faces on the reference box.
        for normal in range(3):
            zero_face = (normal != component) if component < 3 else (normal == component - 3)
            if zero_face:
                view[(indices[normal] == 0) | (indices[normal] == cells[normal])] = 0
        for i in range(view.shape[axis]):
            if i + 0.5 * half_offset <= wall:
                continue
            source = 2 * wall - i - int(half_offset)
            destination_slice = [slice(None)] * 3
            source_slice = [slice(None)] * 3
            destination_slice[axis], source_slice[axis] = i, source
            view[tuple(destination_slice)] = (-1 if odd else 1) * view[tuple(source_slice)]
        target = _component_valid_view(h, component, half)
        target[retained] = view[retained]
        masks.append(retained)
    reference_norm = np.sqrt(
        sum(
            np.sum((_component_valid_view(f, c, full)[m] * (ETA if c >= 3 else 1)) ** 2)
            for c, (f, m) in enumerate(zip(fields(full), masks))
        )
    )
    errors = []
    for _ in range(steps):
        advance(full)
        advance(half)
        difference = sum(
            np.sum(
                (
                    (_component_valid_view(a, c, full)[m] - _component_valid_view(b, c, half)[m])
                    * (ETA if c >= 3 else 1)
                )
                ** 2
            )
            for c, (a, b, m) in enumerate(zip(fields(full), fields(half), masks))
        )
        errors.append(float(np.sqrt(difference) / reference_norm))
    limit = 2e-5 if precision == "single" else 2e-12
    return dict(
        axis="xyz"[axis],
        precision=precision,
        steps=steps,
        maximum_field_relative_error=max(errors),
        limit=limit,
        boundary_edges=system.edge_count,
        passed=max(errors) < limit,
    )


def plane_scene(axis, e_axis, sign, boundary, resistance=1e6):
    """TEM sheet with a causally isolated wall at a main-voxel face."""
    h = 0.001
    third = next(a for a in range(3) if a not in (axis, e_axis))
    order = (axis, third, e_axis)

    def point(x, y, z):
        value = np.zeros(3)
        value[list(order)] = (x if sign > 0 else 1.8 - x, y, z)
        return tuple(value)

    cells = np.full(3, 4)
    cells[axis] = 1800
    scene = base_scene(cells, np.full(3, h))
    scene.single_use_objects = [
        x for x in scene.single_use_objects if not isinstance(x, gprMax.TimeWindow)
    ]
    scene.add(gprMax.TimeWindow(time=4e-9))
    for transverse, kind in ((third, "pmc"), (e_axis, "pec")):
        for side in ("0", "max"):
            scene.add(gprMax.SymmetryBoundary(face="xyz"[transverse] + side, type=kind))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=4e9, id="pulse"))
    if boundary != "reference":
        if boundary == "legacy":
            material = "pmc"
        else:
            material = "wall"
            scene.add(gprMax.SurfaceImpedance(id=material, resistance=resistance))
        corners = np.asarray([point(0.89, 0, 0), point(0.894, 0.004, 0.004)])
        scene.add(
            gprMax.Box(
                p1=tuple(corners.min(axis=0)),
                p2=tuple(corners.max(axis=0)),
                material_id=material,
                averaging="n",
            )
        )
    for j in range(5):
        for k in range(4):
            scene.add(gprMax.HertzianDipole(point(0.8, j * h, k * h), "xyz"[e_axis], "pulse"))
    for x in (0.86, 0.93):
        scene.add(gprMax.Rx(p1=point(x, 0.002, 0.002)))
    return scene


def plane_trace(cache, axis, e_axis, sign, boundary, *, resistance=1e6, precision="double"):
    scene = plane_scene(axis, e_axis, sign, boundary, resistance)
    path = cache / f"plane_{axis}_{e_axis}_{sign}_{boundary}_{resistance:g}_{precision}"
    with exact_pmc() if boundary == "exact" else nullcontext():
        gprMax.run(
            scenes=[scene],
            outputfile=path,
            cpu_precision=precision,
            hide_progress_bars=True,
            log_level=40,
        )
    with h5py.File(path.with_suffix(".h5")) as data:
        traces = np.array([data[f"rxs/rx{r}/E{'xyz'[e_axis]}"][:] for r in (1, 2)])
        dt = float(data.attrs["dt"])
    return traces, dt


def reflection(incident, total, dt, resistance=None):
    frequency = np.arange(1e9, 8e9 + 0.125e9, 0.25e9)
    time = np.arange(incident.shape[1]) * dt
    transform = np.exp(-2j * np.pi * frequency[:, None] * time)
    incoming = transform @ incident[0]
    theta = 2 * np.pi * frequency * dt
    k = 2 / 0.001 * np.arcsin(0.001 / (VELOCITY * dt) * np.sin(theta / 2))
    gamma = (transform @ (total[0] - incident[0])) / incoming * np.exp(2j * k * 0.03)
    expected = np.ones_like(gamma)
    if resistance is not None:
        z, eta = resistance * np.cos(k * 0.001 / 2), ETA * np.cos(theta / 2)
        expected = (z - eta) / (z + eta)
    phase = np.unwrap(np.angle(gamma))
    offset = -np.dot(k, phase) / (2 * np.dot(k, k))
    metrics = dict(
        max_complex_error=float(np.max(abs(gamma - expected))),
        max_magnitude_error=float(np.max(abs(abs(gamma) - 1))),
        max_phase_deg=float(np.max(abs(np.rad2deg(phase)))),
        fitted_plane_offset_m=float(offset),
        transmission_peak_ratio=float(np.max(abs(total[1])) / np.max(abs(incident[1]))),
        incident_min_relative_spectrum=float(np.min(abs(incoming)) / np.max(abs(incoming))),
    )
    return frequency, gamma, expected, k, metrics


def cavity_scene(counts, spacing, iterations):
    counts, spacing = np.asarray(counts), np.asarray(spacing)
    cells = counts + 4
    scene = base_scene(cells, spacing, iterations=iterations)
    scene.add(gprMax.SurfaceImpedance(id="wall", resistance=50))
    scene.add(
        gprMax.Box(
            p1=tuple(spacing), p2=tuple((counts + 3) * spacing), material_id="wall", averaging="n"
        )
    )
    scene.add(
        gprMax.Box(
            p1=tuple(2 * spacing),
            p2=tuple((counts + 2) * spacing),
            material_id="free_space",
            averaging="n",
        )
    )
    return scene


def cavity_mode(
    cache,
    indices,
    *,
    counts=(6, 7, 8),
    spacing=(0.001, 0.0013, 0.0017),
    precision="double",
    steps=2000,
):
    counts, spacing, indices = np.asarray(counts), np.asarray(spacing), np.asarray(indices)
    scene = cavity_scene(counts, spacing, steps)
    original = Solver.solve
    result, samples = {}, []

    def solve(solver, iterator):
        grid = solver.updates.grid
        assert_zero_admittance(grid)
        k = np.pi * indices / (counts * spacing)
        kd = 2 * np.sin(k * spacing / 2) / spacing
        amplitude = np.array([kd[1], -kd[0], 0.0])
        amplitude /= np.linalg.norm(amplitude)
        curl_amplitude = np.array(
            [
                kd[2] * amplitude[1] - kd[1] * amplitude[2],
                kd[0] * amplitude[2] - kd[2] * amplitude[0],
                kd[1] * amplitude[0] - kd[0] * amplitude[1],
            ]
        )
        omega_symbol = VELOCITY * np.linalg.norm(kd)
        theta = 2 * np.arcsin(grid.dt * omega_symbol / 2)
        shapes = []
        for component, field in enumerate(fields(grid)):
            view = _component_valid_view(field, component, grid)
            coordinates = np.indices(view.shape, dtype=float)
            normal = component % 3
            for axis in range(3):
                shifted = (normal == axis) if component < 3 else (normal != axis)
                coordinates[axis] = (coordinates[axis] + 0.5 * shifted - 2) * spacing[axis]
            shape = np.ones(view.shape)
            mask = np.ones(view.shape, dtype=bool)
            for axis in range(3):
                argument = k[axis] * coordinates[axis]
                sine = (normal == axis) if component < 3 else (normal != axis)
                shape *= np.sin(argument) if sine else np.cos(argument)
                mask &= (coordinates[axis] >= -1e-14) & (
                    coordinates[axis] <= counts[axis] * spacing[axis] + 1e-14
                )
            shape *= mask
            shape *= (
                amplitude[normal]
                if component < 3
                else -curl_amplitude[normal] / (mu_0 * omega_symbol)
            )
            shapes.append(shape)
            view[:] = shape if component < 3 else -np.sin(theta / 2) * shape
        e_norm = sum(np.sum(shape**2) for shape in shapes[:3])
        norm = sum(np.sum((shape * (ETA if c >= 3 else 1)) ** 2) for c, shape in enumerate(shapes))
        max_error = 0.0

        def observe(iterations):
            nonlocal max_error
            for iteration in iterations:
                yield iteration
                n = iteration + 1
                error, projection = 0.0, 0.0
                for component, (field, shape) in enumerate(zip(fields(grid), shapes)):
                    view = _component_valid_view(field, component, grid)
                    temporal = np.cos(n * theta) if component < 3 else np.sin((n - 0.5) * theta)
                    error += np.sum(
                        ((view - shape * temporal) * (ETA if component >= 3 else 1)) ** 2
                    )
                    if component < 3:
                        projection += np.sum(view * shape)
                max_error = max(max_error, float(np.sqrt(error / norm)))
                samples.append([n, projection / e_norm, np.cos(n * theta)])

        original(solver, observe(iterator))
        continuum = VELOCITY / 2 * np.linalg.norm(indices / (counts * spacing))
        limit = 2e-4 if precision == "single" else 2e-11
        result.update(
            indices=indices.tolist(),
            cells=counts.tolist(),
            spacing_m=spacing.tolist(),
            precision=precision,
            steps=steps,
            dt_s=grid.dt,
            discrete_frequency_hz=float(theta / (2 * np.pi * grid.dt)),
            continuum_frequency_hz=float(continuum),
            max_all_field_mode_error=max_error,
            limit=limit,
            passed=max_error < limit,
        )

    with exact_pmc(), patch.object(Solver, "solve", solve):
        gprMax.run(
            scenes=[scene],
            outputfile=cache / f"mode_{''.join(map(str,indices))}_{precision}",
            cpu_precision=precision,
            hide_progress_bars=True,
            log_level=40,
        )
    return result, samples


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def plot_results(output):
    """Render the saved numerical records without rerunning the solvers."""
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    for boundary, label in (("exact", "Exact SIBC PMC"), ("legacy", "Existing pmc volume")):
        data = np.genfromtxt(output / f"reflection_x_z_1_{boundary}.csv", delimiter=",", names=True)
        frequency = data["frequency_hz"]
        gamma = data["gamma_real"] + 1j * data["gamma_imag"]
        axes[0].plot(frequency / 1e9, np.rad2deg(np.angle(gamma)), label=label)
        axes[1].plot(frequency / 1e9, abs(gamma), label=label)
    dt = next(c["dt_s"] for c in summary["reflection_cases"] if c["name"] == "x_z_1_exact")
    k = 2 / 0.001 * np.arcsin(0.001 / (VELOCITY * dt) * np.sin(np.pi * frequency * dt))
    axes[0].plot(
        frequency / 1e9,
        np.rad2deg(-k * 0.001),
        ":",
        color="black",
        label="PMC shifted +half cell into volume",
    )
    axes[0].set(
        ylabel="Electric reflection phase (degrees)",
        title="Reflection at the declared main-voxel face",
    )
    axes[1].set(xlabel="Frequency (GHz)", ylabel="Electric reflection magnitude", ylim=(0.99, 1.01))
    axes[1].ticklabel_format(useOffset=False)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(output / "reflection.png", dpi=160)
    plt.close(fig)

    data = np.genfromtxt(output / "mode_121_double.csv", delimiter=",", names=True)[:60]
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    ax.plot(data["step"], data["discrete_pmc_theory"], label="Analytical discrete PMC mode")
    ax.plot(data["step"], data["electric_projection"], ".", label="Full solver, exact SIBC limit")
    ax.set(xlabel="Timestep", ylabel="Electric modal amplitude")
    ax.set_title("Rectangular PMC cavity: mode (1, 2, 1)", pad=40)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=9)
    ax.grid(alpha=0.25)
    fig.savefig(output / "cavity_mode.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    for geometry in ("cavity", "stair"):
        for precision in ("double", "single"):
            data = np.genfromtxt(
                output / f"stability_{geometry}_{precision}.csv", delimiter=",", names=True
            )
            ax.plot(data["step"], data["field_norm_over_initial"], label=f"{geometry}, {precision}")
    ax.set(
        xlabel="Timesteps in full solver",
        ylabel="Field norm / initial norm",
        title="Source-free exact PMC at timestep factor 0.99",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output / "stability.png", dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument(
        "--steps", type=int, default=200000, help="Long source-free stability run length"
    )
    args = parser.parse_args(argv)
    if args.steps < 1000:
        parser.error("--steps must be at least 1000")
    output = args.output_dir
    cache = output.parent / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(
        date="2026-09-12",
        python=platform.python_version(),
        numpy=np.__version__,
        backend="production CPU Cython and full Solver; one OpenMP thread",
        timestep_factor=0.99,
        boundary="exact zero surface admittance; validation-only discrete Z0=+inf override",
        cache_warning="Finite material declarations in cached HDF5 are geometry placeholders for exact-PMC cases",
        image_cases=[],
        reflection_cases=[],
        cavity_modes=[],
        stability_cases=[],
    )

    def save(section, case):
        summary[section].append(case)
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(json.dumps({"section": section, **case}), flush=True)

    for axis in range(3):
        for precision in ("double", "single"):
            save("image_cases", image_case(cache, axis, precision))
    for axis, e_axis, sign in [(a, e, 1) for a in range(3) for e in range(3) if a != e] + [
        (0, 2, -1)
    ]:
        incident, dt = plane_trace(cache, axis, e_axis, sign, "reference")
        for boundary in ("exact", "legacy"):
            total, actual_dt = plane_trace(cache, axis, e_axis, sign, boundary)
            assert dt == actual_dt and total.shape == incident.shape
            frequency, gamma, expected, k, metrics = reflection(incident, total, dt)
            name = f"{'xyz'[axis]}_{'xyz'[e_axis]}_{sign}_{boundary}"
            passed = (
                metrics["max_complex_error"] < 2e-5 and metrics["transmission_peak_ratio"] < 1e-10
            )
            # Legacy PMC is a measured comparison, not an acceptance target.
            save(
                "reflection_cases",
                dict(
                    name=name,
                    boundary=boundary,
                    dt_s=dt,
                    **metrics,
                    passes_face_pmc_criteria=passed,
                    expected_control=boundary == "legacy",
                ),
            )
            write_csv(
                output / f"reflection_{name}.csv",
                ["frequency_hz", "gamma_real", "gamma_imag", "phase_deg"],
                zip(frequency, gamma.real, gamma.imag, np.rad2deg(np.angle(gamma))),
            )
        if axis == 0 and e_axis == 2 and sign == 1:
            for resistance in (1e3, 1e5, 1e7):
                total, _ = plane_trace(cache, axis, e_axis, sign, "finite", resistance=resistance)
                _, gamma, expected, _, metrics = reflection(incident, total, dt, resistance)
                save(
                    "reflection_cases",
                    dict(
                        name=f"finite_{resistance:g}",
                        boundary="finite",
                        resistance_ohm=resistance,
                        **metrics,
                        passed=metrics["max_complex_error"] < 2e-5,
                        distance_to_pmc=float(np.max(abs(gamma - 1))),
                    ),
                )
                write_csv(
                    output / f"finite_{resistance:g}.csv",
                    ["frequency_hz", "gamma_real", "gamma_imag", "expected_real"],
                    zip(frequency, gamma.real, gamma.imag, expected.real),
                )
    for indices, precision in [
        ((1, 1, 0), "double"),
        ((1, 2, 1), "double"),
        ((2, 1, 2), "double"),
        ((1, 2, 1), "single"),
    ]:
        result, samples = cavity_mode(cache, indices, precision=precision)
        save("cavity_modes", result)
        name = "".join(map(str, indices)) + "_" + precision
        write_csv(
            output / f"mode_{name}.csv",
            ["step", "electric_projection", "discrete_pmc_theory"],
            samples,
        )
    for geometry in ("cavity", "stair"):
        for precision in ("double", "single"):
            with exact_pmc():
                result = full_solver_run(
                    cache / f"stability_{geometry}_{precision}",
                    geometry=geometry,
                    resistance=50,
                    precision=precision,
                    steps=args.steps,
                )
                grid, occupied = build_case(
                    cache / f"energy_{geometry}_{precision}",
                    geometry=geometry,
                    resistance=50,
                    precision=precision,
                )
            assert_zero_admittance(grid)
            stepper = Stepper(grid, occupied)
            geometry_metrics, q = audit_geometry(stepper)
            energy = source_free_run(stepper, q, steps=min(args.steps, 20000))
            samples = result["source_free"].pop("norm_samples")
            energy.pop("state_norm_samples")
            result["parameters"][
                "boundary"
            ] = "exact PMC; resistance is only a declaration placeholder"
            drift = abs(energy["final_modified_field_energy_over_initial"] - 1)
            passed = (
                result["source_free"]["finite"]
                and result["source_free"]["peak_state_norm_over_initial"] < 10
                and drift < (0.01 if precision == "single" else 1e-8)
            )
            save(
                "stability_cases",
                dict(
                    geometry=geometry,
                    precision=precision,
                    **result,
                    geometry_metrics=geometry_metrics,
                    energy=energy,
                    passed=passed,
                ),
            )
            write_csv(
                output / f"stability_{geometry}_{precision}.csv",
                ["step", "field_norm_over_initial"],
                samples,
            )
    summary["passed"] = all(
        c["passed"]
        for key in ("image_cases", "cavity_modes", "stability_cases")
        for c in summary[key]
    ) and all(
        c.get("passed", c.get("passes_face_pmc_criteria", False))
        for c in summary["reflection_cases"]
        if not c.get("expected_control", False)
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    plot_results(output)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
