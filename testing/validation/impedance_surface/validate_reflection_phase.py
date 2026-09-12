"""Validate the complex reflection of a TEM plane wave from a metal wall.

A uniform Ez current sheet, PEC z faces and PMC y faces produce a transverse
uniform wave. Subtract an identical no-wall run and de-embed to the wall.
The Debye case places dispersive material directly against the SIBC.
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from time import perf_counter

import h5py
import matplotlib
import numpy as np
from scipy.constants import c, mu_0

import gprMax
from gprMax.impedance_surfaces import MAX_SIBC_TIMESTEP_FACTOR, SurfaceImpedanceModel

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from ._wall_waveguide_common import validation_cache_stem
from .analytical import discrete_normal_reflection, normal_reflection

FREQUENCIES = np.arange(1e9, 8e9 + 0.125e9, 0.25e9)
HOSTS = {
    "air": dict(er_inf=1.0, delta_er=0.0, tau=80e-12),
    "debye": dict(er_inf=2.5, delta_er=2.0, tau=80e-12),
}
DOMAIN_X, SOURCE_X, RECEIVER_X, WALL_X = 1.8, 0.8, 0.86, 0.89
LIMITS = dict(
    continuum_phase_rms_deg=0.1,
    continuum_magnitude_rms=0.002,
    discrete_phase_rms_deg=0.0002,
    discrete_complex_relative_l2=1e-5,
    transverse_uniformity_relative_l2=1e-9,
    incident_min_relative_amplitude=0.001,
    final_tenth_peak_relative=1e-5,
)


def build_scene(dl, host, conductivity, threads, duration):
    # The outer PEC faces are outside the measurement's return-time window.
    # This avoids relying on PML accuracy at the transverse symmetry corners.
    width = 4 * dl
    scene = gprMax.Scene()
    for obj in (
        gprMax.Discretisation(p1=(dl,) * 3),
        gprMax.Domain(p1=(DOMAIN_X, width, width)),
        # Match the incident reference to the automatically capped wall run.
        gprMax.TimeStepStabilityFactor(f=MAX_SIBC_TIMESTEP_FACTOR),
        gprMax.TimeWindow(time=duration),
        gprMax.OMPThreads(threads),
        gprMax.PMLThickness(thickness=0),
        gprMax.Waveform(wave_type="ricker", amp=1, freq=4e9, id="pulse"),
    ):
        scene.add(obj)
    for face, kind in (("y0", "pmc"), ("ymax", "pmc"), ("z0", "pec"), ("zmax", "pec")):
        scene.add(gprMax.SymmetryBoundary(face=face, type=kind))
    if host == "debye":
        scene.add(gprMax.Material(er=2.5, se=0, mr=1, sm=0, id="host"))
        scene.add(
            gprMax.AddDebyeDispersion(poles=1, er_delta=[2.0], tau=[80e-12], material_ids=["host"])
        )
        scene.add(
            gprMax.Box(p1=(0, 0, 0), p2=(DOMAIN_X, width, width), material_id="host", averaging="n")
        )
    wall = None
    if conductivity is not None:
        wall = gprMax.SurfaceImpedance(
            id="metal",
            conductivity=conductivity,
            fit_frequency_range=(0.5e9, 12e9),
            fit_order="auto",
            fit_tolerance=0.001,
        )
        scene.add(wall)
        scene.add(
            gprMax.Box(
                p1=(WALL_X, 0, 0),
                p2=(WALL_X + 0.004, width, width),
                material_id="metal",
                averaging="n",
            )
        )
    for j in range(5):
        for k in range(4):
            scene.add(gprMax.HertzianDipole((SOURCE_X, j * dl, k * dl), "z", "pulse"))
    for j, k in ((2, 2), (1, 1)):
        scene.add(gprMax.Rx(p1=(RECEIVER_X, j * dl, k * dl)))
    return scene, wall


def trace(output, dl, host, conductivity, args):
    config = dict(
        version=2,
        dl=dl,
        host=host,
        host_parameters=HOSTS[host],
        conductivity=conductivity,
        duration=args.duration,
        domain_x=DOMAIN_X,
        transverse_cells=4,
        pml_x=0,
        source_x=SOURCE_X,
        receiver_x=RECEIVER_X,
        wall=(WALL_X, WALL_X + 0.004),
        pulse_hz=4e9,
        fit_band=(0.5e9, 12e9),
        fit_tolerance=0.001,
        precision="double",
        timestep_factor=MAX_SIBC_TIMESTEP_FACTOR,
    )
    cache = output / "_cache" / validation_cache_stem("reflection", config)
    scene, wall = build_scene(dl, host, conductivity, args.threads, args.duration)
    if not (args.reuse and cache.with_suffix(".h5").exists()):
        gprMax.run(
            scenes=[scene],
            outputfile=cache,
            hide_progress_bars=True,
            log_level=logging.WARNING,
            cpu_precision="double",
        )
    with h5py.File(cache.with_suffix(".h5"), "r") as handle:
        dt = float(handle.attrs["dt"])
        electric = np.array([handle[f"rxs/rx{i}/Ez"][:] for i in (1, 2)])
    return electric, dt, wall


def analyse(incident, total, dt, dl, host, wall):
    time = np.arange(incident.shape[-1]) * dt
    transform = np.exp(-2j * np.pi * FREQUENCIES[:, None] * time)
    inc = transform @ incident[0]
    reflected = transform @ (total[0] - incident[0])
    exact_z = (1 + 1j) * np.sqrt(np.pi * FREQUENCIES * mu_0 / wall.conductivity)
    model = SurfaceImpedanceModel("metal", A=wall.A, B=wall.B, C=wall.C, D=wall.D)
    warped = np.tan(np.pi * FREQUENCIES * dt) / (np.pi * dt)
    algorithm_z = model.impedance(warped)
    discrete, k = discrete_normal_reflection(FREQUENCIES, dt, dl, algorithm_z, **HOSTS[host])
    p = HOSTS[host]
    er = p["er_inf"] + p["delta_er"] / (1 + 2j * np.pi * FREQUENCIES * p["tau"])
    continuum = normal_reflection(exact_z, er)
    measured = reflected / inc * np.exp(2j * k * (WALL_X - RECEIVER_X))
    phase_error = np.rad2deg(np.angle(measured / continuum))
    discrete_phase_error = np.rad2deg(np.angle(measured / discrete))
    metrics = dict(
        continuum_phase_rms_deg=float(np.sqrt(np.mean(phase_error**2))),
        continuum_magnitude_rms=float(
            np.sqrt(np.mean((np.abs(measured) - np.abs(continuum)) ** 2))
        ),
        discrete_phase_rms_deg=float(np.sqrt(np.mean(discrete_phase_error**2))),
        discrete_complex_relative_l2=float(
            np.linalg.norm(measured - discrete) / np.linalg.norm(discrete)
        ),
        transverse_uniformity_relative_l2=float(
            np.linalg.norm(total[0] - total[1]) / np.linalg.norm(total[0])
        ),
        incident_min_relative_amplitude=float(np.min(np.abs(inc)) / np.max(np.abs(inc))),
        final_tenth_peak_relative=float(
            max(
                np.max(np.abs(v[:, -v.shape[1] // 10 :])) / np.max(np.abs(v))
                for v in (incident, total)
            )
        ),
    )
    passed = all(
        value >= LIMITS[key] if key == "incident_min_relative_amplitude" else value <= LIMITS[key]
        for key, value in metrics.items()
    )
    return measured, continuum, discrete, metrics, passed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--mesh-mm", type=float, nargs="+", default=[1.0, 0.5])
    parser.add_argument("--conductivity", type=float, nargs="+", default=[1e3, 5.8e7])
    parser.add_argument("--host", choices=tuple(HOSTS), nargs="+", default=list(HOSTS))
    parser.add_argument("--duration", type=float, default=4e-9)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent / "results/reflection_phase"
    )
    args = parser.parse_args(argv)
    earliest_return = min(SOURCE_X + RECEIVER_X, 2 * DOMAIN_X - SOURCE_X - RECEIVER_X) / c
    if not np.isfinite(args.duration) or not 0 < args.duration < earliest_return:
        parser.error(
            f"duration must be positive and shorter than the earliest outer return ({earliest_return:g} s)"
        )
    for mesh in args.mesh_mm:
        if not np.isfinite(mesh) or mesh <= 0:
            parser.error("mesh spacings must be positive and finite")
        coordinates = np.array([DOMAIN_X, SOURCE_X, RECEIVER_X, WALL_X, WALL_X + 0.004]) / (
            mesh * 1e-3
        )
        if not np.allclose(coordinates, np.round(coordinates), atol=1e-8, rtol=0):
            parser.error(
                "mesh must place the source, receivers, wall and domain endpoints on grid nodes"
            )
    output = args.output_dir
    (output / "_cache").mkdir(parents=True, exist_ok=True)
    cases, curves = [], []
    for host in args.host:
        for mesh in sorted(set(args.mesh_mm), reverse=True):
            incident, dt, _ = trace(output, mesh * 1e-3, host, None, args)
            for conductivity in args.conductivity:
                start = perf_counter()
                total, total_dt, wall = trace(output, mesh * 1e-3, host, conductivity, args)
                if total_dt != dt or total.shape != incident.shape:
                    raise ValueError("Reference and wall records must share a time grid")
                measured, continuum, discrete, metrics, passed = analyse(
                    incident, total, dt, mesh * 1e-3, host, wall
                )
                name = f"{host}_sigma_{conductivity:g}_dl_{mesh:g}mm"
                case = dict(
                    name=name,
                    host=host,
                    conductivity_s_per_m=conductivity,
                    dl_m=mesh * 1e-3,
                    dt_s=dt,
                    metrics=metrics,
                    passed=passed,
                    elapsed_seconds=perf_counter() - start,
                    fit_poles=wall.fit_pole_count,
                    fit_max_relative_error=wall.fit_max_relative_error,
                )
                cases.append(case)
                curves.append((measured, continuum, discrete))
                print(json.dumps(case), flush=True)
                with (output / f"{name}.csv").open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(
                        (
                            "frequency_hz",
                            "measured_real",
                            "measured_imag",
                            "continuum_real",
                            "continuum_imag",
                            "discrete_real",
                            "discrete_imag",
                            "measured_phase_deg",
                            "continuum_phase_deg",
                            "discrete_phase_deg",
                        )
                    )
                    writer.writerows(
                        zip(
                            FREQUENCIES,
                            measured.real,
                            measured.imag,
                            continuum.real,
                            continuum.imag,
                            discrete.real,
                            discrete.imag,
                            *[
                                180 + np.rad2deg(np.angle(-v))
                                for v in (measured, continuum, discrete)
                            ],
                        )
                    )
    fig, axes = plt.subplots(3, len(args.host), figsize=(6 * len(args.host), 10), squeeze=False)
    for column, host in enumerate(args.host):
        for i, case in enumerate(cases):
            if case["host"] != host or not np.isclose(case["dl_m"], min(args.mesh_mm) * 1e-3):
                continue
            measured, continuum, discrete = curves[i]
            label = f"σ={case['conductivity_s_per_m']:g} S/m"
            (line,) = axes[0, column].plot(
                FREQUENCIES / 1e9, 180 + np.rad2deg(np.angle(-measured)), ".", label=f"FDTD {label}"
            )
            axes[0, column].plot(
                FREQUENCIES / 1e9,
                180 + np.rad2deg(np.angle(-continuum)),
                color=line.get_color(),
                label=f"Analytical {label}",
            )
            axes[1, column].plot(FREQUENCIES / 1e9, np.abs(measured), ".", color=line.get_color())
            axes[1, column].plot(
                FREQUENCIES / 1e9, np.abs(continuum), color=line.get_color(), label=label
            )
            axes[2, column].plot(
                FREQUENCIES / 1e9,
                np.rad2deg(np.angle(measured / continuum)),
                color=line.get_color(),
                label=f"Continuum {label}",
            )
            axes[2, column].plot(
                FREQUENCIES / 1e9,
                np.rad2deg(np.angle(measured / discrete)),
                "--",
                color=line.get_color(),
                label=f"Discrete {label}",
            )
        axes[0, column].set(
            title=f"{host.capitalize()} exterior; phase at wall",
            ylabel="Reflection phase (degrees)",
        )
        axes[1, column].set(ylabel="Reflection magnitude")
        axes[2, column].set(ylabel="Phase error (degrees)", xlabel="Frequency (GHz)")
    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "reflection_phase.png", dpi=160)
    plt.close(fig)
    summary = dict(
        passed=all(case["passed"] for case in cases),
        acceptance_limits=LIMITS,
        cases=cases,
        duration_s=args.duration,
        frequency_band_hz=[float(FREQUENCIES[0]), float(FREQUENCIES[-1])],
        host_parameters={host: HOSTS[host] for host in args.host},
        earliest_outer_return_s=earliest_return,
        outer_return_margin_s=earliest_return - args.duration,
        domain_x_m=DOMAIN_X,
        source_x_m=SOURCE_X,
        receiver_x_m=RECEIVER_X,
        wall_x_m=WALL_X,
        cpu_precision="double",
        threads=args.threads,
        fit_band_hz=[0.5e9, 12e9],
        deembedding="30 mm to the physical wall using independently derived discrete host wavenumber",
        reference="Gamma=(Zs-eta)/(Zs+eta), exp(+j omega t); phase shown continuously near 180 degrees",
        limitations="Normal incidence, homogeneous air or single-Debye exterior, planar wall; this is not a curved-interface dispersion validation.",
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
