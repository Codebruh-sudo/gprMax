"""Broadband conductor-sphere RCS and complex angular scattering validation.

Run as a module from the repository root. CPU only; --reuse reads compatible
local solver caches. Acceptance applies to the finest requested mesh.
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from time import perf_counter

import gprMax
import matplotlib
import numpy as np
from scipy.constants import mu_0
from gprMax.impedance_surfaces import SurfaceImpedanceModel

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from ._wall_waveguide_common import validation_cache_stem
from .analytical import sphere_reference

FREQUENCIES = np.arange(2e9, 7e9 + 0.125e9, 0.25e9)
ANGLES = np.arange(0., 181., 5.)
RADIUS = 0.016
LIMITS = dict(backscatter_rms_db=1.0, backscatter_max_db=2.0,
              complex_pattern_relative_l2=0.15, impedance_fit_relative_max=0.002,
              sibc_bulk_complex_relative_l2=0.02)


def build_scene(dl, conductivity, threads, duration):
    scene = gprMax.Scene()
    wall = gprMax.SurfaceImpedance(id="metal", conductivity=conductivity,
                                 fit_frequency_range=(1e9, 10e9), fit_order="auto", fit_tolerance=0.002)
    transform = gprMax.NTFFFrequencyTransform(surface_id="surface", id="spectrum", frequencies=FREQUENCIES,
                                             window="rectangular", save_surface_dft=False, plane_wave_index=0)
    far_field = gprMax.NTFFFarField(theta=np.full(ANGLES.shape, 90.), phi=ANGLES, transform_id="spectrum",
                                   id="pattern", outputs=("Etheta", "Ephi", "rcs"))
    for obj in (
        gprMax.Discretisation(p1=(dl,)*3), gprMax.Domain(p1=(0.096,)*3),
        gprMax.TimeWindow(time=duration), gprMax.OMPThreads(threads),
        gprMax.PMLThickness(thickness=8), wall,
        gprMax.Waveform(wave_type="ricker", amp=1, freq=4.5e9, id="pulse"),
        gprMax.Sphere(p1=(0.048,)*3, r=RADIUS, material_id="metal"),
        gprMax.DiscretePlaneWaveVector(p1=(0.024,)*3, p2=(0.072,)*3,
                                      m_vec=(1, 0, 0), psi=90, waveform_id="pulse"),
        gprMax.NTFFSurface(p1=(0.018,)*3, p2=(0.078,)*3, id="surface", origin=(0.048,)*3),
        transform, far_field,
    ):
        scene.add(obj)
    return scene, wall, transform, far_field


def run_case(output, dl, conductivity, args):
    configuration = dict(version=1, dl=dl, conductivity=conductivity, duration=args.duration,
                         frequencies=FREQUENCIES, angles=ANGLES, radius=RADIUS, precision="double",
                         domain=0.096, tfsf=(0.024, 0.072), ntff=(0.018, 0.078), pml=8,
                         pulse_hz=4.5e9, fit_band=(1e9, 10e9), fit_tolerance=0.002)
    cache = output / "_cache" / validation_cache_stem("sphere", configuration)
    scene, wall, transform, far_field = build_scene(dl, conductivity, args.threads, args.duration)
    start = perf_counter()
    reused = args.reuse and cache.with_suffix(".npz").exists()
    if not reused:
        gprMax.run(scenes=[scene], outputfile=cache, hide_progress_bars=True,
                   log_level=logging.WARNING, cpu_precision="double")
        monitor = transform._compiled_outputs.transform_monitor(transform.ID)
        incident = monitor.result.incident_electric[:, 2]
        np.savez(cache.with_suffix(".npz"), amplitude=far_field.result.fields["Etheta"]/incident[:, None],
                 rcs=far_field.result.fields["rcs"])
    with np.load(cache.with_suffix(".npz")) as handle:
        numerical, rcs = handle["amplitude"], handle["rcs"]
    exact_z = (1+1j)*np.sqrt(np.pi*FREQUENCIES*mu_0/conductivity)
    fitted_z = SurfaceImpedanceModel("metal", A=wall.A, B=wall.B, C=wall.C, D=wall.D).impedance(FREQUENCIES)
    sibc_rcs, sibc_amp = sphere_reference(FREQUENCIES, RADIUS, conductivity, np.deg2rad(ANGLES), impedance=exact_z)
    bulk_rcs, bulk_amp = sphere_reference(FREQUENCIES, RADIUS, conductivity, np.deg2rad(ANGLES))
    error_db = 10*np.log10(rcs[:, -1]/sibc_rcs[:, -1])
    metrics = dict(backscatter_rms_db=float(np.sqrt(np.mean(error_db**2))),
                   backscatter_max_db=float(np.max(np.abs(error_db))),
                   complex_pattern_relative_l2=float(np.linalg.norm(numerical-sibc_amp)/np.linalg.norm(sibc_amp)),
                   impedance_fit_relative_max=float(np.max(np.abs(fitted_z/exact_z-1))),
                   sibc_bulk_complex_relative_l2=float(np.linalg.norm(sibc_amp-bulk_amp)/np.linalg.norm(bulk_amp)))
    name = f"sigma_{conductivity:g}_dl_{dl*1e3:g}mm"
    with (output / f"{name}.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("frequency_hz", "angle_deg", "fdtd_rcs_m2", "sibc_mie_rcs_m2", "bulk_mie_rcs_m2",
                         "fdtd_amplitude_real_m", "fdtd_amplitude_imag_m", "sibc_amplitude_real_m",
                         "sibc_amplitude_imag_m", "bulk_amplitude_real_m", "bulk_amplitude_imag_m"))
        for i, frequency in enumerate(FREQUENCIES):
            for j, angle in enumerate(ANGLES):
                writer.writerow((frequency, angle, rcs[i,j], sibc_rcs[i,j], bulk_rcs[i,j],
                                 numerical[i,j].real, numerical[i,j].imag, sibc_amp[i,j].real,
                                 sibc_amp[i,j].imag, bulk_amp[i,j].real, bulk_amp[i,j].imag))
    return dict(name=name, conductivity_s_per_m=conductivity, dl_m=dl, metrics=metrics,
                passed=all(value <= LIMITS[key] for key, value in metrics.items()),
                fit_poles=wall.fit_pole_count,
                maximum_skin_depth_radius_ratio=float(1/(RADIUS*np.sqrt(np.pi*FREQUENCIES[0]*mu_0*conductivity))),
                elapsed_seconds=perf_counter()-start, reused=reused), (rcs, sibc_rcs, bulk_rcs, numerical, sibc_amp)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--mesh-mm", type=float, nargs="+", default=[1.5, 0.75])
    parser.add_argument("--conductivity", type=float, nargs="+", default=[1e3, 5.8e7])
    parser.add_argument("--duration", type=float, default=4e-9)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results/conductor_sphere")
    args = parser.parse_args(argv)
    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("duration must be positive and finite")
    for mesh in args.mesh_mm:
        if not np.isfinite(mesh) or mesh <= 0:
            parser.error("mesh spacings must be positive and finite")
        coordinates = np.array([0.096, 0.048, 0.024, 0.072, 0.018, 0.078])/(mesh*1e-3)
        if not np.allclose(coordinates, np.round(coordinates), atol=1e-8, rtol=0):
            parser.error("mesh must place domain, TFSF and NTFF boundaries and sphere centre on grid nodes")
    output = args.output_dir
    (output / "_cache").mkdir(parents=True, exist_ok=True)
    cases, curves = [], []
    for conductivity in args.conductivity:
        for mesh in sorted(set(args.mesh_mm), reverse=True):
            case, curve = run_case(output, mesh*1e-3, conductivity, args)
            cases.append(case)
            curves.append(curve)
            print(json.dumps(case), flush=True)
    fig, axes = plt.subplots(2, len(args.conductivity), figsize=(6*len(args.conductivity), 8), squeeze=False)
    middle = len(FREQUENCIES)//2
    for column, conductivity in enumerate(args.conductivity):
        selected = [i for i, case in enumerate(cases) if case["conductivity_s_per_m"] == conductivity]
        for i in selected:
            rcs, sibc, bulk, numerical, amplitude = curves[i]
            label = f"FDTD {cases[i]['dl_m']*1e3:g} mm"
            axes[0,column].plot(FREQUENCIES/1e9, 10*np.log10(rcs[:,-1]), ".-", label=label)
            axes[1,column].plot(ANGLES, 10*np.log10(rcs[middle]), ".-", label=label)
        axes[0,column].plot(FREQUENCIES/1e9, 10*np.log10(sibc[:,-1]), "k--", label="Impedance Mie")
        axes[0,column].plot(FREQUENCIES/1e9, 10*np.log10(bulk[:,-1]), "r:", label="Bulk conductor Mie")
        axes[1,column].plot(ANGLES, 10*np.log10(sibc[middle]), "k--", label="Impedance Mie")
        axes[1,column].plot(ANGLES, 10*np.log10(bulk[middle]), "r:", label="Bulk conductor Mie")
        axes[0,column].set(title=f"Conductivity {conductivity:g} S/m; backscatter", xlabel="Frequency (GHz)", ylabel="RCS (dB m²)")
        axes[1,column].set(title=f"Angular pattern at {FREQUENCIES[middle]/1e9:g} GHz", xlabel="Scattering angle (degrees)", ylabel="RCS (dB m²)")
    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output / "conductor_sphere.png", dpi=160)
    plt.close(fig)
    finest = [case for case in cases if np.isclose(case["dl_m"], min(args.mesh_mm)*1e-3)]
    convergence = {}
    for conductivity in args.conductivity:
        selected = [case for case in cases if case["conductivity_s_per_m"] == conductivity]
        if len(selected) > 1:
            convergence[str(conductivity)] = all(selected[-1]["metrics"][key] < selected[0]["metrics"][key]
                                                for key in ("backscatter_rms_db", "complex_pattern_relative_l2"))
    summary = dict(acceptance_limits=LIMITS, acceptance_scope="Finest mesh, each conductivity; coarse mesh is diagnostic",
                   passed=all(case["passed"] for case in finest) and all(convergence.values()), cases=cases,
                   coarse_to_fine_error_decreased=convergence, cpu_precision="double", threads=args.threads,
                   fit_band_hz=[1e9, 10e9], angles_deg=ANGLES.tolist(),
                   radius_m=RADIUS, duration_s=args.duration, frequency_band_hz=[float(FREQUENCIES[0]), float(FREQUENCIES[-1])],
                   reference="https://doi.org/10.1103/PhysRevB.98.235417",
                   limitations="Staircased sphere; no skin-depth cells. Copper is nearly PEC. The planar phase validation resolves finite surface impedance directly.")
    (output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
