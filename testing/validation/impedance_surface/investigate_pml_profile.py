"""Diagnose the duplicated unshifted HORIPML profile and verify a repair.

The Fourier diagnostic uses the native recursive coefficients. The full-grid
experiments use the public scene API and production CPU kernels. No production
PML parameters are silently changed by this driver.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.constants import c, epsilon_0

import gprMax
from gprMax.updates.cpu_updates import CPUUpdates

from .validate_sibc_pml import (
    DL,
    ETA,
    FIELD_NAMES,
    advance,
    build_grid,
    field_norm,
    scene_for,
    seed_pulse,
)

OUTPUT = Path(__file__).parent / "results" / "pml_profile_investigation"
SIGMA_MAX = 4 / (ETA * DL)
# CODATA epsilon0/mu0 and the exact c can differ in their last supplied
# digits. The field coefficient tables use epsilon0 and mu0 directly.
WAVE_SPEED = 1 / (ETA * epsilon_0)


def coefficients(alpha, kappa, sigma, dt):
    """One native HORIPML inverse-stretch factor, in conductivity units."""
    denominator = 2 * epsilon_0 * kappa + dt * (alpha * kappa + sigma)
    return np.asarray(
        (
            (2 * epsilon_0 + dt * alpha) / denominator,
            2 * epsilon_0 * kappa / denominator,
            (2 * epsilon_0 * kappa - dt * (alpha * kappa + sigma)) / denominator,
            2 * sigma * dt / (kappa * denominator),
        )
    )


def filtered_derivative(value, history, coeffs):
    """Cascade the native old-history outputs and history updates."""
    updated = np.empty_like(history)
    for index, (a, b, r, f) in enumerate(coeffs):
        updated[index] = r * history[index] - f * value
        value = a * value + b * history[index]
    return value, updated


def amplification_matrix(dt, transverse_k, normal_k, alpha, sigma):
    """Frozen uniform TE Yee/PML symbol: Ey,Hx,Hz and two history banks.

    H is normalized by eta0 and histories by c*dt. Spatial Fourier phases
    absorb the staggered half-cell factors, leaving derivative i*K. This
    is an exact *local constant-coefficient* symbol, not the spectrum of the
    complete graded guide with its outer boundary.
    """
    coeffs = np.asarray([coefficients(a, 1.0, s, dt) for a, s in zip(alpha, sigma)])
    order = len(coeffs)

    def step(state):
        e, hx, hz = state[:3]
        h_curl, h_state = filtered_derivative(
            1j * WAVE_SPEED * dt * normal_k * e, state[3 : 3 + order], coeffs
        )
        hx += h_curl
        hz -= 1j * WAVE_SPEED * dt * transverse_k * e
        e_curl, e_state = filtered_derivative(
            1j * WAVE_SPEED * dt * normal_k * hx, state[3 + order :], coeffs
        )
        e += e_curl - 1j * WAVE_SPEED * dt * transverse_k * hz
        return np.r_[e, hx, hz, h_state, e_state]

    identity = np.eye(3 + 2 * order, dtype=complex)
    return np.column_stack([step(column) for column in identity.T])


def inverse_stretch(z, coeffs):
    return np.prod([a - b * f / (z - r) for a, b, r, f in coeffs], axis=0)


def analytic_diagnostic():
    dt = 0.99 * DL / (c * np.sqrt(3))
    sigma = 2.0  # Representative interior PML value, S/m.
    transverse_k = 2 * np.sin(np.pi / 16) / DL  # Discrete TE10 cutoff, 8 cells.
    normal_k = transverse_k
    gamma = sigma / epsilon_0
    # Scale continuous Laplace s by gamma to avoid ill-conditioned SI powers.
    a, b = WAVE_SPEED * transverse_k / gamma, WAVE_SPEED * normal_k / gamma
    polynomial = np.polymul([1.0, 0.0, a * a], [1.0, 4.0, 6.0, 4.0, 1.0])
    polynomial[2] += b * b  # (u^2+a^2)(u+1)^4 + b^2*u^4 = 0.
    roots = np.roots(polynomial) * gamma
    root = roots[np.argmax(roots.real)]
    rows = []
    for fraction in (1.0, 0.5, 0.25, 0.125):
        for profile, alpha, sigma_values in (
            ("duplicated_unshifted", [0.0, 0.0], [sigma, sigma]),
            ("shifted_second_1p1", [0.0, 1.1 * sigma], [sigma, sigma]),
            ("first_order", [0.0], [sigma]),
        ):
            h = dt * fraction
            eig = np.linalg.eigvals(
                amplification_matrix(h, transverse_k, normal_k, alpha, sigma_values)
            )
            dominant = eig[np.argmax(abs(eig))]
            rows.append(
                dict(
                    profile=profile,
                    dt_s=h,
                    dt_ratio=fraction,
                    spectral_radius=float(abs(dominant)),
                    amplitude_growth_per_s=float(np.log(abs(dominant)) / h),
                    eigenvalue=[float(dominant.real), float(dominant.imag)],
                )
            )
    theta = np.linspace(0.0001, np.pi - 0.0001, 2048)
    z = np.exp(1j * theta)
    q = 2 * epsilon_0 / dt * (z - 1) / (z + 1)
    coeffs = [coefficients(0.0, 1.0, sigma, dt)] * 2
    expected = (q / (q + sigma)) ** 2
    transfer_error = np.max(abs(inverse_stretch(z, coeffs) - expected))
    cancelled = [coefficients(0.0, 1.0, sigma, dt), coefficients(sigma, 1.0, sigma, dt)]
    single = [coefficients(0.0, 1.0, 2 * sigma, dt)]
    cancellation_error = np.max(abs(inverse_stretch(z, cancelled) - inverse_stretch(z, single)))
    return dict(
        sigma_s_per_m=sigma,
        sigma_max_s_per_m=SIGMA_MAX,
        continuum_dominant_root_per_s=[float(root.real), float(root.imag)],
        continuum_amplitude_doubling_time_s=float(np.log(2) / root.real),
        negative_real_stretch_below_hz=float(sigma / (2 * np.pi * epsilon_0)),
        native_transfer_error=float(transfer_error),
        exact_shift_cancellation_error=float(cancellation_error),
        fourier_cases=rows,
    )


def repaired_scene(length, kind, shift_scale, timestep=0.99, first_order=False):
    """Use explicit matching quartic sigma/alpha grading on both E/H samples."""
    scene = scene_for(length, kind=kind)
    scene.single_use_objects = [
        obj
        for obj in scene.single_use_objects
        if not isinstance(obj, gprMax.TimeStepStabilityFactor)
    ]
    scene.add(gprMax.TimeStepStabilityFactor(f=timestep))
    for order in range(1 if first_order else 2):
        scene.add(
            gprMax.PMLCFS(
                alphascalingprofile="quartic",
                alphascalingdirection="forward",
                alphamin=0.0,
                alphamax=0.0 if order == 0 else shift_scale * SIGMA_MAX,
                kappascalingprofile="constant",
                kappascalingdirection="forward",
                kappamin=1.0,
                kappamax=1.0,
                sigmascalingprofile="quartic",
                sigmascalingdirection="forward",
                sigmamin=0.0,
                sigmamax=SIGMA_MAX,
            )
        )
    return scene


def simulate(
    output,
    *,
    kind,
    precision="double",
    shift_scale=1.1,
    timestep=0.99,
    steps=20000,
    length=72,
    random_seed=True,
):
    key = f"{kind}_{precision}_alpha2_{shift_scale:g}_dt_{timestep:g}_n{length}"
    grid = build_grid(
        repaired_scene(length, kind, shift_scale, timestep), output / "_cache" / key, precision
    )
    seed_pulse(grid)
    if random_seed:
        rng = np.random.default_rng(718)
        interior = (slice(5, 12), slice(5, 8), slice(18, 54))
        for index, name in enumerate(FIELD_NAMES):
            field = getattr(grid, name)
            field[interior] += (
                rng.standard_normal(field[interior].shape) * 1e-3 / (1 if index < 3 else ETA)
            )
    updates = CPUUpdates(grid)
    initial = field_norm(grid)
    records, trace = [], []
    for step in range(steps):
        trace.append(float(grid.Ey[8, 6, 64 if length >= 120 else 40]))
        advance(updates)
        if step % 50 == 0 or step == steps - 1:
            norm = field_norm(grid) / initial
            records.append(((step + 1) * grid.dt, norm))
            if not np.isfinite(norm) or norm > 1e10:
                break
    records = np.asarray(records)
    completed = step + 1
    slope = (
        np.polyfit(
            records[-min(20, len(records)) :, 0], np.log(records[-min(20, len(records)) :, 1]), 1
        )[0]
        / 2
    )
    finite = all(np.all(np.isfinite(getattr(grid, name))) for name in FIELD_NAMES)
    finite &= all(
        np.all(np.isfinite(getattr(slab, name)))
        for slab in grid.pmls["slabs"]
        for name in ("EPhi1", "EPhi2", "HPhi1", "HPhi2")
    )
    if grid.impedance_surfaces is not None:
        finite &= np.all(np.isfinite(grid.impedance_surfaces.state_y))
    result = dict(
        key=key,
        kind=kind,
        precision=precision,
        shift_scale=shift_scale,
        timestep_factor=timestep,
        dt_s=grid.dt,
        steps=completed,
        requested_steps=steps,
        duration_s=completed * grid.dt,
        final_squared_norm=float(records[-1, 1]),
        peak_squared_norm=float(max(records[:, 1])),
        finite=bool(finite),
        fitted_late_amplitude_growth_per_s=float(slope),
        bounded=bool(finite and completed == steps and max(records[:, 1]) < 2),
    )
    np.savetxt(
        output / f"{key}.csv",
        records,
        delimiter=",",
        header="time_s,squared_field_norm",
        comments="",
    )
    return result, np.asarray(trace), grid.dt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--analytic-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / ".gitignore").write_text("_cache/\n")
    summary = args.output / "summary.json"
    report = (
        json.loads(summary.read_text())
        if args.analytic_only and summary.exists()
        else {"runtime": [], "pulse": []}
    )
    report["analytic"] = analytic_diagnostic()
    if not args.analytic_only:
        cases = [
            dict(
                kind="pec",
                shift_scale=0.0,
                timestep=factor,
                steps=int(np.ceil(4500 * 0.99 / factor)),
            )
            for factor in (0.99, 0.495, 0.2475)
        ]
        cases += [
            dict(kind=kind, precision=precision)
            for kind, precision in (
                ("pec", "double"),
                ("resistance", "double"),
                ("foster", "double"),
                ("foster", "single"),
            )
        ]
        for case in cases:
            try:
                result, _, _ = simulate(args.output, **case)
            except ValueError as exc:
                if case.get("shift_scale") != 0 or "negative real total stretch" not in str(exc):
                    raise
                # The public solver now rejects this setting. Preserve the
                # independent continuous/discrete symbol reproduction above;
                # no production validation bypass is provided.
                result = dict(case, bounded=False, build_rejected=True, reason=str(exc))
            report["runtime"].append(result)
            print(json.dumps(result), flush=True)
        for kind in ("resistance", "foster"):
            short, trace, dt = simulate(
                args.output, kind=kind, steps=600, length=120, random_seed=False
            )
            _, reference, _ = simulate(
                args.output, kind=kind, steps=600, length=320, random_seed=False
            )
            result = dict(
                kind=kind,
                peak_relative_difference=float(max(abs(trace - reference)) / max(abs(reference))),
                dt_s=dt,
                record_s=len(trace) * dt,
            )
            report["pulse"].append(result)
            np.savetxt(
                args.output / f"pulse_{kind}.csv",
                np.column_stack((np.arange(len(trace)) * dt, trace, reference)),
                delimiter=",",
                header="time_s,repaired_pml_E,long_reference_E",
                comments="",
            )
            print(json.dumps(result), flush=True)
    analytic = report["analytic"]
    report["checks"] = dict(
        native_transfer=analytic["native_transfer_error"] < 2e-14,
        pole_cancellation=analytic["exact_shift_cancellation_error"] < 2e-14,
        continuum_growth=analytic["continuum_dominant_root_per_s"][0] > 2e9,
    )
    if report["runtime"]:
        bad = [row for row in report["runtime"] if row["shift_scale"] == 0]
        fixed = [row for row in report["runtime"] if row["shift_scale"] > 0]
        report["checks"].update(
            reduced_dt_does_not_fix=bool(len(bad) == 3 and all(not row["bounded"] for row in bad)),
            repaired_long_runs=bool(len(fixed) == 4 and all(row["bounded"] for row in fixed)),
            repaired_reflection=bool(
                len(report["pulse"]) == 2
                and all(row["peak_relative_difference"] < 1e-5 for row in report["pulse"])
            ),
        )
    report["passed"] = all(report["checks"].values())
    summary.write_text(json.dumps(report, indent=2) + "\n")
    plot_results(args.output, report)
    print(json.dumps(report["checks"], indent=2))
    if not report["passed"]:
        raise SystemExit("A profile-investigation check failed; see summary.json")


def plot_results(output, report):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    frequency = np.geomspace(1e9, 200e9, 1000)
    q = 2j * np.pi * frequency * epsilon_0
    sigma = report["analytic"]["sigma_s_per_m"]
    for shift, label in ((0.0, "Duplicated unshifted"), (1.1, "Shifted second factor")):
        stretch = (1 + sigma / q) * (1 + sigma / (shift * sigma + q))
        axes[0].semilogx(frequency / 1e9, stretch.real, label=label)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set(
        yscale="symlog",
        xlabel="Frequency (GHz)",
        ylabel="Real part of total stretch",
        title="Local profile, sigma = 2 S/m",
    )
    axes[0].legend(fontsize=8)
    bad = [
        row
        for row in report["analytic"]["fourier_cases"]
        if row["profile"] == "duplicated_unshifted"
    ]
    axes[1].plot(
        [row["dt_ratio"] for row in bad],
        [row["amplitude_growth_per_s"] / 1e9 for row in bad],
        "o-",
        label="Native discrete symbol",
    )
    axes[1].axhline(
        report["analytic"]["continuum_dominant_root_per_s"][0] / 1e9,
        color="black",
        linestyle="--",
        label="Continuous-time limit",
    )
    axes[1].set(
        xlabel="Time step / original time step",
        ylabel="Amplitude growth rate (1/ns)",
        title="Growth persists as dt tends to zero",
    )
    axes[1].legend(fontsize=8)
    for row in report["runtime"]:
        if row.get("build_rejected"):
            continue
        trace = np.loadtxt(output / (row["key"] + ".csv"), delimiter=",", skiprows=1)
        label = (
            f"Bad PEC, dt factor {row['timestep_factor']:g}"
            if row["shift_scale"] == 0
            else f"Fixed {row['kind']}, {row['precision']}"
        )
        axes[2].semilogy(trace[:, 0] * 1e9, trace[:, 1], label=label)
    axes[2].set(
        xlabel="Time (ns)",
        ylabel="Squared field norm / initial",
        title="Full graded guide: failure and repair",
    )
    if report["runtime"]:
        axes[2].legend(fontsize=7)
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.savefig(output / "diagnosis.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
