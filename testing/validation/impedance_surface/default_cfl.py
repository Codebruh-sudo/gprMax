"""Audit the gprMax timestep policy and rounded cavity coefficients.

Unlike the initial basis-probe audit, this builds the linear map directly
from stored coefficients and also calls gprMax's complete production Solver.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from decimal import Decimal, getcontext, localcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.linalg import eigvals

import gprMax
from gprMax import config
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.solvers import Solver
from testing.validation.impedance_surface.stability import (
    Stepper,
    audit_timestep_policy,
    build_case,
    make_scene,
)


def timestep_measurement(grid):
    """Compare the actual binary timestep with 80-digit CFL references."""
    constants = config.sim_config.em_consts
    spacing = (grid.dx, grid.dy, grid.dz)
    raw = float(1 / (constants["c"] * np.sqrt(sum(1 / d**2 for d in spacing))))
    precision = getcontext().prec
    with localcontext() as ctx:
        ctx.prec = 80
        exact = Decimal.from_float
        inverse_squares = sum(1 / exact(d) ** 2 for d in spacing)
        cfl_c = 1 / (exact(constants["c"]) * inverse_squares.sqrt())
        cfl_mass = (exact(constants["e0"]) * exact(constants["m0"]) / inverse_squares).sqrt()
        c_ratio = exact(grid.dt) / cfl_c
        mass_ratio = exact(grid.dt) / cfl_mass
        identity = exact(constants["c"]) ** 2 * exact(constants["e0"]) * exact(constants["m0"])
        return {
            "spacing_m": spacing,
            "dt_s": grid.dt,
            "dt_hex": grid.dt.hex(),
            "field_dtype": str(grid.Ex.dtype),
            "coefficient_dtype": str(grid.updatecoeffsH.dtype),
            "raw_binary64_cfl_s": raw,
            "raw_binary64_cfl_hex": raw.hex(),
            "ulps_below_raw_binary64": (raw - grid.dt) / np.spacing(raw),
            "decimal_context_precision": precision,
            "rounding_decimal_places": precision - 1,
            "cfl_from_c_s_80digits": str(cfl_c),
            "dt_over_cfl_from_c": str(c_ratio),
            "relative_margin_below_cfl_from_c": str(1 - c_ratio),
            "dt_over_cfl_from_epsilon_mu": str(mass_ratio),
            "c_squared_epsilon_mu": str(identity),
            "constants": {key: float(constants[key]) for key in ("c", "e0", "m0")},
        }


def coefficient_cavity_spectrum(stepper):
    """Use stored coefficients, promoted before algebra, without basis probes.

    The isolated retained voxel has 12 E edges and 6 H faces. Its real-
    arithmetic recurrence with the rounded production coefficients is exact
    here; per-operation roundoff during the actual solve is still separate.
    """
    g, s = stepper.grid, stepper.system
    if stepper.ny or g.maxpoles:
        raise ValueError(
            "The coefficient audit requires a resistive surface and nondispersive exterior"
        )
    ekeys, hkeys = [], []
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        for sb, sc in np.ndindex(2, 2):
            coord = [2, 2, 2]
            coord[b] += sb
            coord[c] += sc
            ekeys.append((a, *coord))
        for side in range(2):
            coord = [2, 2, 2]
            coord[a] += side
            hkeys.append((a, *coord))
    ei, hi = {key: i for i, key in enumerate(ekeys)}, {key: i for i, key in enumerate(hkeys)}
    edge_lookup = {tuple(edge[:4]): i for i, edge in enumerate(s.edge_info)}
    q, r = np.zeros((6, 12)), np.zeros((12, 6))
    a_e = np.empty(12)
    for row, key in enumerate(hkeys):
        axis, *coord = key
        b, c = (axis + 1) % 3, (axis + 2) % 3
        coeff = g.updatecoeffsH[int(g.ID[(axis + 3, *coord)])].astype(float)
        assert coeff[0] == 1
        for derivative, component, sign in ((b, c, -1), (c, b, 1)):
            high = coord.copy()
            high[derivative] += 1
            q[row, ei[(component, *high)]] += sign * coeff[derivative + 1] * stepper.eta
            q[row, ei[(component, *coord)]] -= sign * coeff[derivative + 1] * stepper.eta
    for row, key in enumerate(ekeys):
        index = edge_lookup[key]
        edge = s.edge_info[index]
        old, inverse = map(float, s.edge_runtime[index])
        a_e[row] = old * inverse
        for h in range(edge[4], edge[4] + edge[5]):
            hkey = tuple(s.h_info[h])
            # Every retained H sample must belong to the isolated cavity.
            r[row, hi[hkey]] += float(s.h_weight[h]) * inverse / stepper.eta
    np.testing.assert_allclose(a_e, a_e[0], rtol=0, atol=0)
    matrix = np.block([[np.diag(a_e) + r @ q, r], [q, np.eye(6)]])
    stiffness = float(np.max(eigvals(-r @ q).real))
    denominator = 2 * (1 + a_e[0])
    # For z^2 - (1+a-kappa)z+a=0, the high-frequency root crosses -1
    # when kappa > 2(1+a). This includes the midpoint resistive damping.
    return {
        "electric_retention_a": float(a_e[0]),
        "max_stiffness_kappa": stiffness,
        "stability_ceiling_2_times_1_plus_a": float(denominator),
        "relative_stiffness_excess": float(stiffness / denominator - 1),
        "effective_courant_from_coefficients": float(np.sqrt(stiffness / denominator)),
        "coefficient_matrix_spectral_radius": float(np.max(np.abs(eigvals(matrix)))),
    }


def full_solver_run(
    path,
    *,
    geometry="cavity",
    resistance=1e6,
    copper=False,
    courant=None,
    precision=None,
    steps=200000,
    seed=731,
    automatic_timestep=True,
):
    """Call gprMax.run with its normal Solver, seeding only the initial state.

    The default run uses the automatic SIBC cap; automatic_timestep=False
    bypasses that policy only for historical diagnostics. The solve wrapper seeds
    fields after normal solver construction; the iterator records norms.
    Omit cpu_precision as well when testing complete CPU defaults.
    """
    scene, occupied = make_scene(
        geometry=geometry, resistance=resistance, copper=copper, courant=courant, iterations=steps
    )
    result = {}
    original_solve = Solver.solve

    def observe_solve(solver, iterator):
        stepper = Stepper(solver.updates.grid, occupied)
        rng = np.random.default_rng(seed)
        initial = rng.standard_normal(stepper.size)
        initial[stepper.ne + stepper.nh :] = 0
        stepper.set(initial)
        initial_norm = np.linalg.norm(stepper.get())
        result["timestep"] = timestep_measurement(stepper.grid)
        if geometry == "cavity" and not copper:
            result["coefficients"] = coefficient_cavity_spectrum(stepper)
        samples = [[0, 1.0]]
        peak = 1.0

        def observe(iterations):
            nonlocal peak
            for iteration in iterations:
                yield iteration
                if (iteration + 1) % 20 == 0 or iteration + 1 == steps:
                    value = float(np.linalg.norm(stepper.get()) / initial_norm)
                    peak = max(peak, value)
                    if (iteration + 1) % max(20, steps // 200) == 0 or iteration + 1 == steps:
                        samples.append([iteration + 1, value])

        original_solve(solver, observe(iterator))
        tail = np.asarray(samples)[len(samples) // 2 :]
        fitted_growth = float(np.exp(np.polyfit(tail[:, 0], np.log(tail[:, 1]), 1)[0]))
        result["source_free"] = {
            "steps": steps,
            "seed": seed,
            "peak_state_norm_over_initial": peak,
            "final_state_norm_over_initial": samples[-1][1],
            "norm_samples": samples,
            "finite": bool(np.all(np.isfinite(stepper.get()))),
            "fitted_tail_growth_per_step": fitted_growth,
        }

    precision_arg = {} if precision is None else {"cpu_precision": precision}
    with patch.object(Solver, "solve", observe_solve), audit_timestep_policy(automatic_timestep):
        gprMax.run(
            scenes=[scene], outputfile=path, hide_progress_bars=True, log_level=40, **precision_arg
        )
    result["parameters"] = dict(
        geometry=geometry,
        resistance=resistance,
        copper=copper,
        timestep_factor=courant,
        cpu_precision_argument=precision,
        automatic_sibc_timestep=automatic_timestep,
    )
    return result


def mesh_rounding_sweep():
    """Exercise the actual 3D timestep function at several physical scales."""
    records = []
    for h in (1e-6, 1e-5, 1e-4, 0.001, 0.002, 0.005, 0.01, 0.1, 1.0):
        grid = SimpleNamespace(
            dx=h,
            dy=h,
            dz=h,
            Ex=np.empty(0, dtype=np.float32),
            updatecoeffsH=np.empty(0, dtype=np.float32),
        )
        FDTDGrid.calculate_dt(grid)
        records.append(timestep_measurement(grid))
    return records


def plot_results(results, path):
    import matplotlib.pyplot as plt

    keys = {
        "default_cavity_high_R": ("Default dt, default single", "#ba2434"),
        "default_dt_double_cavity_high_R": ("Default dt, double", "#b86d00"),
        "cavity_high_R_factor_0.9999999": ("Factor 0.9999999, single", "#6f4e9c"),
        "cavity_high_R_factor_0.999999": ("Factor 0.999999, single", "#185a9d"),
        "cavity_high_R_factor_0.99": ("Factor 0.99, single", "#008b81"),
    }
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    for case in results:
        if case["name"] not in keys:
            continue
        label, color = keys[case["name"]]
        if case["parameters"].get("automatic_sibc_timestep", False):
            label = label.replace("Factor", "Requested factor") + " (cap 0.99)"
        samples = np.asarray(case["source_free"]["norm_samples"])
        ax.semilogy(samples[:, 0], samples[:, 1], color=color, label=label)
    ax.set(
        xlabel="Timesteps in the full gprMax solver",
        ylabel="Field norm / initial field norm",
        title="One-voxel cavity, 1 MΩ surface\nFull CPU solver; timestep policy recorded in legend",
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument(
        "--historical",
        action="store_true",
        help="Disable the automatic SIBC cap within this diagnostic to reproduce the old endpoint",
    )
    args = parser.parse_args(argv)
    if args.output is None:
        name = "default_cfl.json" if args.historical else "default_cfl_protected.json"
        args.output = Path("testing/validation/impedance_surface/results") / name
    if args.steps < 20:
        parser.error("--steps must be at least 20")
    cases = [
        dict(name="default_cavity_high_R"),
        dict(name="default_dt_double_cavity_high_R", precision="double"),
        dict(name="default_cavity_R50", resistance=50.0),
        dict(name="default_cavity_copper", copper=True),
        dict(name="default_stair_high_R", geometry="stair"),
        dict(name="default_stair_copper", geometry="stair", copper=True),
    ]
    results, sweep = [], []
    with tempfile.TemporaryDirectory(prefix="gprmax-default-cfl-") as tmp:
        for factor in (
            None,
            np.nextafter(1.0, 0.0),
            1 - 1e-8,
            1 - 1e-7,
            1 - 1e-6,
            0.99999,
            0.9999,
            0.999,
            0.99,
        ):
            g, occupied = build_case(
                Path(tmp) / "coefficients",
                geometry="cavity",
                resistance=1e6,
                courant=factor,
                precision=None,
                automatic_timestep=not args.historical,
            )
            stepper = Stepper(g, occupied)
            sweep.append(
                dict(
                    timestep_factor=factor,
                    timestep=timestep_measurement(g),
                    coefficients=coefficient_cavity_spectrum(stepper),
                )
            )
        rounding = mesh_rounding_sweep()
        for factor in (1 - 1e-7, 1 - 1e-6, 0.99):
            cases.append(dict(name=f"cavity_high_R_factor_{factor}", courant=factor))
        for case in cases:
            case = case.copy()
            name = case.pop("name")
            result = dict(
                name=name,
                **full_solver_run(
                    Path(tmp) / name,
                    steps=args.steps,
                    automatic_timestep=not args.historical,
                    **case,
                ),
            )
            results.append(result)
            print(
                json.dumps(
                    {
                        **result,
                        "source_free": {
                            k: v for k, v in result["source_free"].items() if k != "norm_samples"
                        },
                    }
                ),
                flush=True,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    {"mesh_rounding_sweep": rounding, "coefficient_sweep": sweep, "cases": results},
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    plot_results(results, args.output.with_suffix(".png"))
    return results


if __name__ == "__main__":
    main()
