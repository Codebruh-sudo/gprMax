"""Source-free stability audit of the production clipped-circulation update.

Run ``python -m testing.validation.impedance_surface.stability --help``.
Dense amplification matrices are intentionally restricted to small PEC boxes.
There is no PML, excitation, or dispersive bulk material to obscure the
stability of the geometric coupling and the surface ADE itself.
"""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy
from scipy.linalg import eigvals, svdvals

import gprMax
from gprMax import config
from gprMax.cython.fields_updates_normal import update_electric, update_magnetic
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.impedance_surfaces import _component_valid_view


def make_scene(
    *,
    geometry="stair",
    resistance=50.0,
    copper=False,
    spacing=(0.001, 0.001, 0.001),
    courant=None,
    iterations=1,
):
    """Create the audit geometry; None leaves the timestep at its default."""
    dl = np.asarray(spacing)
    scene = gprMax.Scene()
    for obj in (
        gprMax.Domain(p1=tuple(5 * dl)),
        gprMax.Discretisation(p1=tuple(dl)),
        gprMax.TimeWindow(iterations=iterations),
        gprMax.PMLThickness(thickness=0),
        gprMax.OMPThreads(1),
    ):
        scene.add(obj)
    if courant is not None and courant <= 1:
        scene.add(gprMax.TimeStepStabilityFactor(f=courant))
    if copper:
        scene.add(
            gprMax.SurfaceImpedance(
                id="wall",
                preset="copper",
                fit_frequency_range=(1e9, 20e9),
                fit_order=4,
            )
        )
    else:
        scene.add(gprMax.SurfaceImpedance(id="wall", resistance=resistance))
    cells = {
        "box": [(2, 2, 2)],
        "stair": [(2, 2, 2), (3, 2, 2), (3, 3, 2)],
        "cavity": [tuple(np.asarray(p) + 1) for p in np.ndindex(3, 3, 3) if tuple(p) != (1, 1, 1)],
        "plate": [(2, j, k) for j in (1, 2, 3) for k in (1, 2, 3)],
    }[geometry]
    occupied = np.zeros((5, 5, 5), dtype=bool)
    for cell in cells:
        cell = np.asarray(cell)
        occupied[tuple(cell)] = True
        scene.add(
            gprMax.Box(
                p1=tuple(cell * dl), p2=tuple((cell + 1) * dl), material_id="wall", averaging="n"
            )
        )
    return scene, occupied


def audit_timestep_policy(automatic=True):
    """Validation-only bypass for reproducing the historical CFL failure."""
    if automatic:
        return nullcontext()
    return patch.object(
        gprMax.Scene, "_model_objects_with_sibc_timestep", lambda scene: scene.single_use_objects
    )


def build_case(
    path,
    *,
    geometry="stair",
    resistance=50.0,
    copper=False,
    spacing=(0.001, 0.001, 0.001),
    courant=0.99,
    precision="double",
    automatic_timestep=True,
):
    """Build a five-cell box, including a diagnostic above-CFL override.

    With courant=None no user timestep command is added; the production SIBC
    cap applies unless automatic_timestep=False is explicitly requested for
    a historical diagnostic. precision=None omits cpu_precision.
    """
    scene, occupied = make_scene(
        geometry=geometry, resistance=resistance, copper=copper, spacing=spacing, courant=courant
    )
    captured = []
    original_build, original_dt = FDTDGrid.build, FDTDGrid.calculate_dt

    def capture(grid):
        original_build(grid)
        captured.append(grid)

    def set_dt(grid):
        original_dt(grid)
        grid.dt *= courant  # Above 1 is a deliberate negative control only.

    override = (
        patch.object(FDTDGrid, "calculate_dt", set_dt)
        if courant is not None and courant > 1
        else nullcontext()
    )
    precision_arg = {} if precision is None else {"cpu_precision": precision}
    with patch.object(FDTDGrid, "build", capture), override, audit_timestep_policy(
        automatic_timestep
    ):
        gprMax.run(
            scenes=[scene],
            outputfile=path,
            geometry_only=True,
            **precision_arg,
            hide_progress_bars=True,
            log_level=40,
        )
    return captured[0], occupied


class Stepper:
    """Pack only unconstrained DOFs and call the actual CPU field kernels."""

    def __init__(self, grid, occupied):
        self.grid = grid
        self.system = grid.impedance_surfaces
        self.fields = [getattr(grid, name) for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")]
        self.eta = np.sqrt(config.sim_config.em_consts["m0"] / config.sim_config.em_consts["e0"])
        self.indices, self.weights = [], []
        # Accumulate geometric energy volumes independently, one retained
        # voxel at a time: 1/4 on its 12 E edges, 1/2 on its six H faces.
        volumes = [np.zeros_like(grid.Ex, dtype=float) for _ in range(6)]
        for cell in np.argwhere(~occupied):
            for axis in range(3):
                b, c = (axis + 1) % 3, (axis + 2) % 3
                for sb, sc in np.ndindex(2, 2):
                    edge = cell.copy()
                    edge[b] += sb
                    edge[c] += sc
                    volumes[axis][tuple(edge)] += 0.25
                for side in range(2):
                    face = cell.copy()
                    face[axis] += side
                    volumes[axis + 3][tuple(face)] += 0.5
        for component in range(6):
            ids = _component_valid_view(grid.ID[component], component, grid)
            coeffs = grid.updatecoeffsE if component < 3 else grid.updatecoeffsH
            active = coeffs[ids, 0] != 0
            # The ordinary PEC box leaves tangential electric endpoints fixed.
            if component < 3:
                for axis in range(3):
                    if axis != component:
                        for side in (0, active.shape[axis] - 1):
                            section = [slice(None)] * 3
                            section[axis] = side
                            active[tuple(section)] = False
            idx = tuple(np.argwhere(active).T)
            self.indices.append(idx)
            self.weights.append(volumes[component][idx])
        self.sizes = [len(i[0]) for i in self.indices]
        self.ne = sum(self.sizes[:3])
        self.nh = sum(self.sizes[3:])
        self.ny = sum(int(self.system.model_info[m, 0]) for m, _ in self.system.port_info)
        self.size = self.ne + self.nh + self.ny
        self.we = np.concatenate(self.weights[:3])
        self.wh = np.concatenate(self.weights[3:])
        if np.any(self.we <= 0) or np.any(self.wh <= 0):
            raise AssertionError("An active field has zero retained energy volume")

    def clear(self):
        for field in self.fields:
            field.fill(0)
        self.system.state_y.fill(0)

    def set(self, vector):
        self.clear()
        offset = 0
        for component, (field, idx, count) in enumerate(zip(self.fields, self.indices, self.sizes)):
            field[idx] = vector[offset : offset + count] / (self.eta if component >= 3 else 1)
            offset += count
        self.system.state_y[: self.ny] = vector[offset:]

    def get(self):
        return np.concatenate(
            [
                *(
                    field[idx].astype(float) * (self.eta if c >= 3 else 1)
                    for c, (field, idx) in enumerate(zip(self.fields, self.indices))
                ),
                self.system.state_y[: self.ny].astype(float),
            ]
        )

    def magnetic(self):
        g = self.grid
        update_magnetic(g.nx, g.ny, g.nz, -1, 1, g.updatecoeffsH, g.ID, *self.fields)

    def electric(self):
        g = self.grid
        update_electric(g.nx, g.ny, g.nz, -1, 1, g.updatecoeffsE, g.ID, *self.fields)
        self.system.update(g)

    def step(self):
        self.magnetic()
        self.electric()

    def amplification(self):
        """Probe the complete E/H/ADE transition with basis vectors.

        Floating-point arithmetic is not linear. Especially at a multiple
        unit-circle eigenvalue, this diagnostic cannot replace long runs.
        """
        matrix = np.empty((self.size, self.size))
        basis = np.zeros(self.size)
        for column in range(self.size):
            basis[column] = 1
            self.set(basis)
            self.step()
            matrix[:, column] = self.get()
            basis[column] = 0
        self.clear()
        return matrix

    def curls(self):
        """Measure H<-E and E<-H with the surface load temporarily opened.

        Opening the port isolates the geometric coupling; all compiled
        half-line weights and retained masses remain unchanged.
        """
        q = np.empty((self.nh, self.ne))
        r = np.empty((self.ne, self.nh))
        basis = np.zeros(self.size)
        for column in range(self.ne):
            basis[column] = 1
            self.set(basis)
            self.magnetic()
            q[:, column] = self.get()[self.ne : self.ne + self.nh]
            basis[column] = 0
        system = self.system
        saved = system.edge_runtime.copy(), system.port_g_over_Z0.copy()
        try:
            system.edge_runtime[:, 0] = system.edge_params[:, 1]
            system.edge_runtime[:, 1] = 1 / system.edge_params[:, 0]
            system.port_g_over_Z0.fill(0)
            for column in range(self.nh):
                basis[self.ne + column] = 1
                self.set(basis)
                self.electric()
                r[:, column] = self.get()[: self.ne]
                basis[self.ne + column] = 0
        finally:
            system.edge_runtime[:], system.port_g_over_Z0[:] = saved
            self.clear()
        return q, r


def audit_geometry(stepper):
    q, r = stepper.curls()
    lhs, rhs = stepper.we[:, None] * r, -q.T * stepper.wh[None, :]
    defect = np.max(np.abs(lhs - rhs)) / max(np.max(np.abs(lhs)), 1e-300)
    coupling = np.sqrt(stepper.wh[:, None]) * q / np.sqrt(stepper.we[None, :])
    sigma = float(svdvals(coupling)[0])
    return {
        "weighted_adjoint_relative_defect": float(defect),
        "dt_times_max_frequency": sigma,
        "positive_energy_margin": 1 - sigma / 2,
        "dt_limit_over_current_dt": 2 / sigma,
    }, q


def source_free_run(stepper, q, *, steps=20000, seed=731):
    """Check all fields, not a receiver trace, from broadband random data.

    The modified leapfrog energy is valid for the resistive cases. Copper
    also stores energy in ADE states, so its field energy is descriptive.
    """
    rng = np.random.default_rng(seed)
    initial = rng.standard_normal(stepper.size)
    initial[stepper.ne + stepper.nh :] = 0
    stepper.set(initial)

    def energy(vector):
        e, h = vector[: stepper.ne], vector[stepper.ne : stepper.ne + stepper.nh]
        return float(0.5 * (np.dot(stepper.we * e, e) + np.dot(stepper.wh * h, h + q @ e)))

    initial_energy = energy(stepper.get())
    previous, peak, increase = initial_energy, initial_energy, 0.0
    field_peak, final = np.linalg.norm(initial), initial
    samples = [[0, 1.0]]
    # Sample every 20 steps; no assertion of stepwise monotonicity is made.
    completed = 0
    for iteration in range(steps):
        stepper.step()
        completed = iteration + 1
        if completed % 20 == 0 or completed == steps:
            final = stepper.get()
            if not np.all(np.isfinite(final)):
                return {"finite": False, "steps": completed}
            current = energy(final)
            peak = max(peak, current)
            increase = max(increase, current - previous)
            previous = current
            field_peak = max(field_peak, np.linalg.norm(final))
            if completed % max(20, 20 * (steps // 4000)) == 0 or completed == steps:
                samples.append([completed, float(np.linalg.norm(final) / np.linalg.norm(initial))])
    return {
        "finite": True,
        "steps": completed,
        "final_modified_field_energy_over_initial": previous / initial_energy,
        "peak_modified_field_energy_over_initial": peak / initial_energy,
        "largest_sampled_energy_increase_over_initial": increase / initial_energy,
        "peak_state_norm_over_initial": float(field_peak / np.linalg.norm(initial)),
        "final_state_norm_over_initial": float(np.linalg.norm(final) / np.linalg.norm(initial)),
        "state_norm_samples": samples,
    }


def plot_cavity_results(results, path):
    """Export the observed CFL-endpoint sensitivity as a scientific figure."""
    import matplotlib.pyplot as plt

    selected = {
        "cavity_high_R_at_CFL": ("CFL 1.00, double", "#b86d00", "-"),
        "cavity_high_R_single_at_CFL": ("CFL 1.00, single", "#ba2434", "-"),
        "cavity_high_R_below_CFL": ("CFL 0.99, double", "#185a9d", "-"),
        "cavity_high_R_single_below_CFL": ("CFL 0.99, single", "#008b81", "--"),
    }
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for result in results:
        if result["name"] not in selected:
            continue
        label, color, style = selected[result["name"]]
        if result["parameters"].get("automatic_timestep", False):
            label = label.replace("CFL", "Requested factor") + " (cap 0.99)"
        samples = np.asarray(result["source_free"]["state_norm_samples"])
        ax.semilogy(samples[:, 0], samples[:, 1], label=label, color=color, linestyle=style)
    ax.set(
        xlabel="Source-free timesteps",
        ylabel="Field norm / initial field norm",
        title="One-voxel cavity, passive 1 MΩ surface\nProduction CPU update; fixed random initial fields",
    )
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument(
        "--quick", action="store_true", help="Only the staircase and above-CFL control"
    )
    parser.add_argument(
        "--historical",
        action="store_true",
        help="Disable the automatic SIBC cap within this diagnostic to reproduce the old endpoint",
    )
    args = parser.parse_args(argv)
    if args.output is None:
        name = "stability.json" if args.historical else "stability_protected.json"
        args.output = Path("testing/validation/impedance_surface/results") / name
    if args.steps <= 0:
        parser.error("--steps must be positive")
    cases = [
        dict(name="stair_R50", geometry="stair", resistance=50.0),
        dict(name="above_CFL_control", geometry="stair", resistance=50.0, courant=1.2),
    ]
    if not args.quick:
        cases += [
            dict(name="box_R50", geometry="box", resistance=50.0),
            dict(name="plate_R50", geometry="plate", resistance=50.0),
            dict(name="cavity_R50", geometry="cavity", resistance=50.0),
            dict(name="stair_low_R", geometry="stair", resistance=0.001),
            dict(name="stair_high_R", geometry="stair", resistance=1e6),
            dict(name="stair_anisotropic", geometry="stair", spacing=(0.0005, 0.001, 0.002)),
            dict(name="stair_copper", geometry="stair", copper=True),
            dict(name="stair_at_CFL", geometry="stair", courant=1.0),
            dict(name="stair_copper_at_CFL", geometry="stair", copper=True, courant=1.0),
            dict(
                name="cavity_copper_single_at_CFL",
                geometry="cavity",
                copper=True,
                precision="single",
                courant=1.0,
            ),
            dict(name="cavity_at_CFL", geometry="cavity", courant=1.0),
            dict(name="cavity_high_R_at_CFL", geometry="cavity", resistance=1e6, courant=1.0),
            dict(name="cavity_high_R_below_CFL", geometry="cavity", resistance=1e6),
            dict(name="cavity_single_at_CFL", geometry="cavity", courant=1.0, precision="single"),
            dict(
                name="cavity_high_R_single_at_CFL",
                geometry="cavity",
                resistance=1e6,
                courant=1.0,
                precision="single",
            ),
            dict(
                name="cavity_high_R_single_below_CFL",
                geometry="cavity",
                resistance=1e6,
                precision="single",
            ),
            dict(name="stair_single", geometry="stair", precision="single"),
        ]
    results = []
    with tempfile.TemporaryDirectory(prefix="gprmax-stability-") as tmp:
        for case in cases:
            case = case.copy()
            name = case.pop("name")
            case["automatic_timestep"] = not args.historical and name != "above_CFL_control"
            grid, occupied = build_case(Path(tmp) / name, **case)
            stepper = Stepper(grid, occupied)
            geometry_result, q = audit_geometry(stepper)
            matrix = stepper.amplification()
            eigenvalues = eigvals(matrix)
            tolerance = 2e-6 if case.get("precision") == "single" else 2e-8
            result = dict(
                name=name,
                parameters=case,
                dt=grid.dt,
                dofs=stepper.size,
                surface_states=stepper.ny,
                retained_fractions=np.unique(stepper.system.edge_fraction).tolist(),
                **geometry_result,
                spectral_radius=float(np.max(np.abs(eigenvalues))),
                eigenvalue_tolerance=tolerance,
                eigenvalues_outside_tolerance=int(
                    np.count_nonzero(np.abs(eigenvalues) > 1 + tolerance)
                ),
            )
            steps = min(args.steps, 100) if name == "above_CFL_control" else args.steps
            if name.startswith("cavity_high_R_") or name == "cavity_copper_single_at_CFL":
                steps = max(steps, 200000)
            result["source_free"] = source_free_run(stepper, q, steps=steps)
            results.append(result)
            print(
                json.dumps(
                    {
                        **result,
                        "source_free": {
                            k: v
                            for k, v in result["source_free"].items()
                            if k != "state_norm_samples"
                        },
                    }
                ),
                flush=True,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    {
                        "python": platform.python_version(),
                        "numpy": np.__version__,
                        "scipy": scipy.__version__,
                        "platform": platform.platform(),
                        "backend": "production CPU Cython; one OpenMP thread",
                        "norm": "Euclidean norm of (E, eta0 H, surface y); y initially zero",
                        "spectral_caveat": "Basis probes approximate floating-point arithmetic; use long runs too.",
                        "cases": results,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    if not args.quick:
        plot_cavity_results(results, args.output.with_suffix(".png"))
    return results


if __name__ == "__main__":
    main()
