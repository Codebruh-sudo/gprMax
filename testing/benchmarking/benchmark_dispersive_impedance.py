"""Reproduce dispersive-contact correctness checks and CPU timing results.

The driven 3-D checks compare Cython with the Python boundary update, measure
post-pulse decay, and check convergence toward an ordinary PEC box as surface
resistance tends to zero. These are discrete solver checks, not a continuum
accuracy or unconditional-stability certificate. The timing sweep uses the
same exterior material in each ordinary-grid baseline and impedance case.

Example::

    python -m testing.benchmarking.benchmark_dispersive_impedance \
        --cells 48 --threads 4 --explicit-orders 4 8 \
        --output testing/benchmarking/results/dispersive_impedance.json
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

import gprMax
import gprMax.impedance_surfaces as implementation
from gprMax.grid.fdtd_grid import FDTDGrid
from testing.benchmarking.benchmark_impedance_box import _parser, add_exterior, run_benchmark


def validate_contact(kind, directory: Path, *, iterations=2400):
    """Run a pulse-driven box with dynamic surface and two PEC-limit cases."""
    dl = 0.001
    captured = {}
    original_build = FDTDGrid.build

    def capture(grid):
        original_build(grid)
        captured["grid"] = grid

    def run(label, *, python=False, resistance=None, pec=False):
        from gprMax.impedance_surfaces import MAX_SIBC_TIMESTEP_FACTOR

        scene = gprMax.Scene()
        for obj in (
            gprMax.Domain(p1=(8 * dl,) * 3),
            gprMax.Discretisation(p1=(dl,) * 3),
            gprMax.TimeWindow(iterations=iterations),
            gprMax.OMPThreads(1),
            gprMax.PMLThickness(thickness=0),
            # Compare the PEC limit on the same time samples as SIBC.
            gprMax.TimeStepStabilityFactor(f=MAX_SIBC_TIMESTEP_FACTOR),
        ):
            scene.add(obj)
        add_exterior(scene, kind, 8 * dl)
        if not pec:
            surface = (
                gprMax.SurfaceImpedance(id="wall", resistance=resistance)
                if resistance is not None
                else gprMax.SurfaceImpedance(
                    id="wall",
                    preset="copper",
                    fit_frequency_range=(8e9, 12e9),
                    fit_order=4,
                )
            )
            scene.add(surface)
        scene.add(
            gprMax.Box(p1=(3 * dl,) * 3, p2=(5 * dl,) * 3, material_id="pec" if pec else "wall")
        )
        scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="pulse"))
        scene.add(gprMax.HertzianDipole((2 * dl, 4 * dl, 4 * dl), "z", "pulse"))
        scene.add(gprMax.Rx(p1=(2 * dl, 4 * dl, 4 * dl)))
        kernel = None if python else implementation._cython_update
        with patch.object(FDTDGrid, "build", capture), patch.object(
            implementation, "_cython_update", kernel
        ):
            gprMax.run(
                scenes=[scene],
                outputfile=directory / label,
                cpu_precision="double",
                hide_progress_bars=True,
                log_level=40,
            )
        grid = captured["grid"]
        trace = np.asarray(grid.rxs[0].outputs["Ez"]).copy()
        fields = np.concatenate(
            [getattr(grid, name).ravel() for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")]
        )
        assert np.isfinite(fields).all() and np.isfinite(trace).all()
        return grid, trace, fields

    grid, trace, fields = run("cython")
    _, reference, reference_fields = run("python", python=True)
    peak = float(np.max(np.abs(trace)))
    if peak <= 1e-6:
        raise AssertionError("contact validation source did not excite the fields")
    relative_trace_error = float(np.linalg.norm(trace - reference) / np.linalg.norm(reference))
    relative_field_error = float(
        np.linalg.norm(fields - reference_fields) / max(np.linalg.norm(reference_fields), 1e-30)
    )
    tail_ratio = float(np.max(np.abs(trace[-200:])) / peak)
    _, pec_trace, _ = run("pec", pec=True)
    pec_errors = []
    for resistance in (1e-2, 1e-3):
        _, limit_trace, _ = run(f"resistance_{resistance:g}", resistance=resistance)
        pec_errors.append(
            float(np.linalg.norm(limit_trace - pec_trace) / np.linalg.norm(pec_trace))
        )
    passed = (
        relative_trace_error < 1e-11
        and relative_field_error < 1e-9
        and tail_ratio < 0.02
        and pec_errors[1] < 0.2 * pec_errors[0]
        and pec_errors[1] < 1e-3
    )
    result = {
        "exterior": kind,
        "passed": bool(passed),
        "domain_cells": [8, 8, 8],
        "iterations": iterations,
        "dt_seconds": grid.dt,
        "source_frequency_hz": 10e9,
        "surface": "copper, four Foster poles, 8-12 GHz fit",
        "python_cython_trace_relative_l2": relative_trace_error,
        "python_cython_final_field_relative_l2": relative_field_error,
        "last_200_samples_peak_over_run_peak": tail_ratio,
        "pec_limit_resistances_ohm": [1e-2, 1e-3],
        "pec_limit_trace_relative_l2": pec_errors,
        "boundary_polarization_poles": len(grid.impedance_surfaces.pole_coeffs),
    }
    return result


def main():
    parser = _parser()
    parser.description = __doc__
    parser.set_defaults(cells=48, explicit_orders=(4, 8))
    parser.add_argument(
        "--exteriors",
        nargs="+",
        choices=("none", "debye", "lorentz", "drude", "mixed"),
        default=("none", "debye", "lorentz", "drude", "mixed"),
    )
    parser.add_argument("--skip-timing", action="store_true")
    args = parser.parse_args()
    if (
        min(
            args.cells,
            args.iterations,
            args.threads,
            args.repeats,
            args.kernel_iterations,
            args.kernel_repeats,
            args.hot_iterations,
            args.hot_repeats,
        )
        <= 0
        or args.cells < 24
    ):
        parser.error("cells must be >=24 and iteration/thread/repeat counts must be positive")
    result = {
        "timings": [],
        "validation": [],
        "validation_scope": (
            "3-D pulse solves: Python/Cython agreement, late-time decay, and convergence "
            "to the ordinary PEC boundary. Timing scenes have no sources."
        ),
    }
    for kind in dict.fromkeys(args.exteriors):
        args.exterior = kind
        if not args.skip_timing:
            result["timings"].append(run_benchmark(args))
    with TemporaryDirectory(prefix="dispersive_contact_", dir=Path.cwd()) as tmp:
        for kind in dict.fromkeys(args.exteriors):
            if kind != "none":
                directory = Path(tmp) / kind
                directory.mkdir()
                result["validation"].append(validate_contact(kind, directory))
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)
    if not all(row["passed"] for row in result["validation"]):
        raise SystemExit("Dispersive-contact validation failed; see the saved numerical results.")


if __name__ == "__main__":
    main()
