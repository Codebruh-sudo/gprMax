"""Run maintained analytical scenes with MPI or OpenCL backend selection.

Run from any working directory using a Python environment containing the
repository's dependencies and compiled extensions. For example, from the repo:

    mpiexec -n 2 python docs/reports/support/cross_backend_analytical.py \
        hertzian /tmp/hertzian-mpi --mpi 2 1 1 --precision double

The maintained drivers provide the scenes, reference formulations and acceptance
limits. This support script changes only backend selection and output locations.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import gprMax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("hertzian", "fresnel"))
    parser.add_argument("output", type=Path)
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--mpi", nargs=3, type=int, metavar=("NX", "NY", "NZ"))
    backend.add_argument("--opencl", type=int, metavar="DEVICE")
    parser.add_argument("--precision", choices=("single", "double"), default="double")
    args = parser.parse_args()
    if args.mpi and any(size < 1 for size in args.mpi):
        parser.error("--mpi dimensions must be positive")
    if args.opencl is not None and args.opencl < 0:
        parser.error("--opencl requires a non-negative device index")
    comm = None
    if args.mpi:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    root = comm is None or comm.rank == 0
    args.output.mkdir(parents=True, exist_ok=True)
    options = {"cpu_precision": args.precision}
    if args.mpi:
        options["mpi"] = tuple(args.mpi)
    if args.opencl is not None:
        options = {"opencl": [args.opencl], "gpu_precision": args.precision}

    def solve(scene, name):
        start = time.perf_counter()
        output = args.output / name
        gprMax.run(
            scenes=[scene],
            n=1,
            outputfile=output,
            hide_progress_bars=True,
            log_level=logging.WARNING,
            **options,
        )
        return output.with_suffix(".h5"), time.perf_counter() - start

    if args.case == "hertzian":
        from testing.validation import validate_hertzian_dipole as driver

        far_scene, requests = driver.build_far_field_scene(threads=1)
        _, far_seconds = solve(far_scene, "far")
        if root:
            far = driver.analyse_far_field(requests)
        if comm is not None:
            comm.Barrier()
        near_scene, receiver = driver.build_near_field_scene(threads=1)
        path, near_seconds = solve(near_scene, "near")
        if root:
            near = driver.analyse_near_field(path, receiver)
            driver._save_far_csv(args.output / "far.csv", far)
            driver._save_near_csv(args.output / "near.csv", near)
            driver._plot_far(args.output / "far.png", far)
            driver._plot_near(args.output / "near.png", near)
            summary = {
                "far_field": driver._compact_far_summary(far),
                "near_field": driver._compact_near_summary(near),
                "acceptance": driver._acceptance_summary(far, near),
                "seconds": {"far": far_seconds, "near": near_seconds},
            }
    else:
        import h5py
        from testing.validation import validate_plane_wave_dispersive_halfspace as driver

        results, totals = {}, {}
        for name in ["free_space", *driver.CASES]:
            path, seconds = solve(driver.build_scene(name, 1), name)
            if root:
                with h5py.File(path) as data:
                    trace, dt = data["rxs/rx1/Ez"][...], float(data.attrs["dt"])
                if name == "free_space":
                    incident = trace
                else:
                    totals[name] = trace
                    results[name] = driver.analyse_case(name, incident, trace, dt)
                    driver._save_csv(args.output / (name + ".csv"), results[name])
            if comm is not None:
                comm.Barrier()
        if root:
            driver._plot_comparison(results, args.output)
            summary = {
                "acceptance": driver._acceptance_summary(results),
                "results": {
                    k: {m: v for m, v in r.items() if isinstance(v, (float, int, str))} for k, r in results.items()
                },
            }
    if root:
        summary["backend_options"] = options
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
        accepted = summary["acceptance"]["passed"]
    else:
        accepted = None
    if comm is not None:
        accepted = comm.bcast(accepted, root=0)
    if not accepted:
        raise SystemExit("Analytical acceptance limits exceeded")


if __name__ == "__main__":
    main()
