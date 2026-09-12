"""A driven 2D TE/TM guide with exact PMC or dispersive SIBC walls.

Run from the repository root:
    python examples/features/impedance_surface/virtual_waveguide_2d.py --mode TM --wall pmc
    python examples/features/impedance_surface/virtual_waveguide_2d.py --mode TE --wall foster
"""

import argparse
from pathlib import Path

import gprMax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("TE", "TM"), default="TM")
    parser.add_argument("--wall", choices=("pmc", "resistive", "foster"), default="pmc")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--output-dir", type=Path, default=Path("results/virtual_waveguide_2d"))
    args = parser.parse_args()
    inf = float("inf")
    scene = gprMax.Scene()
    for obj in (
        gprMax.DomainMode(mode=args.mode),
        gprMax.Domain(p1=(0.072, 0.026, inf)),
        gprMax.Discretisation(p1=(0.001,) * 3),
        gprMax.TimeWindow(iterations=args.steps),
        gprMax.PMLThickness(thickness=(8, 0, 0, 8, 0, 0)),
        gprMax.OMPThreads(1),
    ):
        scene.add(obj)
    if args.wall == "foster":
        scene.add(
            gprMax.SurfaceImpedance(
                id="wall", conductivity=1e4, fit_frequency_range=(1e9, 200e9), fit_order=4
            )
        )
    else:
        scene.add(gprMax.SurfaceImpedance(id="wall", resistance=inf if args.wall == "pmc" else 5.0))
    for lo, hi in ((0.002, 0.005), (0.021, 0.024)):
        scene.add(gprMax.Box(p1=(0, lo, 0), p2=(0.072, hi, inf), material_id="wall", averaging="n"))
    scene.add(gprMax.Waveform(wave_type="contsine", amp=1.0, freq=22e9, id="drive"))
    scene.add(gprMax.EigenmodeBand(id="band", fmin=22e9, fmax=22e9, points=1))
    scene.add(
        gprMax.EigenmodePort(
            port=1,
            p1=(0.024, 0.003, 0),
            p2=(0.024, 0.023, inf),
            direction="+",
            modes=(1,),
            anchors=(22e9,),
            plot_fields=False,
        )
    )
    scene.add(
        gprMax.VirtualWaveguide(port=1, length_cells=24, pml_cells=8, source_clearance_cells=4)
    )
    scene.add(gprMax.EigenmodeExcitation(port=1, mode=1, waveform="drive", plot_waveform=False))
    scene.add(gprMax.Rx(p1=(0.036, 0.010, inf), id="probe"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gprMax.run(
        scenes=[scene],
        outputfile=args.output_dir / f"{args.mode}_{args.wall}",
        cpu_precision="double",
        hide_progress_bars=True,
    )


if __name__ == "__main__":
    main()
