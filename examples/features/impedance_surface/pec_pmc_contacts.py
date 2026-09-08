"""A copper SIBC block touching PEC and PMC volumes.

Run from the repository root::

    python examples/features/impedance_surface/pec_pmc_contacts.py --output-dir results/contacts

Add --symmetry pec or --symmetry pmc to cut the volumes at an x0 symmetry plane.

The lower y face touches PEC and its tangential E is zero. The upper y face
touches a PMC volume and uses its existing constrained-H discretisation.
This demonstrates mixed-material compatibility, not PMC boundary accuracy.
"""

import argparse
from pathlib import Path

import gprMax


def build_scene(symmetry=None):
    scene = gprMax.Scene()
    xlo, xhi = (0, 0.006) if symmetry else (0.012, 0.018)
    xmid = (xlo + xhi) / 2
    front_x = xhi if symmetry else xlo
    for obj in (
        gprMax.Domain(p1=(0.03, 0.03, 0.03)),
        gprMax.Discretisation(p1=(0.001, 0.001, 0.001)),
        gprMax.TimeWindow(time=1e-9),
        gprMax.PMLThickness(thickness=3),
        gprMax.OMPThreads(1),
        gprMax.SurfaceImpedance(
            id="copper", preset="copper", fit_frequency_range=(1e9, 15e9), fit_order=8,
        ),
        gprMax.Box(p1=(xlo, 0.012, 0.012), p2=(xhi, 0.018, 0.018), material_id="copper"),
        gprMax.Box(p1=(xlo, 0.009, 0.012), p2=(xhi, 0.012, 0.018), material_id="pec"),
        gprMax.Box(p1=(xlo, 0.018, 0.012), p2=(xhi, 0.021, 0.018), material_id="pmc"),
        gprMax.Waveform(wave_type="ricker", amp=1, freq=5e9, id="pulse"),
        gprMax.HertzianDipole((0.008, 0.015, 0.015), "z", "pulse"),
        gprMax.Rx(p1=(xmid, 0.012, 0.015), id="pec_contact", outputs=["Ez"]),
        gprMax.Rx(p1=(xmid, 0.018, 0.015), id="pmc_contact", outputs=["Ez"]),
        gprMax.Rx(p1=(front_x, 0.015, 0.015), id="air_contact", outputs=["Ez"]),
        gprMax.GeometryView(
            p1=(0, 0, 0), p2=(0.03, 0.03, 0.03), dl=(0.001, 0.001, 0.001),
            filename="pec_pmc_contacts", output_type="n",
        ),
    ):
        scene.add(obj)
    if symmetry:
        scene.add(gprMax.SymmetryBoundary(face="x0", type=symmetry))
        scene.add(gprMax.Rx(p1=(0, 0.015, 0.012), id="symmetry_contact", outputs=["Ey"]))
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/pec_pmc_contacts"))
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--symmetry", choices=("pec", "pmc"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gprMax.run(
        scenes=[build_scene(args.symmetry)], outputfile=args.output_dir / "pec_pmc_contacts",
        geometry_only=args.geometry_only, cpu_precision="double",
    )


if __name__ == "__main__":
    main()
