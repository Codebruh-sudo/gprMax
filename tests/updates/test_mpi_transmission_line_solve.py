# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

"""Real MPI/serial parity for transmission-line magnetic sampling stages.

The overlapping magnetic dipole writes a *different Yee edge* from the
electric TL feed, but that H edge contributes to the feed's Ampere contour.
For transverse decompositions the lower-boundary case deliberately puts
this writer on the neighbouring rank, so a fresh H halo is essential.
"""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = Path(sys.executable).parent / "mpiexec"
MPIEXEC = str(LAUNCHER) if LAUNCHER.is_file() else shutil.which("mpiexec")
ITERATIONS = 100
CASES = [(polarisation, placement) for polarisation in "xyz" for placement in ("inside", "lower")]
pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        MPIEXEC is None or importlib.util.find_spec("mpi4py") is None,
        reason="requires mpi4py and an MPI launcher",
    ),
]


def _run(output, axis, precision, drive, distributed):
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        str(output),
        "--axis",
        str(axis),
        "--precision",
        precision,
        "--drive",
        drive,
    ]
    if distributed:
        command = [MPIEXEC, "-n", "2", *command, "--mpi"]
    environment = os.environ.copy()
    # Each child is a complete MPI application. Do not inherit pytest's
    # MPI finalisation override, or constrain the launcher's transport.
    environment.pop("MPI4PY_RC_FINALIZE", None)
    environment.pop("FI_PROVIDER", None)
    environment.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    environment.update(
        PYTHONPATH=str(ROOT) + os.pathsep + environment.get("PYTHONPATH", ""),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
    )
    result = subprocess.run(
        command,
        env=environment,
        cwd=output.parent,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    output.with_suffix(".log").write_text(result.stdout + result.stderr, encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Traceback" not in result.stdout + result.stderr


@pytest.mark.parametrize("axis", range(3), ids=("split-x", "split-y", "split-z"))
@pytest.mark.parametrize("precision", ("single", "double"))
@pytest.mark.parametrize("drive", ("overlap", "separated", "active"))
def test_transmission_line_stage_matches_serial(tmp_path, axis, precision, drive):
    """Batch six small scenes per launch, covering all TL axes/placements."""
    serial, distributed = tmp_path / "serial", tmp_path / "mpi"
    _run(serial, axis, precision, drive, distributed=False)
    _run(distributed, axis, precision, drive, distributed=True)

    tolerance = 2e-5 if precision == "single" else 2e-12
    for model, (polarisation, placement) in enumerate(CASES, start=1):
        context = f"{drive}, {precision}, split={'xyz'[axis]}, TL={polarisation}, {placement}"
        with h5py.File(f"{serial}{model}.h5") as reference, h5py.File(f"{distributed}{model}.h5") as actual:
            assert set(actual["tls"]) == set(reference["tls"])
            for line_name in reference["tls"]:
                line = reference[f"tls/{line_name}"]
                actual_line = actual[f"tls/{line_name}"]
                np.testing.assert_array_equal(actual_line.attrs["Position"], line.attrs["Position"])
                passive = drive != "active" or line_name == "tl2"
                for name in ("Vinc", "Iinc", "Vtotal", "Itotal"):
                    expected = line[name][...]
                    observed = actual_line[name][...]
                    assert expected.shape == observed.shape == (ITERATIONS + 1,)
                    assert np.isfinite(expected).all(), context
                    assert np.isfinite(observed).all(), context
                    if passive and name in ("Vinc", "Iinc"):
                        # Zero drive is not a dormant line: its total signal
                        # below must come from the external electromagnetic field.
                        np.testing.assert_array_equal(expected, 0)
                        np.testing.assert_array_equal(observed, 0)
                    scale = max(float(np.max(np.abs(expected))), 1e-30)
                    np.testing.assert_allclose(
                        observed,
                        expected,
                        rtol=tolerance,
                        atol=tolerance * scale,
                        err_msg=f"{context}, {line_name}/{name}",
                    )
                assert np.max(np.abs(line["Vtotal"][...])) > 1e-8, context
                assert np.max(np.abs(line["Itotal"][...])) > 1e-10, context

            # Compare colocated and remote receiver fields, not only the
            # feedback line's own output arrays.
            for receiver in ("rx1", "rx2"):
                for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
                    expected = reference[f"rxs/{receiver}/{component}"][...]
                    observed = actual[f"rxs/{receiver}/{component}"][...]
                    assert np.isfinite(expected).all(), context
                    assert np.isfinite(observed).all(), context
                    scale = max(float(np.max(np.abs(expected))), 1e-30)
                    np.testing.assert_allclose(
                        observed,
                        expected,
                        rtol=tolerance,
                        atol=tolerance * scale,
                        err_msg=f"{context}, {receiver}/{component}",
                    )
            assert np.max(np.abs(reference[f"rxs/rx2/E{polarisation}"][...])) > 1e-8, context


def _scene(axis, polarisation, placement, drive):
    import gprMax

    dl = 0.001
    feed_axis = "xyz".index(polarisation)
    position = np.full(3, 12, dtype=int)
    position[axis] = 15 if placement == "inside" else 12
    # Choose a contour difference crossing the split whenever it is transverse
    # to the TL. A parallel split has no transverse H samples in its halo.
    transverse = axis if axis != feed_axis else (feed_axis + 1) % 3
    magnetic_axis = 3 - feed_axis - transverse

    def point(cells):
        return tuple(float(value) * dl for value in cells)

    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl,) * 3))
    scene.add(gprMax.Domain(p1=(0.024,) * 3))
    scene.add(gprMax.PMLThickness(thickness=3))
    scene.add(gprMax.TimeWindow(iterations=ITERATIONS))
    scene.add(gprMax.Waveform(wave_type="gaussian", amp=0, freq=2e10, id="zero"))
    scene.add(gprMax.Waveform(wave_type="gaussian", amp=1, freq=2e10, id="voltage"))
    scene.add(gprMax.Waveform(wave_type="gaussian", amp=1e-5, freq=2e10, id="magnetic"))
    scene.add(
        gprMax.TransmissionLine(
            p1=point(position),
            polarisation=polarisation,
            resistance=50,
            waveform_id="voltage" if drive == "active" else "zero",
        )
    )
    if drive == "active":
        # Keep the receiving passive line on the same rank face when the
        # transmitting line is there, with a distinct electric feed edge.
        receiver_position = position.copy()
        receiver_position[(axis + 1) % 3] += 3
        scene.add(
            gprMax.TransmissionLine(
                p1=point(receiver_position),
                polarisation=polarisation,
                resistance=50,
                waveform_id="zero",
            )
        )
    else:
        writer_position = position.copy()
        writer_position[transverse] -= 1 if drive == "overlap" else 4
        scene.add(
            gprMax.MagneticDipole(
                p1=point(writer_position),
                polarisation="xyz"[magnetic_axis],
                waveform_id="magnetic",
            )
        )
    scene.add(gprMax.Rx(p1=point(position), id="feed"))
    scene.add(gprMax.Rx(p1=point(position + 1), id="remote"))
    return scene


def _worker(args):
    import gprMax

    partition = [1, 1, 1]
    partition[args.axis] = 2
    scenes = [_scene(args.axis, polarisation, placement, args.drive) for polarisation, placement in CASES]
    gprMax.run(
        scenes=scenes,
        n=len(scenes),
        outputfile=Path(args.output),
        mpi=tuple(partition) if args.mpi else None,
        cpu_precision=args.precision,
        hide_progress_bars=True,
        log_level=40,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--axis", type=int, choices=range(3), required=True)
    parser.add_argument("--precision", choices=("single", "double"), required=True)
    parser.add_argument("--drive", choices=("overlap", "separated", "active"), required=True)
    parser.add_argument("--mpi", action="store_true")
    _worker(parser.parse_args())
