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

"""Fine-grid transmission lines sample every completed magnetic time level."""

import h5py
import numpy as np
import pytest

import gprMax
from gprMax.sources import TransmissionLine


FINE_ITERATIONS = 60
FEED = (0.048, 0.048, 0.048)
RECEIVER = (0.050, 0.049, 0.048)


def _scene(*, ratio, active, use_subgrid):
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.096, 0.096, 0.096)))
    spacing = 0.001 * ratio if use_subgrid else 0.001
    scene.add(gprMax.Discretisation(p1=(spacing,) * 3))
    scene.add(gprMax.TimeWindow(iterations=FINE_ITERATIONS // ratio if use_subgrid else FINE_ITERATIONS))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.OMPThreads(1))

    owner = scene
    if use_subgrid:
        owner = gprMax.SubGridHSG(
            p1=(0.018, 0.018, 0.018),
            p2=(0.078, 0.078, 0.078),
            ratio=ratio,
            id="fine_grid",
        )
        scene.add(owner)

    owner.add(gprMax.Waveform(wave_type="gaussian", amp=float(active), freq=2e10, id="line_pulse"))
    owner.add(gprMax.Waveform(wave_type="gaussian", amp=1e-5, freq=2e10, id="magnetic_pulse"))
    # Iteration-defined coarse/fine models have different inherited final
    # source times. Use the same explicit window, inside both time axes.
    owner.add(
        gprMax.TransmissionLine(
            p1=FEED,
            polarisation="z",
            resistance=50,
            waveform_id="line_pulse",
            start=0,
            stop=1e-10,
        )
    )
    # Hy at this integer coordinate is half an x-cell from the TL's Ez
    # edge. It writes a contour sample, not the electric terminal itself.
    owner.add(
        gprMax.MagneticDipole(
            p1=FEED,
            polarisation="y",
            waveform_id="magnetic_pulse",
            start=0,
            stop=1e-10,
        )
    )
    owner.add(gprMax.Rx(p1=RECEIVER, id="remote"))
    return scene


@pytest.mark.integration
@pytest.mark.parametrize("ratio", [1, 3])
@pytest.mark.parametrize("active", [False, True], ids=["passive", "active"])
def test_subgrid_tl_sampling_matches_uniform_fine_grid(tmp_path, monkeypatch, ratio, active):
    """Check real TL updates and current/voltage/field parity before IS returns.

    The source is 30 fine cells from each inner surface. The 60-step window
    ends before an interface-scattered signal returns to the source/receiver
    with appreciable amplitude, so ratio-3 coupling error is not conflated
    with source-sampling order. Ratio-1 long-time phase parity is separately
    covered by test_subgrid_integration.py.
    """

    calls = []
    original = TransmissionLine.update_magnetic

    def record_update(self, iteration, *args):
        calls.append(iteration)
        return original(self, iteration, *args)

    monkeypatch.setattr(TransmissionLine, "update_magnetic", record_update)
    paths = []
    for use_subgrid in (False, True):
        calls.clear()
        path = tmp_path / ("subgrid" if use_subgrid else "uniform")
        gprMax.run(
            scenes=[_scene(ratio=ratio, active=active, use_subgrid=use_subgrid)],
            outputfile=path,
            subgrid=use_subgrid,
            autotranslate=use_subgrid,
            cpu_precision="double",
            hide_progress_bars=True,
            log_level=40,
        )
        assert calls == list(range(FINE_ITERATIONS))
        paths.append(path.with_suffix(".h5"))

    with h5py.File(paths[0], "r") as plain, h5py.File(paths[1], "r") as nested:
        fine = nested["subgrids/fine_grid"]
        assert fine.attrs["Iterations"] == FINE_ITERATIONS
        assert fine.attrs["dt"] == pytest.approx(plain.attrs["dt"], rel=1e-13)
        np.testing.assert_allclose(fine.attrs["dx_dy_dz"], (0.001,) * 3)
        np.testing.assert_allclose(fine["tls/tl1"].attrs["Position"], FEED)
        np.testing.assert_allclose(fine["srcs/src1"].attrs["Position"], FEED)
        np.testing.assert_allclose(fine["rxs/rx1"].attrs["Position"], RECEIVER)
        assert fine["tls/tl1"].attrs["TimeCurrentOffset"] == pytest.approx(-0.5 * fine.attrs["dt"])

        for dataset in ("Vtotal", "Itotal", "Itotal_spectrum"):
            expected = plain[f"tls/tl1/{dataset}"][:]
            observed = fine[f"tls/tl1/{dataset}"][:]
            assert np.linalg.norm(expected) > 0
            relative_l2 = np.linalg.norm(observed - expected) / np.linalg.norm(expected)
            assert relative_l2 < 1e-10, (dataset, relative_l2)

        for family in ("E", "H"):
            expected = np.stack([plain[f"rxs/rx1/{family}{axis}"][:] for axis in "xyz"])
            observed = np.stack([fine[f"rxs/rx1/{family}{axis}"][:] for axis in "xyz"])
            assert np.linalg.norm(expected) > 0
            relative_l2 = np.linalg.norm(observed - expected) / np.linalg.norm(expected)
            assert relative_l2 < 1e-10, (family, relative_l2)
