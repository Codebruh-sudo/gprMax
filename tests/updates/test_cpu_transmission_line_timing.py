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

"""CPU baseline preservation when no magnetic writer touches a TL loop."""

import h5py
import numpy as np
import pytest

import gprMax
from gprMax.updates.cpu_updates import CPUUpdates

pytestmark = pytest.mark.integration


def _scene(passive):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(0.001, 0.001, 0.001)))
    scene.add(gprMax.Domain(p1=(0.012, 0.012, 0.012)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-10))
    scene.add(gprMax.Waveform(wave_type="gaussian", amp=1, freq=2e10, id="active"))
    scene.add(gprMax.TransmissionLine(polarisation="z", p1=(0.006, 0.006, 0.006), resistance=50, waveform_id="active"))
    if passive:
        scene.add(gprMax.Waveform(wave_type="gaussian", amp=0, freq=2e10, id="passive"))
        scene.add(
            gprMax.TransmissionLine(polarisation="z", p1=(0.008, 0.006, 0.006), resistance=50, waveform_id="passive")
        )
    scene.add(gprMax.Rx(p1=(0.006, 0.006, 0.006)))
    return scene


def _traces(path):
    result = {}
    with h5py.File(str(path) + ".h5", "r") as output:
        for group_name in ("tls", "rxs"):
            output[group_name].visititems(
                lambda name, obj: result.update({f"{group_name}/{name}": obj[:]})
                if isinstance(obj, h5py.Dataset)
                else None
            )
    return result


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("passive", [False, True], ids=["tl_only", "active_passive"])
def test_cpu_transmission_line_preserves_legacy_schedule(tmp_path, monkeypatch, precision, passive):
    current_path = tmp_path / "current"
    legacy_path = tmp_path / "legacy"
    options = dict(n=1, hide_progress_bars=True, cpu_precision=precision)
    gprMax.run(scenes=[_scene(passive)], outputfile=current_path, **options)

    sample = CPUUpdates.update_magnetic_edge_devices
    write_sources = CPUUpdates.update_magnetic_sources

    def legacy_magnetic_sources(self, iteration):
        sample(self, iteration)
        write_sources(self, iteration)

    with monkeypatch.context() as legacy:
        legacy.setattr(CPUUpdates, "update_magnetic_sources", legacy_magnetic_sources)
        legacy.setattr(CPUUpdates, "update_magnetic_edge_devices", lambda self, iteration: None)
        gprMax.run(scenes=[_scene(passive)], outputfile=legacy_path, **options)

    current, legacy = _traces(current_path), _traces(legacy_path)
    assert current.keys() == legacy.keys()
    assert np.max(np.abs(current["tls/tl1/Vtotal"])) > 1e-3
    if passive:
        assert np.max(np.abs(current["tls/tl2/Vtotal"])) > 1e-6
        np.testing.assert_array_equal(current["tls/tl2/Vinc"], 0)
    for name in current:
        np.testing.assert_array_equal(current[name], legacy[name], err_msg=name)
