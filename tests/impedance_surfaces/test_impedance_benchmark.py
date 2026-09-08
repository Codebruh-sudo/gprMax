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

"""Configuration and reporting checks for the impedance-box benchmark."""

from types import SimpleNamespace

import numpy as np
import pytest

from gprMax.user_objects.cmds_multiuse import SurfaceImpedance
from testing.benchmarking.benchmark_impedance_box import (
    SurfaceCase,
    _surface_metadata,
    benchmark_cases,
    build_scene,
)


def test_benchmark_cases_cover_resistance_auto_and_explicit_orders():
    cases = benchmark_cases((4, 8, 4, 16))

    assert [case.key for case in cases] == [
        "baseline",
        "resistive",
        "metal_auto",
        "metal_order_4",
        "metal_order_8",
        "metal_order_16",
    ]
    assert [case.requested_order for case in cases] == [None, None, "auto", 4, 8, 16]


def test_benchmark_rejects_nonpositive_explicit_orders():
    with pytest.raises(ValueError, match="positive"):
        benchmark_cases((4, 0, 8))


@pytest.mark.parametrize(
    "case,expected_order",
    (
        (SurfaceCase("auto", "metal", "auto"), "auto"),
        (SurfaceCase("explicit", "metal", 4), 4),
    ),
)
def test_metal_scene_preserves_requested_fit_mode(case, expected_order):
    scene = build_scene(
        24,
        1,
        1,
        case,
        fit_fmin_hz=8e9,
        fit_fmax_hz=12e9,
        fit_tolerance=2e-3,
    )
    surfaces = [item for item in scene.grid_objects if isinstance(item, SurfaceImpedance)]

    assert len(surfaces) == 1
    assert surfaces[0].fit_order == expected_order
    assert surfaces[0].fit_frequency_range == (8e9, 12e9)


def test_surface_metadata_reports_one_state_per_port_pole():
    model = SimpleNamespace(
        order=3,
        fit_requested_order="auto",
        fit_tolerance=2e-3,
        fit_max_relative_error=1e-3,
        fit_rms_relative_error=5e-4,
    )
    system = SimpleNamespace(
        model_ids=("wall",),
        port_count=7,
        state_y=np.zeros(21, dtype=np.float64),
        model_f=np.zeros(3, dtype=np.float64),
        model_q=np.zeros(3, dtype=np.float64),
        model_Z0=np.zeros(1, dtype=np.float64),
        edge_runtime=np.zeros((5, 2), dtype=np.float64),
        port_g_over_Z0=np.zeros(7, dtype=np.float64),
        port_inv_Z0=np.zeros(7, dtype=np.float64),
    )
    grid = SimpleNamespace(
        surface_impedance_models={"wall": model},
        Ex=np.zeros(1, dtype=np.float64),
    )

    result = _surface_metadata(grid, system)

    assert result["selected_poles"] == 3
    assert result["port_state_values"] == 21
    assert result["port_state_bytes"] == 21 * 8
    assert result["packed_state_array_bytes"] == 21 * 8
    assert result["model_local_coefficient_bytes"] == 7 * 8
    assert result["precomputed_edge_port_bytes"] == (10 + 7 + 7) * 8


@pytest.mark.integration
def test_mixed_dispersive_benchmark_checks_driven_fields_and_pec_limit(tmp_path):
    from testing.benchmarking.benchmark_dispersive_impedance import validate_contact

    result = validate_contact("mixed", tmp_path)
    assert result["passed"]
    assert result["boundary_polarization_poles"] > 0


@pytest.mark.integration
def test_dispersive_timing_executes_both_bulk_stages():
    from testing.benchmarking.benchmark_impedance_box import _parser, run_benchmark

    args = _parser().parse_args([
        "--cells", "24", "--iterations", "1", "--threads", "1", "--repeats", "1",
        "--kernel-iterations", "1", "--kernel-repeats", "1", "--hot-iterations", "1",
        "--hot-repeats", "1", "--explicit-orders", "4", "--exterior", "debye",
    ])
    result = run_benchmark(args)
    assert result["baseline"]["bulk_hot_median_seconds"] > 0
    assert all(row["polarization_poles"] > 0 for row in result["surface_cases"])
