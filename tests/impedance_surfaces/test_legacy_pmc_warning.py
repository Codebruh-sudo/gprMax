from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.materials import Material
from testing.validation.impedance_surface.validate_2d import compare_invariant_3d, run_2d


@pytest.mark.parametrize("custom", (False, True))
def test_warns_once_for_used_volume_or_component(custom):
    host = Material(0, "free_space")
    pmc = Material(1, "custom" if custom else "pmc")
    pmc.sm = np.inf
    grid = SimpleNamespace(
        name="main",
        materials=[host, pmc],
        solid=np.array([[[1]]]),
        ID=np.zeros((6, 1, 1, 1), dtype=np.uint32),
    )
    with patch("gprMax.grid.fdtd_grid.logger.warning") as warning:
        FDTDGrid._warn_legacy_pmc_geometry(grid)
        FDTDGrid._warn_legacy_pmc_geometry(grid)
    warning.assert_called_once()
    assert "half a cell" in warning.call_args[0][0]
    assert "resistance=float('inf')" in warning.call_args[0][0]


def test_unused_pmc_declaration_does_not_warn():
    grid = SimpleNamespace(
        name="main",
        materials=[Material(0, "free_space"), Material(1, "pmc")],
        solid=np.zeros((1, 1, 1)),
        ID=np.zeros((6, 1, 1, 1)),
    )
    with patch("gprMax.grid.fdtd_grid.logger.warning") as warning:
        FDTDGrid._warn_legacy_pmc_geometry(grid)
    warning.assert_not_called()


def test_te_internal_constraints_and_exact_pmc_do_not_warn(tmp_path):
    with patch("gprMax.grid.fdtd_grid.logger.warning") as warning:
        run_2d(tmp_path / "te", polarization="TE", geometry_only=True, steps=1)
    assert not any("Legacy PMC volume" in call.args[0] for call in warning.call_args_list)


def test_pmc_symmetry_planes_do_not_warn(tmp_path):
    with patch("gprMax.grid.fdtd_grid.logger.warning") as warning:
        compare_invariant_3d(tmp_path, polarization="TE", steps=1)
    assert not any("Legacy PMC volume" in call.args[0] for call in warning.call_args_list)
