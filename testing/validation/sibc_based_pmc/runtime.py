"""Validation-only exact PMC limit; public material APIs are unchanged.

A finite, zero-order SIBC declaration supplies geometry. Its discrete load
is replaced by Z0=+inf *before* the production compiler forms reciprocals.
Thus every runtime surface admittance is exactly zero, with no Foster states.
No field-update kernel, geometry weight, bulk mass, or timestep is replaced.
The declaration's finite resistance is a placeholder, not the simulated load;
the validation JSON records this override. Cached HDF5 material definitions
alone must not be used to reproduce an exact-PMC run.
"""

from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import patch

import numpy as np

import gprMax
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.impedance_surfaces import SurfaceImpedanceModel


@contextmanager
def exact_pmc(model_ids=("wall",)):
    original = SurfaceImpedanceModel.discretise

    def discretise(model, dt):
        discrete = original(model, dt)
        if model.ID in model_ids:
            if model.order:
                raise ValueError("The exact PMC diagnostic requires a zero-order surface model")
            return replace(discrete, Z0=np.inf)
        return discrete

    with patch.object(SurfaceImpedanceModel, "discretise", discretise):
        yield


def assert_zero_admittance(grid):
    system = grid.impedance_surfaces
    assert system is not None and system.edge_count > 0
    assert not np.any(system.model_info[:, 0])
    assert np.all(np.isposinf(system.model_Z0))
    assert np.all(system.port_inv_Z0 == 0)
    assert np.all(system.port_g_over_Z0 == 0)
    assert np.all(np.isfinite(system.edge_runtime))
    return system


def build_grid(scene, path, precision="double"):
    captured = []
    original = FDTDGrid.build

    def capture(grid):
        original(grid)
        captured.append(grid)

    with patch.object(FDTDGrid, "build", capture):
        gprMax.run(
            scenes=[scene],
            outputfile=path,
            geometry_only=True,
            cpu_precision=precision,
            hide_progress_bars=True,
            log_level=40,
        )
    return captured[0]


def base_scene(cells, spacing, *, iterations=1):
    scene = gprMax.Scene()
    for obj in (
        gprMax.Domain(p1=tuple(np.asarray(cells) * spacing)),
        gprMax.Discretisation(p1=tuple(spacing)),
        # All references use the same time grid as the protected SIBC.
        gprMax.TimeStepStabilityFactor(f=0.99),
        gprMax.TimeWindow(iterations=iterations),
        gprMax.PMLThickness(thickness=0),
        gprMax.OMPThreads(1),
    ):
        scene.add(obj)
    return scene
