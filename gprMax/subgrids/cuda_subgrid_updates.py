# Copyright (C) 2015-2025: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, John Hartley,
#                          and Nathan Mannall
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax.  If not, see <http://www.gnu.org/licenses/>.

"""GPU subgrid updaters.

Everything CUDA-specific about driving an HSG subgrid lives here, so
subgrids/updates.py needs only a small dispatch change and its CPU classes
stay exactly as written.

    CUDASubgridUpdates    drives every subgrid, owns the CUDA context
    CUDASubgridUpdater    drives one subgrid, launches the interface kernels

Nothing in updates/cuda_updates.py needs changing. It already provides the
two things a second grid requires:

    __init__(G, shared=...)   attaches to an existing context instead of
                                creating one, with cleanup() guarded by
                                _owns_context. Added for virtual waveguides,
                                which have the same need - a kernel that
                                addresses two grids without per-step
                                transfers.
    CUDA implementations of store_snapshots, update_eigenmode_sources_*,
    observe_eigenmode_ports and update_network_terminals, so the extended
    HSG phase sequence runs on the GPU unchanged.

IMPORTANT - the phase sequence below is a verbatim copy of
SubgridUpdater.hsg_1 / hsg_2 and their helpers in subgrids/updates.py. It
cannot be inherited or referenced: those methods use zero-argument super(),
which binds to SubgridUpdater's MRO at compile time and would dispatch the
bulk updates back to the Cython path.

Because it is a copy, it can go stale. tools/check_subgrid_drift.py compares
the two by AST and fails if they diverge; run it after every upstream sync.
"""

import logging

import gprMax.config as config
from gprMax.cuda_opencl import knl_subgrid_hsg, knl_subgrid_precursors
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.subgrids.grid import SubGridBaseGrid
from gprMax.subgrids.precursor_nodes import PrecursorNodes

from ..updates.cuda_updates import CUDAUpdates

logger = logging.getLogger(__name__)

INTERFACE_KNLS = {
    "update_is": knl_subgrid_hsg.update_is,
    "update_electric_os": knl_subgrid_hsg.update_electric_os,
    "update_magnetic_os": knl_subgrid_hsg.update_magnetic_os,
}

PRECURSOR_KNLS = {
    "gather_taps": knl_subgrid_precursors.gather_taps,
    "bilinear_interp": knl_subgrid_precursors.bilinear_interp,
    "time_blend": knl_subgrid_precursors.time_blend,
}


def _upload_mat_coeffs(grid):
    """Put updatecoeffsE/H in global memory, once per grid.

    The plane-wave setup uploads these too, so skip it when they are already
    there rather than paying for a second transfer.
    """
    if getattr(grid, "updatecoeffsE_dev", None) is None:
        grid.htod_mat_coeff_arrays()


class CUDASubgridUpdater(CUDAUpdates):
    """Handles updating the electric and magnetic fields of an HSG subgrid on
    the GPU. The IS, OS, subgrid region and the electric/magnetic sources are
    updated using the precursor regions.

    The subgrid it drives is a CUDASubGridHSG, whose update_*_is/os methods
    launch kernels; the precursors are CUDAPrecursorNodes, which keep their
    slices on the device. Neither appears in the phase sequence, so the copy
    below stays textually identical to the CPU original.
    """

    def __init__(self, subgrid: SubGridBaseGrid, precursors: PrecursorNodes,
                 G: FDTDGrid, shared=None):
        """
        Args:
            subgrid: SubGrid3d instance to be updated.
            precursors (PrecursorNodes): PrecursorNodes instance nodes associated
                                            with the subgrid - contain interpolated
                                            fields.
            G: FDTDGrid class describing a grid in a model.
            shared: CUDAUpdates whose context this updater attaches to. The
                        main grid and the subgrid must share one context or
                        they cannot address each other's arrays.
        """
        super().__init__(subgrid, shared=shared)
        self.precursors = precursors
        self.G = G
        self.iteration = 0

        self._set_subgrid_knls()

    def _set_subgrid_knls(self):
        """HSG interface and precursor updates - prepares kernels, and gets
        kernel functions.

        The kernels take every array dimension as a runtime argument rather
        than through the macro preamble: a model may hold several subgrids of
        different sizes, so the dimensions cannot be baked in the way the bulk
        field kernels bake theirs.
        """
        # The interface kernels look material coefficients up per node and
        # take the tables as pointer arguments, so both grids need them in
        # global memory. _set_macros() does not upload them - only
        # _set_planewave_knls() does, and only when the model has a plane
        # wave - so a plain subgrid model would otherwise have no
        # updatecoeffsE_dev at all.
        _upload_mat_coeffs(self.grid)

        opts = config.sim_config.devices["nvcc_opts"]

        self.knls_interface = {}
        for name, knl in INTERFACE_KNLS.items():
            bld = self._build_knl(knl, self.subs_name_args, self.subs_func)
            module = self.source_module(bld, options=opts)
            self.knls_interface[name] = module.get_function(name)

        self.knls_precursor = {}
        for name, knl in PRECURSOR_KNLS.items():
            bld = self._build_knl(knl, self.subs_name_args, self.subs_func)
            module = self.source_module(bld, options=opts)
            self.knls_precursor[name] = module.get_function(name)

    # ---- verbatim from SubgridUpdater; see the module docstring ----------

    def store_outputs(self):
        super().store_outputs(self.iteration)
        super().store_snapshots(self.iteration)

    def update_electric_sources(self):
        iteration = self.iteration
        super().update_electric_sources(iteration)
        super().update_eigenmode_sources_electric(iteration)
        self.iteration += 1

    def update_magnetic_sources(self):
        super().update_magnetic_sources(self.iteration)
        super().update_eigenmode_sources_magnetic(self.iteration)
        super().observe_eigenmode_ports(self.iteration)

    def update_network_terminals(self):
        """Update sparse terminals at the fine-grid electric time step."""

        # update_electric_sources() has just advanced the shared fine-grid
        # iteration counter, so the terminal history index is one behind it.
        return super().update_network_terminals(self.iteration - 1)

    def hsg_1(self):
        """First half of the subgrid update. Takes the time step up to the main
        grid magnetic update.
        """

        G = self.G
        subgrid = self.grid
        precursors = self.precursors

        # Copy the main grid electric fields at the IS position
        precursors.update_electric()

        upper_m = int(subgrid.ratio / 2 - 0.5)

        for m in range(1, upper_m + 1):
            self.store_outputs()
            self.update_electric_a()
            self.update_electric_pml()
            precursors.interpolate_magnetic_in_time(int(m + subgrid.ratio / 2 - 0.5))
            subgrid.update_electric_is(precursors)
            self.update_electric_sources()
            self.update_electric_b()
            self.update_network_terminals()
            self.update_magnetic()
            self.update_magnetic_pml()
            precursors.interpolate_electric_in_time(m)
            subgrid.update_magnetic_is(precursors)
            self.update_magnetic_sources()

        self.store_outputs()
        self.update_electric_a()
        self.update_electric_pml()
        precursors.calc_exact_magnetic_in_time()
        subgrid.update_electric_is(precursors)
        self.update_electric_sources()
        self.update_electric_b()
        self.update_network_terminals()
        subgrid.update_electric_os(G)

    def hsg_2(self):
        """Second half of the subgrid update. Takes the time step up to the main
        grid electric update.
        """

        G = self.G
        subgrid = self.grid
        precursors = self.precursors

        # Copy the main grid magnetic fields at the IS position
        precursors.update_magnetic()

        upper_m = int(subgrid.ratio / 2 - 0.5)

        for m in range(1, upper_m + 1):
            self.update_magnetic()
            self.update_magnetic_pml()
            precursors.interpolate_electric_in_time(int(m + subgrid.ratio / 2 - 0.5))
            subgrid.update_magnetic_is(precursors)
            self.update_magnetic_sources()
            self.store_outputs()
            self.update_electric_a()
            self.update_electric_pml()
            precursors.interpolate_magnetic_in_time(m)
            subgrid.update_electric_is(precursors)
            self.update_electric_sources()
            self.update_electric_b()
            self.update_network_terminals()

        self.update_magnetic()
        self.update_magnetic_pml()
        precursors.calc_exact_electric_in_time()
        subgrid.update_magnetic_is(precursors)
        self.update_magnetic_sources()
        subgrid.update_magnetic_os(G)


class CUDASubgridUpdates(CUDAUpdates):
    """Updates for subgrids on the GPU.

    Owns the CUDA context; every subgrid updater attaches to it so that the
    main grid and the subgrids share one address space.
    """

    def __init__(self, G, updaters):
        super().__init__(G)
        self.updaters = updaters

    def hsg_1(self):
        """Updates the subgrids over the first phase."""
        for sg_updater in self.updaters:
            sg_updater.hsg_1()

    def hsg_2(self):
        """Updates the subgrids over the second phase."""
        for sg_updater in self.updaters:
            sg_updater.hsg_2()


def create_cuda_updates(model, subgrid_hsg_cls):
    """Build the CUDA subgrid solver.

    Mirrors create_updates() in subgrids/updates.py, but constructs the main
    updates object first so its context can be shared with each updater -
    separate contexts would not share device memory, and the two grids could
    not exchange fields at all.

    Args:
        model: model containing the main grid and subgrids.
        subgrid_hsg_cls: the SubGridHSG class, for the type check.

    Returns:
        CUDASubgridUpdates instance.
    """
    from .cuda_precursor_nodes import (
        CUDAPrecursorNodes,
        CUDAPrecursorNodesFiltered,
    )

    updates = CUDASubgridUpdates(model.G, [])

    # The OS kernels read the MAIN grid's coefficients as pointer arguments
    _upload_mat_coeffs(model.G)

    for sg in model.subgrids:
        if not issubclass(type(sg), subgrid_hsg_cls):
            logger.exception(f"{str(sg)} is not a subgrid type")
            raise ValueError

        cls = CUDAPrecursorNodesFiltered if sg.filter else CUDAPrecursorNodes
        precursors = cls(model.G, sg)

        sgu = CUDASubgridUpdater(sg, precursors, model.G, shared=updates)

        # Both grids' arrays are on the device by now, so the interface and
        # the precursor descriptors can be built against them.
        sg.setup_cuda_interface(sgu.knls_interface, model.G)
        precursors.setup_device(sgu.knls_precursor, sg.gpuarray, tpb=sg.tpb[0])

        updates.updaters.append(sgu)

    return updates
