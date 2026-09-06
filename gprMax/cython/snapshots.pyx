# cython: cdivision=True
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

from cython.parallel import prange
from cython cimport view

from gprMax.config cimport float_or_double


cdef inline float_or_double interpolate_native(
    float_or_double[:, :, ::view.contiguous] field, int i, int j, int k,
    int dx, int dy, int dz, int sx, int sy, int sz,
    int ox, int oy, int oz
) noexcept nogil:
    # Twice the native fractional index relative to the coarse lower corner.
    # Yee offsets ox/oy/oz are measured in half native cells.
    cdef int qx = sx * (dx - ox)
    cdef int qy = sy * (dy - oy)
    cdef int qz = sz * (dz - oz)
    cdef int a, b, c
    cdef float_or_double value = 0
    for a in range(1 + qx % 2):
        for b in range(1 + qy % 2):
            for c in range(1 + qz % 2):
                value = value + field[i * dx + qx // 2 + a,
                                      j * dy + qy // 2 + b,
                                      k * dz + qz // 2 + c]
    return value / ((1 + qx % 2) * (1 + qy % 2) * (1 + qz % 2))


cpdef void calculate_snapshot_fields(
    int nx,
    int ny,
    int nz,
    int nthreads,
    bint isEx,
    bint isEy,
    bint isEz,
    bint isHx,
    bint isHy,
    bint isHz,
    float_or_double[:, :, ::view.contiguous] Exslice,
    float_or_double[:, :, ::view.contiguous] Eyslice,
    float_or_double[:, :, ::view.contiguous] Ezslice,
    float_or_double[:, :, ::view.contiguous] Hxslice,
    float_or_double[:, :, ::view.contiguous] Hyslice,
    float_or_double[:, :, ::view.contiguous] Hzslice,
    float_or_double[:, :, ::1] Exsnap,
    float_or_double[:, :, ::1] Eysnap,
    float_or_double[:, :, ::1] Ezsnap,
    float_or_double[:, :, ::1] Hxsnap,
    float_or_double[:, :, ::1] Hysnap,
    float_or_double[:, :, ::1] Hzsnap,
    int sx=1,
    int sy=1,
    int sz=1,
    int dx=1,
    int dy=1,
    int dz=1
):
    """Calculates electric and magnetic values at points from averaging values
        in cells.

    Args:
        nx, ny, nz: ints for size of snapshot array.
        nthreads: int for number of threads to use.
        is: boolean to determine whether that field snapshot is required.
        slice: native field-array views starting at the lower snapshot corner.
            Only their last axis must be contiguous: leading-axis strides let
            an interior ROI share the original grid's storage without copies.
        snap: memoryviews to access snapshot arrays.
        sx, sy, sz: neighbour-offset strides along x, y, z (1 = genuine
            averaging with the +1 neighbour, as in 3D/2D-TM mode; 0 = no
            genuine neighbour exists along that axis, so both terms of any
            pair on that axis collapse to the same index - used for a 2D
            TE-mode model's invariant axis, where there is only ever one
            real field value flanked by forced-zero boundary padding, not
            a second genuine value to average against. Defaults to 1 for
            every axis, reproducing the original (pre-2D-TE-mode) formula
            exactly.
        dx, dy, dz: coarse output-cell widths in native cells. Components
            are interpolated at the declared coarse-cell centre, not averaged
            over the coarse cell. Defaults preserve the stride-one API.
    """

    cdef Py_ssize_t i, j, k

    for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
        for j in range(ny):
            for k in range(nz):
                if dx != 1 or dy != 1 or dz != 1:
                    if isEx:
                        Exsnap[i,j,k] = interpolate_native(Exslice, i,j,k, dx,dy,dz, sx,sy,sz, 1,0,0)
                    if isEy:
                        Eysnap[i,j,k] = interpolate_native(Eyslice, i,j,k, dx,dy,dz, sx,sy,sz, 0,1,0)
                    if isEz:
                        Ezsnap[i,j,k] = interpolate_native(Ezslice, i,j,k, dx,dy,dz, sx,sy,sz, 0,0,1)
                    if isHx:
                        Hxsnap[i,j,k] = interpolate_native(Hxslice, i,j,k, dx,dy,dz, sx,sy,sz, 0,1,1)
                    if isHy:
                        Hysnap[i,j,k] = interpolate_native(Hyslice, i,j,k, dx,dy,dz, sx,sy,sz, 1,0,1)
                    if isHz:
                        Hzsnap[i,j,k] = interpolate_native(Hzslice, i,j,k, dx,dy,dz, sx,sy,sz, 1,1,0)
                    continue
                # Keep the original stride-one arithmetic order exactly.
                # The electric field component value at a point comes from the
                # average of the 4 electric field component values in that cell.
                if isEx:
                    Exsnap[i, j, k] = (Exslice[i, j, k] +
                                       Exslice[i, j + sy, k] +
                                       Exslice[i, j, k + sz] +
                                       Exslice[i, j + sy, k + sz]) / 4
                if isEy:
                    Eysnap[i, j, k] = (Eyslice[i, j, k] +
                                       Eyslice[i + sx, j, k] +
                                       Eyslice[i, j, k + sz] +
                                       Eyslice[i + sx, j, k + sz]) / 4
                if isEz:
                    Ezsnap[i, j, k] = (Ezslice[i, j, k] +
                                       Ezslice[i + sx, j, k] +
                                       Ezslice[i, j + sy, k] +
                                       Ezslice[i + sx, j + sy, k]) / 4

                # The magnetic field component value at a point comes from
                # average of 2 magnetic field component values in that cell and
                # the neighbouring cell.
                if isHx:
                    Hxsnap[i, j, k] = (Hxslice[i, j, k] +
                                       Hxslice[i + sx, j, k]) / 2
                if isHy:
                    Hysnap[i, j, k] = (Hyslice[i, j, k] +
                                       Hyslice[i, j + sy, k]) / 2
                if isHz:
                    Hzsnap[i, j, k] = (Hzslice[i, j, k] +
                                       Hzslice[i, j, k + sz]) / 2
