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

"""Compatibility check for the optional distributed-FFT dependency."""

import numpy as np
import pytest


@pytest.mark.unit
@pytest.mark.parametrize("shape", [(8, 8), (20, 20, 20), (20, 20, 22)])
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_mpi4py_fft_fftw_round_trip(shape, dtype):
    """A one-rank FFTW transform exercises the compiled mpi4py-fft modules."""

    MPI = pytest.importorskip("mpi4py.MPI")
    mpi4py_fft = pytest.importorskip("mpi4py_fft")

    fft = mpi4py_fft.PFFT(
        MPI.COMM_SELF,
        shape,
        axes=tuple(range(len(shape))),
        dtype=dtype,
        backend="fftw",
        collapse=False,
    )
    values = mpi4py_fft.newDistArray(fft, False)
    transformed = mpi4py_fft.newDistArray(fft, True)
    values[...] = np.arange(values.size).reshape(values.shape)
    expected = np.asarray(values).copy()

    try:
        fft.forward(values, transformed)
        fft.backward(transformed, values)
        np.testing.assert_allclose(np.asarray(values), expected, rtol=1e-12, atol=1e-10)
    finally:
        fft.destroy()
