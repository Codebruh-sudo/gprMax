"""Standalone check: the three CUDA HSG subgrid kernels vs their Cython originals.

Builds synthetic arrays, runs both implementations on identical inputs, and
compares element by element. No solver, no model file - each kernel is
isolated so any mismatch is unambiguous.

Both sides are float64 and perform the same multiply-add per element in the
same order, so the expected difference is EXACTLY zero. Anything nonzero is
a real index or logic error, not rounding.

Run from the gprMax root:
    !python test_subgrid_kernels.py
"""

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from gprMax.cython.fields_updates_hsg import (
    update_electric_os as cy_electric_os,
    update_is as cy_update_is,
    update_magnetic_os as cy_magnetic_os,
)

import knl_subgrid_hsg

REAL = "double"
DTYPE = np.float64
CUDA_IDX = "int i = blockIdx.x * blockDim.x + threadIdx.x;"
NMAT = 5
NY_MATCOEFFS = 5
TPB = 128

rng = np.random.default_rng(0)


def build(kernel, name):
    src = (kernel["args_cuda"].substitute(REAL=REAL) + "{"
           + kernel["func"].substitute(REAL=REAL, CUDA_IDX=CUDA_IDX) + "}")
    return SourceModule(src).get_function(name)


def to_dev(a):
    d = drv.mem_alloc(a.nbytes)
    drv.memcpy_htod(d, a)
    return d


# ---------------------------------------------------------------- update_is
def check_is():
    nwx, nwy, nwz, n = 12, 10, 14, 3
    F_NX, F_NY, F_NZ = nwx + 2 * n + 2, nwy + 2 * n + 2, nwz + 2 * n + 2
    fn = build(knl_subgrid_hsg.update_is, "update_is")

    cases = [(1, -1, 1, -1, 3, 4), (1, 0, -1, 1, 2, 1),
             (2, -1, -1, 1, 3, 3), (2, 0, 1, -1, 1, 0),
             (3, -1, 1, -1, 2, 5), (3, 0, -1, 1, 1, 2)]
    worst = 0.0
    for face, offset, sl, su, co, lid in cases:
        nwl, nwm = {1: (nwx, nwy), 2: (nwy, nwz), 3: (nwx, nwz)}[face]

        coeffs = rng.random((NMAT, NY_MATCOEFFS))
        ID = rng.integers(0, NMAT, (6, F_NX, F_NY, F_NZ), dtype=np.uint32)
        f0 = rng.random((F_NX, F_NY, F_NZ))
        inc_l = rng.random((nwl, nwm))
        inc_u = rng.random((nwl, nwm))

        f_cpu = f0.copy()
        cy_update_is(nwx, nwy, nwz, coeffs, ID, n, offset, nwl, nwm, 0,
                     face, f_cpu, inc_l, inc_u, lid, sl, su, co, 1)

        f_gpu = f0.copy()
        d_f = to_dev(f_gpu)
        total = nwl * nwm
        fn(np.int32(nwx), np.int32(nwy), np.int32(nwz), np.int32(n),
           np.int32(offset), np.int32(nwl), np.int32(nwm), np.int32(face),
           np.int32(sl), np.int32(su), np.int32(co), np.int32(lid),
           np.int32(NY_MATCOEFFS),
           np.int32(F_NX), np.int32(F_NY), np.int32(F_NZ),
           np.int32(F_NX), np.int32(F_NY), np.int32(F_NZ), np.int32(nwm),
           to_dev(coeffs), to_dev(ID), d_f, to_dev(inc_l), to_dev(inc_u),
           block=(TPB, 1, 1), grid=((total + TPB - 1) // TPB, 1, 1))
        drv.memcpy_dtoh(f_gpu, d_f)

        d = np.abs(f_cpu - f_gpu).max()
        worst = max(worst, d)
        print(f"  update_is        face={face} off={offset:2d} co={co}  "
              f"changed={np.count_nonzero(f_cpu != f0):5d}  max|diff|={d:.3e}")
    return worst


# ------------------------------------------------------- update_*_os shared
def check_os(cy_fn, knl, name, label):
    # main grid and subgrid geometry
    r, s, nb = 3, 2, 6           # ratio, IS/OS separation, boundary cells
    nwn = 9                      # working cells normal to the face
    MG = 40                      # main grid dimension (cube, for simplicity)
    SG = 60                      # subgrid dimension (cube)
    fn = build(knl, name)

    cases = [(1, 1, -1, 1, 3, 4), (1, 0, -1, 1, 2, 1),
             (2, 1, 1, -1, 3, 3), (2, 0, 1, -1, 1, 0),
             (3, 1, -1, 1, 2, 5), (3, 0, 1, -1, 1, 2)]
    worst = 0.0
    for face, mid, sign_n, sign_f, co, lid in cases:
        l_l, l_u = 5, 5 + 8
        m_l, m_u = 4, 4 + 7
        n_l, n_u = 7, 20

        coeffs = rng.random((NMAT, NY_MATCOEFFS))
        ID = rng.integers(0, NMAT, (6, MG, MG, MG), dtype=np.uint32)
        f0 = rng.random((MG, MG, MG))
        inc = rng.random((SG, SG, SG))

        f_cpu = f0.copy()
        cy_fn(coeffs, ID, face, l_l, l_u, m_l, m_u, n_l, n_u, nwn, lid,
              f_cpu, inc, co, sign_n, sign_f, mid, r, s, nb, 1)

        f_gpu = f0.copy()
        d_f = to_dev(f_gpu)
        total = (l_u - l_l) * (m_u - m_l)
        fn(np.int32(face), np.int32(l_l), np.int32(l_u),
           np.int32(m_l), np.int32(m_u), np.int32(n_l), np.int32(n_u),
           np.int32(nwn), np.int32(lid), np.int32(co),
           np.int32(sign_n), np.int32(sign_f), np.int32(mid),
           np.int32(r), np.int32(s), np.int32(nb), np.int32(NY_MATCOEFFS),
           np.int32(MG), np.int32(MG), np.int32(MG),
           np.int32(MG), np.int32(MG),
           np.int32(SG), np.int32(SG),
           to_dev(coeffs), to_dev(ID), d_f, to_dev(inc),
           block=(TPB, 1, 1), grid=((total + TPB - 1) // TPB, 1, 1))
        drv.memcpy_dtoh(f_gpu, d_f)

        d = np.abs(f_cpu - f_gpu).max()
        worst = max(worst, d)
        print(f"  {label:16s} face={face} mid={mid}  co={co}  "
              f"changed={np.count_nonzero(f_cpu != f0):5d}  max|diff|={d:.3e}")
    return worst


print("CUDA vs Cython - HSG subgrid interface kernels (float64)\n")
w1 = check_is()
print()
w2 = check_os(cy_electric_os, knl_subgrid_hsg.update_electric_os,
              "update_electric_os", "update_electric_os")
print()
w3 = check_os(cy_magnetic_os, knl_subgrid_hsg.update_magnetic_os,
              "update_magnetic_os", "update_magnetic_os")

worst = max(w1, w2, w3)
print("\n" + "=" * 60)
print(f"worst difference across all kernels: {worst:.3e}")
print("PASS - kernels are exact" if worst == 0.0
      else "MISMATCH - index or logic error, investigate")
