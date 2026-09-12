# Dispersive material touching a surface impedance

All four driven validation cases passed. The implementation retains the clipped
H circulation and adds area-weighted bulk polarization histories to the local
scalar electric solve. Debye, Lorentz, Drude, and mixed exterior materials are
included. Boundary modal rows use the discrete polarization transfer.

Raw samples and machine information are in
[dispersive_impedance_2026-09-07.json](dispersive_impedance_2026-09-07.json).

## Reproduce

Run from the repository root after rebuilding the Cython extensions:

```console
python -m testing.benchmarking.benchmark_dispersive_impedance --cells 48 --iterations 250 --threads 4 --repeats 3 --explicit-orders 4 8 --kernel-iterations 1000 --kernel-repeats 3 --hot-iterations 250 --hot-repeats 3 --output testing/benchmarking/results/dispersive_impedance_2026-09-07.json
```

Use `--skip-timing` to run just the driven correctness checks. The command exits
nonzero if any numerical acceptance criterion fails. The original box benchmark
also accepts `--exterior debye|lorentz|drude|mixed` for an individual timing run.

## Driven 3-D validation

Each model uses an 8 x 8 x 8 grid, 1 mm cells, a central 2 x 2 x 2 impedance box,
a 10 GHz Ricker electric dipole, no PML, and PEC outer domain boundaries.
The receiver shares the source location. Runs use double precision, one CPU
thread, 2,400 steps, and dt = 1.92583320154647 ps (about 4.62 ns total).
The dynamic boundary is a four-pole copper fit over 8-12 GHz. The bulk material
has epsilon-infinity 3, conductivity 0.02 S/m, and two poles per material.
Exact pole parameters are in `add_exterior` in the box benchmark. The mixed
case divides the exterior into Debye, Lorentz, and Drude slabs.

| Exterior | Python/Cython receiver relative L2 error | Final-field relative L2 error | Last 200 samples: peak / run peak | PEC error at 0.01 ohm | PEC error at 0.001 ohm |
| --- | ---: | ---: | ---: | ---: | ---: |
| Debye | 3.99e-16 | 7.47e-13 | 3.90e-5 | 3.46e-6 | 3.46e-7 |
| Lorentz | 1.16e-15 | 1.05e-12 | 1.04e-4 | 2.30e-6 | 2.30e-7 |
| Drude | 5.01e-16 | 8.19e-14 | 1.42e-4 | 3.14e-6 | 3.14e-7 |
| Mixed | 8.03e-16 | 8.75e-13 | 3.90e-5 | 3.55e-6 | 3.55e-7 |

The PEC comparison is a separate resistive-wall sweep against an ordinary PEC
volume in the same dispersive exterior. Lowering resistance tenfold lowers
the receiver error approximately tenfold. Acceptance limits are: receiver
agreement below 1e-11, final-field agreement below 1e-9, tail peak ratio below
0.02, final PEC error below 1e-3, and at least fivefold PEC-error reduction.
Python/Cython agreement checks the two implementations; PEC-limit convergence
provides an independent limiting case. These checks do not establish continuum
accuracy at arbitrary impedance or unconditional stability for every pole set.

## CPU timing and storage

Windows 11, Intel64 Family 6 Model 186 Stepping 2, Python 3.14.4, NumPy 2.4.3,
MSVC 19.51, double precision, four OpenMP threads. Timing models use a 48 cubed
grid, a 24 cubed impedance box, five PML cells per side, and no sources. Each
exterior has its own ordinary-grid baseline with identical bulk materials.
Baseline and surface cases alternate order over three repeats. The hot loop
includes both dispersive electric stages. All source-free solves start at zero;
the isolated sparse kernel uses prescribed nonzero H fields.

The table selects the explicit four-pole copper case from the larger saved
sweep (which also includes a resistive boundary, automatic copper order, and
eight-pole copper). Every listed surface has 6,912 boundary E edges.

| Exterior | Additional bulk-polarization poles | Polarization state bytes | Polarization coefficient/index bytes | Sparse step median (microseconds) | Baseline solve (s) | Surface solve (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Free space | 0 | 0 | 0 | 67.58 | 0.4341 | 0.4065 |
| Debye | 13,824 | 221,184 | 801,796 | 84.80 | 0.7164 | 0.8058 |
| Lorentz | 13,824 | 221,184 | 801,796 | 60.26 | 0.5031 | 0.5242 |
| Drude | 13,824 | 221,184 | 801,796 | 64.66 | 0.8231 | 0.9136 |
| Mixed | 14,208 | 227,328 | 820,228 | 83.40 | 0.8650 | 0.8645 |

The extra arrays contain two state values and six coefficient values per pole,
two instantaneous corrections per edge, and one prefix-sum offset per edge
plus its terminal entry. They are empty when dispersion does not touch the
boundary. Mixed-material seams require additional pole histories.

Timings are descriptive measurements from a shared desktop. The raw repeats
show variability; apparent negative overhead or a faster complex-material
sample should not be interpreted as an intrinsic speedup or material ranking.
Use the saved samples and rerun on the target machine for performance decisions.

Independent analytical validations are now retained in
[the surface-impedance validation report](../../validation/impedance_surface/analytical_validation_report.md).
The 0.75 mm conductor-sphere runs have 0.855/0.832 dB backscatter RMS error
against Mie theory. Reflected plane-wave phase on the 0.5 mm mesh agrees with
the continuum wall coefficient to at most 0.001497 degree RMS, including a
Debye exterior. Both validation drivers pass their acceptance criteria;
these accuracy results complement the kernel timings and discrete checks above.
