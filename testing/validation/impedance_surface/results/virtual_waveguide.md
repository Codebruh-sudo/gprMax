# Surface-impedance virtual-waveguide validation

The mixed PEC/SIBC rectangular guide is solved with its physical rear intact,
then with a virtual guide replacing that rear at the same Yee plane. Auxiliary
and physical PML locations coincide. Both center and wall receivers record all
six fields. The pulse test covers every propagation axis and direction with
HORIPML and MRIPML. All runs use double precision and timestep factor 0.99.

| Surface model | Passive comparisons | Maximum normalized trace error | Maximum active reflection |
|---|---:|---:|---:|
| exact_pmc | 12 | 0.000e+00 | 1.681e-04 |
| constant_1000_ohm | 12 | 0.000e+00 | 3.551e-04 |
| copper_8_poles | 12 | 0.000e+00 | 1.638e-03 |

Passive acceptance is a maximum normalized six-field trace difference below
2e-8. Each channel uses its own reference peak, floored at 1e-12 of the largest
channel peak. Active runs use a 22 GHz continuous sine and require |S11| < 0.01,
finite fields/ADE states, and nonzero surface E on the auxiliary source plane.
The active test exercises the sparse source correction and Foster history;
the passive comparisons exercise full-mass aperture closure and PML histories.

The copper case uses the public copper preset fitted with eight Foster poles
over 0.1–100 GHz. The zero-admittance case uses public resistance=inf. The finite
constant case uses 1000 ohms. These measurements cover the stated scenes rather
than establishing unconditional stability for arbitrary material realizations.

The supported extension is a CPU 3D guide with longitudinally invariant walls,
lossless nondispersive retained materials, and surfaces strictly inside the
modal window. The window includes an opaque voxel beyond each wall. Auxiliary
surface rows retain the full main-grid dual-cell masses; unused tangential-H
padding samples supply the missing main-side curl at the aperture. Surface
source increments use the local implicit denominator and update ADE histories.

Reproduce from the repository root:

```powershell
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
python -m testing.validation.impedance_surface.virtual_waveguide
```

The JSON records individual cases, timestep, sparse boundary counts, PML row
counts, and active ADE-state counts. HDF5 run outputs remain in the workdir.
