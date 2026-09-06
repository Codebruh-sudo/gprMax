# Release correction verification — 6 September 2026

Base: `63160a96931a4be08da3ac980adcc2a0141f7b79` (`devel`, PR827).

This report covers the accumulated release-review corrections and the
subsequent adversarial counterexamples. It distinguishes regression parity
from independent analytical validation. Passing these checks does not prove
that every possible model is correct.

## Corrections and regression evidence

| Area | Defect and correction | Maintained regression tests |
|---|---|---|
| MPI source/receiver motion | Translate both the current coordinate and stepping origin during rank migration; complete outstanding sends before reusing their buffers. | `tests/grid/test_mpi_release_regressions.py` |
| MPI and task-farm failures | Return a nonzero status for failed distributed solves. Collect pickle-safe worker errors, finish other submitted tasks, join workers, then report failure without deleting successful outputs. | `tests/grid/test_mpi_release_regressions.py`, `tests/utilities/test_taskfarm_failures.py` |
| MPI network-port construction | Use the shared port-ID reservation helper on every rank. | `tests/ports/test_mpi_port_gather.py`, `tests/grid/test_mpi_release_regressions.py` |
| Geometry export | Include materials referenced by cells as well as Yee components. Compact before narrowing file indices; reject an oversized exported catalogue. Broadcast collective metadata and write the JSON catalogue only on the coordinator. | `tests/outputs/test_geometry_objects.py`, `tests/grid/test_mpi_release_regressions.py` |
| Small dispersive-pole increments | Evaluate existing exponential differences with `expm1` and double-precision intermediates. Warn if the stored decay multiplier rounds to one. | `tests/materials/test_materials.py`, `tests/materials/test_small_pole_transient.py` |
| Accelerator snapshot resources | Keep allocation/indexing dimensions per updater instead of accumulating historical maxima; download each component history once. Preserve MPI HDF5 snapshot precision. | `tests/outputs/test_snapshot_devices.py`, `tests/updates/test_gpu_snapshot_indexing.py`, `tests/grid/test_mpi_release_regressions.py` |
| CUDA context cleanup | Release owned contexts on partial construction and failed solves, without destroying shared virtual-guide contexts or masking the original error. | `tests/updates/test_accelerator_resource_lifecycle.py` |
| CUDA mixed sources (R1) | Initialise material coefficients in **each** independently compiled source-family module, not just the last module. | `tests/updates/test_cuda_source_module_initialisation.py`, `tests/updates/test_cuda_mixed_source_solve.py` |
| Shared waveform windows (R2) | Reuse a sampled history only when waveform configuration, start/stop window, grid timing and precision match. Exclude replaced or study-scaled histories from construction-time reuse. | `tests/sources/test_source_waveform_cache.py`, `tests/updates/test_source_windows_solve.py` |
| Hard voltage-source activity (R3) | Distinguish an inactive source from an active zero-valued sample. Append inclusive activity bounds to voltage-source device metadata; preserve soft-source arithmetic and existing coordinate strides. | `tests/sources/test_voltage_source_activity.py`, `tests/updates/test_source_windows_solve.py` |
| TL magnetic-contour timing (R4) | Sample after all magnetic source/TFSF corrections and the required MPI halo exchange. Apply the same stage to serial, accelerator and fine-subgrid updates. | `tests/updates/test_cpu_transmission_line_timing.py`, `tests/updates/test_mpi_transmission_line_timing.py`, `tests/updates/test_mpi_transmission_line_solve.py`, `tests/ports/test_tl_sampling_subgrid.py` |
| Magnetic volume overwrite (R5) | Let a volume own its H-rigidity claims through its own cells, so overwriting that volume removes its claims without removing neighboring objects' claims. Keep explicit magnetic-edge setters and PEC magnetic behavior unchanged. | `tests/geometry_primitives/test_volume_overwrite.py`, `tests/geometry_objects/test_geometry_overwrite_solve.py` |
| Transparent component imports (R6) | Preserve existing rigidity for each transparent component ID, independently of the cell-material/tag mask. | `tests/outputs/test_geometry_import_transparency.py`, `tests/outputs/test_geometry_objects_read.py` |
| Wide imported material IDs (R7) | Preserve unsigned file indices until validation; map to signed 32-bit global IDs before voxel construction. Retain the existing 16-bit Cython caller path as a fused specialization. | `tests/outputs/test_geometry_objects_read.py`, `tests/outputs/test_geometry_import_transparency.py` |
| Reduced-dimensional MPI outputs (R8) | Gather the validated active electric components: one for TM, two for TE, three for 3-D. Validate monitor signatures and payload shapes collectively. SAR and radiometry share this correction. | `tests/outputs/test_sar_mpi_components.py`, `tests/outputs/test_mpi_reduced_outputs.py` |
| Strided snapshot coordinates (R9) | Interpolate native Yee samples at each declared coarse-cell center; do not interpolate already-strided corners. Exchange only required native samples on requested MPI snapshot iterations. Reject stencils outside native grid support. | `tests/outputs/test_snapshot_native_collocation.py`, `tests/outputs/test_snapshot_native_devices.py`, `tests/outputs/test_mpi_reduced_outputs.py` |
| Padded snapshot launches | Guard the shared CUDA/OpenCL/Metal kernel before decoding its index, removing modulo wrapping and duplicate writers. Use the updater's maximum snapshot-volume range for OpenCL. | `tests/updates/test_snapshot_single_writer.py`, `tests/outputs/test_snapshot_overlaunch_devices.py` |
| OpenCL PML dispatch (R10) | Bound each kernel launch by the larger of its two spatial Phi-history extents, rather than the full six-component material array. Preserve all existing PML equations and guards. | `tests/pml/test_opencl_pml_dispatch.py`, `tests/pml/test_opencl_pml_range_solve.py`, `tests/pml/test_opencl_pml_geometry_fixed.py` |

## Numerical meaning of the corrections

For an existing dispersive pole represented by `w` and `q`, the cancellation-
resistant expressions are

```math
z_t=-\frac{w}{q\Delta t}\operatorname{expm1}(q\Delta t),\qquad
z_{t,2}=-\frac{w}{q}\operatorname{expm1}(q\Delta t/2).
```

These are algebraically equivalent evaluations of the existing recurrence,
not a new material model. State storage still uses the selected solver
precision; unresolved long-time decay may require double precision.

A hard voltage source constrains its electric edge only when
`start <= n*dt <= stop`. A zero waveform sample within this interval still
imposes zero electric field; outside it the ordinary field solution is left
unchanged by the source. Nonzero source resistance still represents a physical
load, including when its drive is zero.

For snapshot lower corner `r`, native spacing `d` and integer stride `s`, all
components must represent `r + s*d/2` (componentwise). An affine test field
`f(x,y,z)=x+2y+3z`, sampled at the true Yee positions, is an independent
interpolation oracle. With `d=(1,2,3) mm` and `s=(2,2,2)`, the first declared
center has `f=0.014`. Before correction, the six component outputs ranged
from `0.0075` to `0.0135`, despite CPU/CUDA agreement. The new interpolation
must reproduce the common value to the selected precision. It is point
collocation, **not** a spatial volume average or temporal interpolation.

## Verification record

All final selected tests passed. Counts below are distinct test identities
within each backend, not the sum of every repeated diagnostic run.

| Selection | Result | Scope |
|---|---:|---|
| Standard CPU, `not gpu and not slow` | 7,299 passed | Full fresh selection; 16 reported skips including three collection skips |
| Separate slow CPU batch | 50 passed | Includes real MPI solves and parallel FFTW |
| New reduced-output MPI matrix | 56 passed | Additional slow cases, not present in the earlier slow-batch collection |
| Final snapshot single-writer CPU checks | 3 new identities passed | Eleven focused tests passed on both Python 3.12 and 3.13; eight repeat existing tests |
| Reconciled Python 3.12 CPU inventory | **7,408 passed, 13 test skips, zero missing/failing tests** | 7,421 collected non-GPU identities, plus separate collection skips |
| CPU-only PyCUDA-import test | 1 passed | Separately exercised under Python 3.13; unavailable in the 3.12 environment |
| CUDA | **125 passed** | Complete applicable collected hardware inventory; no selected hardware skips |
| NVIDIA OpenCL | **107 passed** | Complete applicable collected hardware inventory |
| Intel CPU OpenCL | **107 passed** | Same OpenCL identities on a second implementation |

Thus 7,409 distinct CPU/MPI tests passed across the available environments.
The 13 individual skips are seven optional medical/mesh-format dependency
checks and six existing unimplemented test placeholders. The Python 3.12
collection also skips optional pythonocc-core and PyVista modules, plus the
PyCUDA-import test exercised separately above. No tests were weakened or
disabled to obtain these results. Some previously unmarked expensive checks
were explicitly classified as slow and still run locally.

The broad device runs cover source families, virtual guides, network ports,
studies/reuse, PML, thin-wire coefficients, SAR, NTFF and symmetry. After the
last shared snapshot bounds correction, all affected native-collocation device
tests were rerun, together with the new write-counter tests. This accounts for
125 CUDA and 107 OpenCL identities without counting overlapping reruns twice.

### Independent numerical comparisons

Fresh manual runs completed eight CPU drivers, ten CUDA drivers and two
two-rank MPI drivers. Sixteen main drivers and both MPI drivers passed their
declared acceptance checks; multilayer and core-shell runs are explicitly
report-only. In total, 178 threshold/validity checks passed: 161 comparisons
against physical reference solutions, six reference-series residual checks,
two normalization identities, one stability ratio and eight validity checks.
These are **not** 178 independent experiments or extra pytest cases.

Representative results:

| Comparison | Measured result | Existing acceptance limit |
|---|---:|---:|
| PEC sphere RCS, Mie, 0.5-mm grid | RMS 0.441689 dB; max absolute 0.953946 dB | 0.75 / 1.25 dB |
| Dielectric sphere RCS, Mie, same grid | RMS 0.270687 dB; max absolute 0.724350 dB | 0.75 / 1.25 dB |
| Six dielectric/dispersive half-spaces, 0.5-mm grid | Worst complex reflection relative L2 0.244243% | 1% |
| Hertzian direct/KSIR near fields, 1-mm grid | Significant-window relative L2 0.0335327% / 0.0166692% | 0.1% each |
| SAR sphere, 0.75-mm grid | Absorbed-power relative error 6.36748% | 8% |
| TM/TE tissue cylinders, 0.4-mm grid | Worst integrated absorbed-power relative error 0.948949% | 2% |
| R/C/L/RC/RLC TEM sheets | Worst reflection magnitude RMSE 0.00610431 | 0.02 |

See the [full analytical table](release_analytical_2026-09-06.md) and
[commands/threshold ledger](release_analytical_2026-09-06.json) for every case,
mesh, error definition and calibration assumption. In particular:

- Hertzian full-history direct error is 0.601176%, larger than its gated
  significant-window error. Both are retained in the evidence.
- Cylinder local accuracy gates exclude a two-cell interface band. Including
  that band gives up to 12.3335% L2 and 75.4254% maximum pointwise error in the
  reported cases; passing the interior gates is not uniform boundary accuracy.
- The coarse core-shell run has 10.1187% / 15.1069% averaged/staircased RCS
  L2 error and no accuracy gate. It is not certified merely because it exits
  successfully. Multilayer runs are similarly report-only.
- The sphere's 1-g/10-g peaks are reported, not independently analytically
  validated mass averages. Grounded-pattern checks fit a common complex scale
  and validate shape/directivity, not absolute source calibration.

### Build and documentation checks

Both Python 3.12 and 3.13 Cython builds succeeded. The normal Sphinx HTML
build with `-W --keep-going -E` succeeded. An additional, stricter `-n` link
audit reported 679 unresolved references, largely generated ReFrame/NumPy
API targets; it did **not** pass and remains a documentation-cleanup task.

A clean exported source tree produced an installable Linux CPython 3.13 wheel.
Outside the source tree, the installed package imported all 25 compiled
extensions, found packaged examples/MATLAB/STL resources, and ran a tiny CPU
model. The final artifact is 11,951,540 bytes, SHA-256
`2b62e79cad919c803facfb3a40dda7eba8cf921fecafe62bb01f41324c342705`.
It contains the final snapshot correction and no excluded `.pxd`, `.pyx` or
`.c` payload. This is a local smoke test, not a claim of a portable manylinux,
Windows or macOS release artifact.

The first wheel built in the long-lived developer tree failed the payload
check because stale `build/lib` files survived from earlier work. A clean
export removed those obsolete entries without any packaging-code change.
Use a clean checkout/build tree for release wheels. Likewise, an earlier CPU
run during editing retained obsolete in-memory MPI snapshot fixtures; the
fresh full rerun recorded above is the accepted result, not that failed run.

### Targeted counterexamples

- Shared-waveform construction order was reversed across voltage, electric
  dipole, magnetic dipole and transmission-line sources. Separate waveform
  IDs, shared IDs and serial/MPI results produced byte-identical six-component
  receiver traces in the tested cases.
- Hard-source windows covered incident fields before the source starts,
  zero-valued samples while active, and propagation after it stops. The CUDA
  and OpenCL selections each contained 66 CPU/device comparisons, including
  both precisions and geometry-fixed reuse. Across those comparisons the
  largest peak-normalized voltage error was `3.087e-5` in single precision and
  `4.018e-14` in double precision; same-backend reuse/order controls were exact.
- Magnetic-volume overwrite controls used all seven supported volume
  primitives. Reference and fully overwritten CPU models gave byte-identical
  six-component traces. The largest CUDA/CPU relative L2 differences were
  `4.510e-13` for E and `1.292e-12` for H. Actual MPI x/y/z decompositions and
  an off-origin ratio-three subgrid also reproduced the expected geometry,
  rigidity, tags and material catalogue.
- Wide-ID imports were built with a 40,001-material destination catalogue on
  the main grid, MPI and subgrid. Compact file index 1 mapped to global ID
  40,000 without signed-16-bit overflow. Transparent component tests covered
  all 12 electric and six magnetic rigidity claims independently.
- The final reduced-output MPI matrix contains 56 passing cases, exercising
  TMx/TMy/TMz and TEx/TEy/TEz, both precisions and live-axis decompositions,
  plus eight-rank 3-D corners. Snapshot arrays are byte-identical to serial
  Cython output; SAR/radiometry numerical comparisons use declared tolerances.
  Independent checks poisoned unowned halos with NaNs, removed selected halo
  planes and forced seven-cell exchange batches: all 18 global snapshots
  remained finite, byte-exact to serial, and consistent with the affine oracle.
- The final device snapshot test uses an atomic write counter around the
  common six-component store block, rotated smaller windows in a 315-cell
  allocation, padded launches, both precisions and a forced single-work-item
  OpenCL loop. Corrected output cells are written exactly once and padding
  remains untouched. Restoring legacy modulo wrapping in an isolated process
  makes the regression fail: six cells are written twice. Equal numerical
  values in earlier comparisons did not establish race-free execution.

### Reproduction and environment

The standard and slow CPU selections were run separately, with every collected
test identity reconciled against their JUnit records and the separately run
new MPI output matrix. Repeated focused runs are not additional independent
test counts. The one CPU-only PyCUDA-import test unavailable in the Python
3.12 environment was also run under Python 3.13.

```console
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest tests -m "not gpu and not slow"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest tests -m "not gpu and slow"
python -m pytest tests/outputs/test_mpi_reduced_outputs.py
python -m sphinx -b html -W --keep-going -E docs/source /tmp/gprmax-docs
```

The full CPU environment used Python 3.12.2, Open MPI 5.0.8, parallel HDF5/h5py
and the working parallel FFTW stack. Complementary source/output runs used
Python 3.13.5 and MPICH 4.3.2. Hardware testing used NVIDIA TITAN RTX devices
through CUDA and OpenCL, and the installed Intel CPU OpenCL implementation.
Hardware selections were checked against collected node IDs, not inferred
from a passing `-k` expression. Both single and double precision were tested.

Raw logs, JUnit XML, input/output files, numerical summaries and diagnostic
scripts are retained locally at
`/tmp/gprmax-release-closeout-20260906/`. This report and maintained regression
tests are the repository evidence; transient generated models and large
historical research directories are not part of the PR.

## CI follow-up: Open MPI 5 slot allocation

PR828's standard CPU job exposed a launcher condition missed by the initial
many-core local runs: the task-farm regression needs three ranks, while the
runner supplied fewer slots. Open MPI 5/PRRTE ignored the older
`OMPI_MCA_rmaps_base_oversubscribe` setting, so gprMax never started. This was
not a failure of task-farm error handling.

Pytest now defaults both that legacy variable and
`PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe`, preserving explicit
user settings. This applies to the whole test suite, including eight-rank
cases, without changing ordinary simulation launch policy or reducing tests'
requested rank counts. Four new unit cases check defaulting, preservation and
idempotence; they are additional to the earlier inventory above.

The original insufficient-slots failure was reproduced with three processes
and an explicit `localhost slots=2` allocation. Verification after correction:

- Open MPI 5.0.8, forced two slots: all 14 selected environment and release
  regressions passed, including the three-rank task farm. The expected bad job
  was reported and the good job's output retained.
- MPICH 4.3.2: the same 14 selected tests passed.
- Open MPI, eight ranks on two slots: both snapshot corner tests passed,
  retaining the single/double precision and serial-equivalence assertions.

Logs and JUnit records are in `/tmp/gprmax-pr828-slots/`. These focused results
are not a rerun of the entire earlier verification inventory. Optional toolbox
dependency skips and pre-existing unimplemented test placeholders are unrelated
to this launcher correction.

## Scope and compatibility

- No dependency or Python-version requirement is changed. Python 3.12 remains
  the supported configuration for distributed FFTW testing.
- No dual-magnetic-cell redesign is included. These geometry corrections
  repair ownership and overwrite behavior in the existing generalized Yee
  model.
- Device hard-source metadata is internal; no new public command is required.
- Geometry files retain compact file-local material indices. Wide destination
  IDs do not increase the number of distinct materials the writer can encode
  in its signed 16-bit catalogue.
- Complete imported component meshes remain authoritative. The reader cannot
  reconstruct the construction history of an old file containing stale H
  rigidity from R5. Regenerate such cached geometry from its original model
  with the corrected builder; these fixes do not silently reinterpret it.
- Snapshot coordinates, time levels and file precision are explicit. Invalid
  partial final cells are rejected instead of silently moving output bounds
  or changing the requested live-axis spacing.
- No claim of new macOS Metal hardware validation is made from this Linux
  server. Shared-kernel generation and mocked contracts are tested, but a
  physical Mac run remains a release check.

## Performance evidence already established

The earlier controlled PML diagnostic used a 100-cubed domain, six 10-cell
slabs, first-order HORIPML, single precision and five warmed update steps.
Bounded dispatch produced byte-identical fields and history arrays to legacy
dispatch. Kernel time changed from 18.9057 to 0.387627 seconds on Intel OpenCL
and from 8.68422 to 1.23018 milliseconds on NVIDIA OpenCL. These are measured
diagnostic kernel timings, **not** general simulation speedup guarantees.

The snapshot allocation probe used successively rotated `(20,2,2)` output
shapes. Its third six-component allocation decreased from 192,000 to 1,920
bytes. Four retained snapshots now require six full-history downloads instead
of twenty-four. These are allocation/transfer measurements, not wall-clock
claims.

The final two-rank MPI snapshot probe used a 100 x 90 x 80 affine field,
spacing `(1,2,3) mm`, cropped native ROI `[2,3,4]:[98,87,76]` and double
precision. After one warm-up, the slower rank's median of three calls was:

| Sampling | Legacy time | Corrected time | Legacy/corrected maximum coordinate error |
|---|---:|---:|---:|
| Native, 580,608 output cells | 67.881 ms | 40.916 ms | `4.44e-16` / `4.44e-16` |
| Stride three, 21,504 output cells | 1.124 ms | 1.437 ms | `0.013` / `4.44e-16` |

The native case's maximum per-rank RSS increment fell from 29,896 to 17,128
KiB. The coarse case established no new process high-water mark; that is not
a claim of zero temporary allocation. Legacy coarse sampling was inaccurate,
so its timing is not an accuracy-equivalent alternative. Other correctness
campaigns ran on separate CPU affinities/devices during this local diagnostic.
