# Eigenmode-port and impedance-surface accelerator regression run

The accompanying `port_surface_accelerator_regressions.json` records the test
inventory, environment, final results, and device field-error measurements for
the Windows validation on 12 September 2026. It supplements the existing physics
reports; their assumptions and acceptance criteria still apply.

Final result: **2,798 passed, 23 skipped, no unresolved failures**, across 88
test modules. Skips comprise 9 Apple Metal cases, 12 MPI/parallel-HDF5 cases,
and 2 Windows symbolic-link privilege cases.

All 72 new degenerate-mode device cases passed. The maximum normalized field
error across physical and virtual guides was `2.643e-15` in double precision
and `1.456e-6` in single precision. The full copper-wall validation also passed:
its FDTD attenuation relative L2 error was 0.75%, against a 2% acceptance limit.

## Scope

The inventory includes every test module mentioning eigenmode ports, surface
impedance/SIBC, or virtual waveguides, plus the complete FDFD, impedance-surface,
and PML test directories. CUDA solver regression modules and the related hash
include tests are also included. No GPU or slow markers were excluded.

CUDA and OpenCL run on the NVIDIA RTX 4070 Laptop GPU, in both precisions where
the tests parameterize them. The available Intel OpenCL device was not selected.
SIBC models retain their documented CPU-only restriction; this run does not add
GPU support for SIBC. The tests cover the supported port/device combinations
and the rejection of unsupported combinations.

The additional circular TE11 device regression tests each physical polarization
and a broadband quadrature combination, all three propagation axes, both
directions, and both precisions. Each case compares a CPU physical guide with
device physical and virtual guides after independently rotating the raw modal
basis at every solve. The normalized receiver-field error weights H by the
free-space impedance before comparing it with E. Its acceptance limits are
`2e-8` in double precision and `2e-4` in single precision. Output metadata must
also retain the aligned pair and mixed-mode residuals below `1e-9`.

## Reproduction

From the repository root, with the gprMax Python environment active:

```powershell
. ./packaging/activate_cuda.ps1
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MPLBACKEND = 'Agg'
$report = Get-Content testing/validation/port_surface_accelerator_regressions.json -Raw | ConvertFrom-Json
$testFiles = @($report.test_files)
python -m pytest @testFiles -q --junitxml=port_surface_results.xml
```

The compiler setup affects the current PowerShell session only. It locates the
MSVC x64 environment using Visual Studio's `vswhere` and `vcvars64.bat`. A fresh
NVCC kernel compilation and execution were also checked outside the restricted
test runner, with PyCUDA's disk cache disabled.

For a restricted runner that cannot write the normal user caches, create local
cache directories and set these variables before starting Python:

```powershell
New-Item -ItemType Directory -Force .pytest_cache/cuda, .pytest_cache/opencl, .pytest_cache/platform | Out-Null
$env:PYCUDA_CACHE_DIR = (Resolve-Path .pytest_cache/cuda).Path
$env:PYOPENCL_CACHE_DIR = (Resolve-Path .pytest_cache/opencl).Path
$env:WIN_PD_OVERRIDE_LOCAL_APPDATA = (Resolve-Path .pytest_cache/platform).Path
```

The last variable is a platformdirs override, used by the installed PyOpenCL/
pytools cache stack. Check that your platformdirs version supports this override.
It is unnecessary when normal user cache directories are writable.

## Qualifications

The first pass exposed restricted cache access in OpenCL and missing Windows
symbolic-link privilege. The cache was redirected and all affected OpenCL tests
rerun. The two include-cycle tests now skip only the Windows missing-privilege
error when creating their symbolic-link fixture; ordinary cycles remain tested,
and other filesystem errors still fail. This does not change include handling
in the application.

Final counts in the JSON merge the completed runs by test identity, replacing
earlier results with their corrected reruns, rather than counting repeated tests
twice. Hardware/dependency skips are listed separately. Apple Metal cannot be
executed on this Windows host, and the installed HDF5 build lacks parallel I/O.
The documentation build retained the 58 previously recorded warnings and added
none in the changed accelerator instructions.
