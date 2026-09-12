# Regression checks — 12 September 2026

Windows CPU validation used the existing `gprMax2` Python environment with
`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `MPLBACKEND=Agg`.
Temporary pytest files were placed in an unused directory inside `.pytest_cache`.

The broad run completed with **1,543 passed, 12 skipped, 16 deselected, and two
accelerator environment failures** in 715 seconds. All selected CPU checks passed.
The exact selection was:

```text
python -m pytest tests/impedance_surfaces tests/pml tests/test_virtual_waveguide_impedance.py tests/test_virtual_waveguide_coupling.py tests/test_virtual_waveguide_integration.py tests/cmds_multiuse/test_virtual_waveguide_commands.py tests/cmds_multiuse/test_eigenmode_source_2d.py tests/fdfd_eigenmode_solver --ignore=tests/pml/test_opencl_pml_geometry_fixed.py --ignore=tests/pml/test_opencl_pml_range_solve.py --ignore=tests/fdfd_eigenmode_solver/test_virtual_waveguide_device.py -k "not device" -q --basetemp=.pytest_cache/sibc-regression
```

- The 12 skips require MPI and parallel HDF5, unavailable in this environment.
- `test_cuda_multimode_virtual_waveguide_matches_cpu` failed during CUDA
  preprocessing. A minimal compiler probe confirmed `nvcc fatal: Cannot find
  compiler 'cl.exe' in PATH`; the CUDA toolkit itself is installed.
- `test_opencl_multimode_virtual_waveguide_matches_cpu` failed because the sandbox
  denies creation of the pytools cache under the user's `AppData/Local` directory.
  It did not reach the field comparison.

These two accelerator tests were included because their names do not contain
`device`. To repeat only the selected CPU scope, additionally exclude them with
`-k "not device and not cuda_multimode and not opencl_multimode"`. Accelerator
results are not claimed by this validation; reduced SIBC/virtual coupling is CPU only.

After the final source-storage optimization and additional PEC-only 2D tests,
the focused selection below passed **211 tests** in 145 seconds. These overlap
the broad run and are not an additional count of unique regressions.

```text
python -m pytest tests/impedance_surfaces/test_2d_sibc.py tests/impedance_surfaces/test_legacy_pmc_warning.py tests/pml/test_profile_guard.py -q --basetemp=.pytest_cache/sibc-focused
```

A subsequent seven-test selection also passed after strengthening assertions for
independent Foster state storage, resetting all four PML histories and both SIBC
history arrays, and suppressing the legacy warning at PMC symmetry planes:

```text
python -m pytest tests/impedance_surfaces/test_2d_sibc.py tests/impedance_surfaces/test_legacy_pmc_warning.py -k "virtual_continuation or symmetry_planes" -q --basetemp=.pytest_cache/sibc-reset
```

Both documented examples ran for their default 1,200 steps:

```text
python examples/features/impedance_surface/virtual_waveguide_2d.py --mode TM --wall pmc
python examples/features/impedance_surface/virtual_waveguide_2d.py --mode TE --wall foster
```

The separate analytical instability reproduction also passed:

```text
python -m testing.validation.impedance_surface.investigate_pml_profile --analytic-only
```

It retains the continuous/discrete unstable-symbol calculation independently of
normal model construction. No public bypass of the new PML error is provided.

The [physics report](README.md) records the 288 physical/virtual comparisons,
36 matched-timestep 3D comparisons, and twelve 20,000-step runs. The
[PMC report](../../../sibc_based_pmc/2d_validation.md) adds exact discrete-mode checks.
`git diff --check` passed. A full Sphinx documentation build was not run because
Sphinx is not installed in this environment; the executable examples were tested.

## Commit preparation

After applying the repository's Black/isort settings, **270 tests passed** in
212 seconds with this selection:

```text
python -m pytest tests/impedance_surfaces/test_2d_sibc.py tests/impedance_surfaces/test_automatic_timestep.py tests/impedance_surfaces/test_pmc_api.py tests/impedance_surfaces/test_legacy_pmc_warning.py tests/pml/test_profile_guard.py tests/test_virtual_waveguide_impedance.py tests/fdfd_eigenmode_solver/test_fdfd_1d_mode_solver.py -q
```

All applicable staged pre-commit hooks passed: whitespace, end-of-file,
added-file size, Black, and isort. YAML/TOML checks had no staged inputs.
The repository-wide hook audit was run in a temporary review copy: it found
pre-existing formatting issues outside this change and was stopped during the
remaining repository-wide Black pass. Only this commit's files received fixes;
a clean repository-wide formatting result is not claimed.
