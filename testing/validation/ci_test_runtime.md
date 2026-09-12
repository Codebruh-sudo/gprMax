# CPU CI runtime: modal example builds and horn normalization

The CPU job for PR #833 timed out on 12 September 2026 at 81% completion:
https://github.com/gprMax/gprMax/actions/runs/34694359428/job/103555163189
The job has a 30-minute timeout including installation. Its selection is
`not unit and not gpu and not slow`; the copper-wall and automatic-cutoff
validation tests are already excluded and cannot explain this timeout.

The GitHub log provides timestamps for groups of tests, not individual test
durations. It shows about 240 seconds in `test_eigenmode_source_2d.py` and a
68-second batch containing the horn normalization test. Local JUnit timings
identify the following two expensive test functions in the selected feature
tests. These measurements are not a complete per-test ranking of the whole
GitHub job.

| Test | Before (seconds) | After (seconds) |
| --- | ---: | ---: |
| `test_2d_regression_example_builds`, all 12 parameters | 204.57 | 3.71 |
| `test_eigenmode_port_normalises_gain_and_realized_gain` | 129.76 | 86.40 |

Before timings are from the saved Windows feature-regression run. After
timings use the same Python environment with `OMP_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, and `MPLBACKEND=Agg`. Measurements are approximate
wall times rather than CI performance guarantees. A separate cProfile run
of the original horn test took 146 seconds; profiling overhead is not used
in the table. The changes save about four minutes locally without removing
tests from CI or increasing its timeout.

## What changed

The 2D test previously rendered several large multi-anchor PNGs for each
geometry, although it only checked their existence. The test now disables
plots in its temporary input copy, builds the original geometry and every
anchor, and checks snapshot counts, both prepared ports, mode validity, and
finite, nonzero E/H profiles. The examples themselves are unchanged. Existing
plot-control tests and invariant-axis build tests still exercise real plotting.

The horn test keeps the full geometry, mesh, 4 ns simulation, broadband modal
anchors, and all power-normalization assertions. Its temporary input copy
requests NTFF output at 8, 10, and 12 GHz and at 30-degree angular intervals,
instead of nine frequencies and 5-degree intervals. The independent full-sphere
power quadrature is unchanged. At the three retained frequencies, the original
and optimized runs produce exactly equal incident power, accepted power,
radiation efficiency, and total efficiency. The full-resolution example remains
unchanged for antenna studies.

The workflow now prints the 20 slowest test durations on completion, making
future bottlenecks easier to identify.

## Validation

All 91 tests in the two affected modules passed, including their PNG output
checks and the horn gain/realized-gain identities. No assertion thresholds
were relaxed.

```powershell
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MPLBACKEND = 'Agg'
python -m pytest tests/cmds_multiuse/test_eigenmode_source_2d.py tests/ntff/test_hash_command.py --durations=20
```
