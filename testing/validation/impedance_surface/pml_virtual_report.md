# General SIBC through PML and virtual waveguides

12 September 2026. CPU implementation, based on `977780ce` plus the local
impedance-surface changes. General passive constant and dispersive Foster
impedances now support uniform continuation through longitudinal PML and
virtual-waveguide apertures. The exact `resistance=inf` PMC is one supported
limit. The automatic time-step factor remains at most 0.99.

## Discrete coupling

For retained electric dual area $A_e$, the longitudinal clipped H derivative
has the same retained fraction as the electric mass. Under uniform extrusion,
the ordinary native PML derivative correction $Q_e$ therefore contributes
$A_e Q_e$ to the sparse electric circulation. With local implicit denominator
$d_e$, its solved increment is

$$
\Delta E_e=\frac{A_e Q_e}{d_e},\qquad
\Delta y_{pm}=\frac{q_{pm}}{2Z_{0p}}\Delta E_e.
$$

The second equation updates the Foster state consistently with midpoint-time
surface current. Changing the old electric field before the local solve would
give the wrong finite-impedance damping and history. The implementation
captures the native PML correction, restores the true old E, and applies the
equivalent solved increment and history correction. Independent dense
two-port solves verify both constant forcing and nonzero Foster histories.

The auxiliary virtual guide copies full dual-cell masses, clipped H weights,
surface models, and independent history storage. At the aperture, main-side
H supplies the missing part of the circulation. Source-plane electric
corrections use the same implicit denominator and history increment.

## Physical evidence

- [Native finite/dispersive PML report](results/sibc_pml/README.md): resistive
  and four-pole fitted conductor guides compared with longer causal reference
  guides, plus source-free runs with weak high-frequency random seeds. The
  four default-profile runs cover both surface models and both CPU precisions
  for 20,000 steps. All remained bounded; their final squared field norms were
  about $2.56\times10^{-5}$ of the initial value. These field norms are decay
  diagnostics, not a proof of passivity of the complete PML state.
  Two further 20,000-step Foster runs with the selected second-order
  HORIPML/MRIPML profiles also remained bounded. The default first-order
  pulse discrepancy was at most $4.43\times10^{-7}$ of the reference peak;
  the selected second-order HORIPML profile gave $4.72\times10^{-4}$.
- [Virtual-guide report](results/virtual_waveguide.md): 36 comparisons across
  all axes and directions, HORIPML and MRIPML, and PMC/1000-ohm/copper models.
  All six fields at center and wall receivers were bitwise identical to the
  corresponding monolithic guide. Six active runs gave maximum $|S_{11}|$
  of $1.64\times10^{-3}$ at 22 GHz.
- [PMC/PML image report](../sibc_based_pmc/pml_mirror.md): 24 independent
  mirrored-domain comparisons, including PML histories, both precisions,
  both formulations, and one/two CFS terms. Maximum retained-field error was
  $4.38\times10^{-15}$ in double and $3.10\times10^{-6}$ in single precision.

## Profile dependence and supported scope

Uniform extrusion alone does not make an arbitrary PML profile stable. A
custom HORIPML made by duplicating an unshifted first-order CFS term produced
late growth in **both an ordinary PEC guide and an SIBC guide**. The
[profile audit](results/sibc_pml/profile_audit.json) preserves that failure
and its measured duration. This is a native PML configuration failure, not
evidence of instability introduced by the SIBC coupling. Default first-order
PML results are recorded separately from custom second-order profiles.

The [follow-up diagnosis and tested repair](results/pml_profile_investigation/README.md)
identify a negative real stretch from the unshifted product, confirm a
continuous growing mode and persistence at smaller time steps, and validate
a second shift graded as alpha2 = 1.1 sigma1.

The [follow-up diagnosis and tested repair](results/pml_profile_investigation/README.md)
identify a negative real stretch from the unshifted product, confirm a
continuous growing mode and persistence at smaller time steps, and validate
a second shift graded as alpha2 = 1.1 sigma1.

The supported intersection requires:

- A passive surface model and wall/host extrusion along every intersecting
  PML absorption direction; no end caps or steps in the absorber.
- A homogeneous, isotropic, lossless, nondispersive retained host at each
  intersecting edge. The **surface impedance may be dispersive**.
- Transverse PML coverage extending into the opaque wall volume, and virtual
  modal windows enclosing the walls with opaque-voxel padding.
- A 3D CPU main grid. Existing MPI, accelerator, and subgrid restrictions on
  impedance surfaces still apply.

The detailed equations, input syntax, geometry rules, and limitations are in
[`docs/source/impedance_surfaces.rst`](../../../docs/source/impedance_surfaces.rst).

## Regression checks and environment

The complete impedance-surface suite passed: **530 tests**. The new virtual
guide tests passed (17 physical/modal tests plus the subsequently added
reset/padding test). Another targeted run passed 61 related CPU checks for
ordinary virtual guides, FDFD impedance operators, internal PMLs, and local
surface solves; this overlaps some tests in the 530-test suite. Two new
finite/dispersive PML physical regressions also passed.

Two unrelated accelerator integration checks could not run successfully in
this environment: CUDA failed during `nvcc` preprocessing, and OpenCL could
not create its cache outside the writable workspace. No accelerator support
is claimed for SIBC. A documentation build was attempted but Sphinx is not
installed in the available solver environment.

Reproduce the physical comparisons from the repository root:

```text
python -m testing.validation.impedance_surface.validate_sibc_pml
python -m testing.validation.impedance_surface.validate_sibc_pml --profile-audit-only
python -m testing.validation.impedance_surface.virtual_waveguide
python -m testing.validation.sibc_based_pmc.pml_mirror
```

Use one OpenMP/BLAS thread for the saved comparisons. Raw solver outputs stay
in ignored cache directories; JSON, CSV, and PNG results are retained.
