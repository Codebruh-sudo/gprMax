============================
Surface-impedance validation
============================

This package contains independent analytical comparisons for the
surface-impedance boundary implementation. Each driver writes numerical CSV
data, a PNG comparison, and a machine-readable ``summary.json``. Solver HDF5
files are retained only below an ignored ``_cache`` directory for optional
``--reuse`` analysis.

The general SIBC/PML and virtual-waveguide extension is summarized in
`pml_virtual_report.md <pml_virtual_report.md>`_, including the coupled
update, supported scope, physical comparisons, and a native second-order
PML profile which failed the stability audit.

The validations are:

* ``validate_2d.py``: every TE/TM invariant axis and propagation direction,
  independent 3D extrusion, active/passive virtual guides, native infinite
  resistance, and long PML runs. See ``results/2d_sibc/README.md``;
* ``validate_conductor_sphere.py``: conducting-sphere backscatter and complex
  angular scattering against impedance-boundary and bulk-conductor Mie theory;
* ``validate_reflection_phase.py``: planar reflection magnitude and phase in
  air and a Debye exterior, directly exercising dispersive SIBC contact;
* ``validate_copper_wall_waveguide.py``: copper-preset TE10 propagation;
* ``validate_sibc_pml.py``: finite-resistance and fitted Foster walls
  extruded through longitudinal PML, compared with longer causal reference
  guides, followed by long source-free runs. Results are under
  ``results/sibc_pml``;
* ``virtual_waveguide.py``: physical/virtual guide comparisons for constant
  resistance, fitted copper, and exact PMC, plus active modal injection.
  See ``results/virtual_waveguide.md`` and its JSON results;
* ``investigate_pml_profile.py``: continuous and native discrete diagnosis
  of the duplicated unshifted HORIPML instability, smaller-time-step
  controls, and a tested frequency-shift repair. See
  ``results/pml_profile_investigation/README.md``;
* ``stability.py``: production-kernel curl adjointness, complete E/H/ADE
  spectra, and long-time source-free growth, including a CFL-endpoint
  high-resistance cavity reproducer. See ``stability_report.md``;
* ``default_cfl.py``: default timestep policy and CPU-precision runs
  through the full solver, with 80-digit CFL comparisons and an independent
  stored-coefficient explanation of the cavity growth. See
  ``default_cfl_report.md``.

The related `SIBC-based PMC validation <../sibc_based_pmc/README.md>`_
derives the exact zero-admittance limit and checks reflection at the
main-voxel face, 3D mirror equivalence, cavity modes, and long-time behavior.
It also measures the half-cell displacement of the existing ``pmc`` volume.
Its ``pml_mirror.py`` driver compares the public exact-PMC implementation
against an independent mirrored domain through longitudinal PML, including
all retained fields and PML histories.
The reduced-mode PMC results are in ``../sibc_based_pmc/2d_validation.md``.

The PML/virtual-guide extension accepts general passive surface dispersion.
Walls and their retained hosts must be invariant along the PML absorption
direction. Each intersecting edge requires a homogeneous, isotropic,
lossless, nondispersive retained host. These checks distinguish surface
dispersion from unsupported bulk dispersion inside the PML intersection.
Virtual modal windows must enclose the walls with opaque-voxel padding.

SIBC declarations automatically cap the existing time-step stability factor
at 0.99 and preserve smaller user factors. The reflection driver sets the
same factor on its no-wall reference so both traces use an identical time
grid. Solver-cache keys include the factor to avoid reusing pre-cap results.

Run the current stability policy from the repository root::

    python -m testing.validation.impedance_surface.default_cfl --steps 200000
    python -m testing.validation.impedance_surface.stability --steps 20000

These write ``results/default_cfl_protected.json`` and
``results/stability_protected.json`` plus matching PNG files. Add
``--historical`` to disable the cap locally within these diagnostics and
reproduce the saved pre-protection ``default_cfl.json`` and ``stability.json``
results. The intentionally above-CFL control also bypasses the cap locally;
this is not a supported simulation setting.

Conducting-sphere scattering
----------------------------

Run from the repository root::

    python -m testing.validation.impedance_surface.validate_conductor_sphere --threads 4

This case illuminates a 16 mm radius sphere in air with a plane-wave pulse
and extracts the scattered far field using a near-to-far-field transform.
It compares 21 frequencies from 2 to 7 GHz on 1.5 and 0.75 mm cubic meshes,
using CPU double precision. The sphere is an opaque surface-impedance
volume; its skin depth is not resolved by interior cells.

Here :math:`\sigma` is the sphere's bulk electrical conductivity, in S/m.
The defaults are :math:`10^3` S/m and :math:`5.8\times10^7` S/m, the latter
representing copper-like conductivity. Conductivity determines the complex
good-conductor surface impedance through

.. math::

   Z_s(f)=(1+j)\sqrt{\frac{\pi f\mu_0}{\sigma}}.

Thus ``sigma`` in the filenames and conductivity in the plot titles specify
the material used to obtain :math:`Z_s`; they are not a resistance in Ohms
or the radar cross section (RCS). Larger conductivity gives smaller
surface impedance and approaches the PEC limit.

Reading the sphere figure
~~~~~~~~~~~~~~~~~~~~~~~~~

The top row shows backscatter RCS versus frequency. The bottom row shows
the angular RCS pattern at 4.5 GHz, with 0 degrees forward and 180 degrees
backward. Each column uses one conductivity. RCS is plotted in dB relative
to one square metre.

* **FDTD 1.5 mm / FDTD 0.75 mm** are the simulated results for the two
  staircased representations of the sphere.
* **Impedance Mie** is the continuum analytical solution for a smooth sphere
  satisfying the impedance boundary condition with the physical
  :math:`Z_s(f)` above.
* **Bulk conductor Mie** is the full continuum solution for a homogeneous
  conducting sphere with :math:`\epsilon_r=1-j\sigma/(\omega\epsilon_0)`.
  It includes the field inside the conductor analytically. It does not use
  the impedance boundary approximation.

The two Mie curves almost overlap because the surface-impedance
approximation is accurate for these good conductors. Their agreement tests
that physical approximation; agreement between FDTD and impedance Mie tests
the numerical implementation, including the staircased geometry.

.. figure:: results/conductor_sphere/conductor_sphere.png
   :alt: Conducting-sphere backscatter and angular RCS compared with two Mie references.

   Two mesh spacings and two conductivities, compared with smooth-sphere theory.

Sphere results and acceptance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The retained 2026-09-07 results are:

.. list-table:: Errors against impedance-boundary Mie theory
   :header-rows: 1

   * - Conductivity (S/m)
     - Mesh (mm)
     - Backscatter RMS (dB)
     - Maximum backscatter error (dB)
     - Complex-pattern relative L2
   * - 1,000
     - 1.5
     - 1.30447
     - 2.51283
     - 12.59%
   * - 1,000
     - 0.75
     - 0.85464
     - 1.54738
     - 7.20%
   * - 58,000,000
     - 1.5
     - 1.25862
     - 2.43024
     - 11.85%
   * - 58,000,000
     - 0.75
     - 0.83204
     - 1.54577
     - 6.68%

Backscatter error is :math:`10\log_{10}(\mathrm{RCS}_{FDTD}/\mathrm{RCS}_{Mie})`;
the RMS and maximum are taken over frequency. The complex-pattern metric
compares incident-normalized complex far-field amplitudes over all sampled
frequencies and angles, so it includes phase as well as amplitude error.

Refinement reduces both errors. The staircased shape shifts the backscatter
minimum and explains why errors remain visible near that minimum. Copper
is nearly PEC here, so sphere RCS alone is not a sensitive test of its tiny
finite-conductivity correction; the planar phase case below provides a
more direct check of finite surface impedance.

Both fine meshes pass: backscatter RMS/max errors must be below 1/2 dB,
complex-pattern relative L2 below 15%, impedance-fit error below 0.2%, and
impedance/bulk-Mie complex disagreement below 2%. Refinement must reduce
backscatter RMS and complex-pattern errors. The coarse meshes do not pass
the fine-mesh gates and are retained as convergence diagnostics. The actual
impedance-fit error on the analysis frequencies is 0.11627%; impedance and
bulk Mie differ by relative L2 about :math:`9.24\times10^{-6}` at 1,000 S/m.

Results are in ``results/conductor_sphere/``: ``conductor_sphere.png``,
``summary.json``, and one complex-field/RCS CSV per conductivity and mesh.
Use ``--reuse`` to reanalyse compatible local caches, ``--output-dir`` to
change the destination, ``--mesh-mm`` to select meshes, or ``--conductivity``
to select material conductivities. The driver returns nonzero if acceptance
fails. Raw HDF5 and normalized NTFF caches remain in the ignored ``_cache``
subdirectory.

Plane-wave reflection magnitude and phase
-----------------------------------------

Run from the repository root::

    python -m testing.validation.impedance_surface.validate_reflection_phase --threads 4

The reflected-wave comparison covers 1--8 GHz on 1 and 0.5 mm meshes.
A uniform current sheet launches a TEM wave between transverse PEC/PMC
symmetry faces. A matching reference run supplies the incident wave. The
reflected/incident ratio is de-embedded 30 mm to the physical wall using
the independently derived discrete host wavenumber. Outer boundaries are
far enough away that their earliest return is after the 4 ns record.
The Debye host has relative permittivity
``2.5 + 2/(1 + j*omega*80 ps)`` and directly touches the impedance wall.

The wall conductivities are again :math:`\sigma=10^3` and
:math:`5.8\times10^7` S/m. These specify the wall's :math:`Z_s(f)` through
the good-conductor formula in the sphere section. They are not the
conductivity of the air or Debye exterior. The exterior instead determines
the incident wave impedance :math:`\eta(\omega)`.

Continuum, discrete, and sigma in the legend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The continuum analytical electric reflection coefficient at the wall is

.. math::

   \Gamma_{\mathrm{continuum}}(\omega)
   =\frac{Z_s(\omega)-\eta(\omega)}{Z_s(\omega)+\eta(\omega)},
   \qquad
   \eta(\omega)=\sqrt{\frac{\mu_0}{\epsilon_0\epsilon_r(\omega)}}.

It uses the physical, continuous-frequency material laws. The separate
discrete prediction describes the finite-space-step, finite-time-step FDTD
scheme. It includes Yee staggering, the retained half-cell electric mass,
the Debye polarization recurrence, and the fitted impedance realization.
The latter is evaluated at the bilinear-warped frequency
:math:`f_b=\tan(\pi f\Delta t)/(\pi\Delta t)`. The discrete prediction is
computed analytically; it is not a second simulation.

.. list-table:: Meaning of the reflection-figure legend
   :header-rows: 1

   * - Legend entry
     - Meaning
   * - FDTD sigma=...
     - Measured reflection for a wall with that conductivity, after removing
       receiver-to-wall propagation phase.
   * - Analytical sigma=...
     - Continuum reflection coefficient for the same conductivity, shown
       alongside FDTD in the phase panel.
   * - Continuum sigma=...
     - Bottom-panel phase error of FDTD relative to the continuum coefficient.
   * - Discrete sigma=...
     - Bottom-panel phase error of FDTD relative to the discrete coefficient.

For example, ``Discrete sigma=1000 S/m`` means the error relative to the
discrete prediction for a 1,000 S/m wall. There is no separate "discrete
conductivity": continuum and discrete comparisons use the same physical
conductivity, but different reference equations.

Reading the reflection figure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The left column is air and the right column is the Debye exterior. All
plotted FDTD curves use the finest requested mesh, 0.5 mm by default.
The top row compares reflection phase, the middle row compares reflection
magnitude, and the bottom row separates continuum and discrete phase errors.
In the magnitude row, dots are FDTD and solid lines are continuum theory.

With the ``exp(+j*omega*t)`` convention, phase is displayed continuously
near 180 degrees to avoid a wrap at plus/minus 180 degrees. The more
conductive wall lies closer to PEC: reflection magnitude approaches one
and electric reflection phase approaches 180 degrees. The 1,000 S/m wall
has a larger finite-impedance phase shift and greater loss. The Debye
exterior changes both through its complex wave impedance.

The bottom-panel error is
:math:`\arg(\Gamma_{FDTD}/\Gamma_{reference})`, converted to degrees.
Continuum error includes the surface fit and boundary discretization errors;
discrete error measures agreement with the expected implemented scheme,
including finite-record numerical residuals. Both use the same measured
coefficient de-embedded with the discrete host wavenumber. Consequently,
these are wall errors after propagation correction, not the uncorrected
physical phase error at the receiver.

.. figure:: results/reflection_phase/reflection_phase.png
   :alt: Reflection phase, magnitude, and continuum and discrete phase errors in air and Debye material.

   Finite-conductivity wall reflection on the 0.5 mm mesh.

Reflection results and acceptance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The retained 2026-09-07 fine-mesh results are:

.. list-table:: Reflection errors on the 0.5 mm mesh
   :header-rows: 1

   * - Exterior
     - Wall conductivity (S/m)
     - Continuum phase RMS (degrees)
     - Discrete phase RMS (degrees)
     - Continuum magnitude RMS
   * - Air
     - 1,000
     - 0.00035899
     - 0.00000346
     - 0.00000565
   * - Air
     - 58,000,000
     - 0.00000149
     - 0.0000000143
     - 0.0000000241
   * - Debye
     - 1,000
     - 0.00149638
     - 0.00005359
     - 0.00005251
   * - Debye
     - 58,000,000
     - 0.00000620
     - 0.000000214
     - 0.000000229

Magnitude RMS is the RMS of :math:`|\Gamma_{FDTD}|-|\Gamma_{continuum}|`,
not a percentage or a dB value. Phase RMS is taken over all 29 frequencies.
For the 1,000 S/m wall, refining from 1 to 0.5 mm reduces continuum phase
RMS from 0.001468 to 0.000359 degree in air and from 0.006124 to 0.001496
degree in Debye, approximately a factor of four. Discrete errors are much
smaller and change little with refinement, consistent with the discrete
reference accounting for the leading boundary discretization error.

All eight host/conductivity/mesh combinations pass their gates.
Every plane-wave case must have continuum phase RMS error below 0.1 degree
and magnitude RMS error below 0.002. The discrete phase RMS gate is
0.0002 degree, with complex relative L2 below 0.00001. A regression test
intentionally removes boundary polarization history and verifies that this
stricter phase gate fails: the coarse Debye/1,000 S/m case increases from
0.00005387 to 0.001994 degree discrete phase RMS. This makes the comparison
sensitive to the new dispersive-contact update. Additional gates check
transverse uniformity, incident spectral coverage, and decay in the final
tenth of the record.

Results are in ``results/reflection_phase/``: ``reflection_phase.png``,
``summary.json``, and one complex-reflection CSV per host, conductivity and
mesh. The driver supports ``--reuse``, ``--output-dir``, ``--mesh-mm`` and
``--conductivity``, plus ``--host air`` or ``--host debye``. It returns nonzero
if acceptance fails; raw HDF5 files stay in the ignored ``_cache`` directory.
See ``analytical_validation_report.md`` for the complete reference formulas
and source citations. This case establishes normal-incidence planar
accuracy; it does not establish arbitrary curved dispersive-interface
accuracy.

Copper-wall waveguide
---------------------

The copper excitation covers 120--150 GHz with 31 one-GHz DFT points. Its
source port uses all 31 frequencies plus four guards for 35 anchors, while the
passive ports use 11 guarded anchors. This preserves exact source ``neff``
validation and modal injection without repeating dense FDFD solves at both
propagation monitors. The 80--180 GHz fit explicitly uses
``fit_order='auto'`` and selects three poles. The case uses a 0.1 mm cubic
grid, a 210 mm domain, source/passive planes at 90, 105, and 145 mm, and a
500 ps record. The record ends 97.415 ps before the conservative earliest
wall return. The retained result includes bulk numerical-dispersion
compensation. Its impedance-fit, FDFD-attenuation, and FDTD-attenuation errors
are 0.026023%, 0.681438%, and 0.759867%; maximum :math:`S_{11}` is
-101.0893 dB. The four-thread rerun on 2026-09-04 took 147.019 s including
analysis and plot generation.

The figure uses four panels: actual and fitted :math:`Z_s`,
driven-port :math:`S_{11}`, FDFD effective-index attenuation against
perturbation theory, and FDTD two-plane :math:`S_{21}` attenuation against
perturbation theory. The acceptance gates are -20 dB maximum reflection and
1%/2% FDFD/FDTD attenuation error.

Run from the repository root::

    python -m testing.validation.impedance_surface.validate_copper_wall_waveguide --threads 4

By default, outputs are written under:

.. code-block:: text

    results/copper_wall_waveguide/
        copper_wall_waveguide.png
        summary.json

The directory also contains its corresponding CSV table. Pass
``--output-dir`` to select a different location, and ``--reuse`` to regenerate
CSV, PNG, and JSON results from an existing compatible local cache without
repeating the FDTD run.
