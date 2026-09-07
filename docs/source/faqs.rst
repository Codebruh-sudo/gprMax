****
FAQs
****

This section provides answers to frequently asked questions about gprMax and its uses. You should also check out our `YouTube channel <https://www.youtube.com/@Gprmax>`_ which contains screencasts and videos that explain how to install gprMax, demonstrate some of its key features, and give example models showing applications.

**What applications can gprMax simulate?**
gprMax is electromagnetic wave simulation software that is based on the Finite-Difference Time-Domain (FDTD) method. Many of its features have been designed to benefit simulating Ground Penetrating Radar (GPR), however, it can be used to simulate many other applications in areas such as engineering, geophysics, archaeology, and medicine.

**Why does gprMax not have a GUI?**
We considered developing a CAD-based graphical user interface (GUI) but, for now, decided against it. There were two guiding principals behind this design decision: firstly, users most often perform a series of related simulations with varying parameters to solve or optimize a particular problem; and secondly, we decided the limited resources we had were best concentrated on developing advanced modelling features for GPR within software that could easily be interfaced with other tools. Although a CAD-based GUI is useful for creating single simulations it becomes increasingly cumbersome for a series of simulations or where simulations contain heterogeneities, e.g. a model of a soil with stochastically varying electrical properties.

**How is gprMax licensed?**
gprMax is released under the `GNU General Public License v3 or higher <http://www.gnu.org/copyleft/gpl.html>`_. This means when distributing derived works, the source code of the work must be made available under the same license.

**Where does the name gprMax come from?**
The name gprMax comes from the joining of the acronym for Ground Penetrating Radar - **gpr** - and the name of the Scottish scientist who formulated the classical theory of electromagnetic radiation, `James Clerk Maxwell <https://en.wikipedia.org/wiki/James_Clerk_Maxwell>`_ - **Max**.

**Do I need to learn Python to use gprMax?**
No, you can use gprMax purely through commands in your input. However, gprMax also has a Python API which can be used to create more complex simulations and scripting.

**Can I still do all my pre/post-processing for gprMax in MATLAB?**
Yes. MATLAB has `built-in HDF5 functions <https://www.mathworks.com/help/matlab/hdf5-files.html>`_, and the :doc:`Utilities toolbox <inc_Utilities>` includes ``gprmax_read_h5.m`` for recursively loading a gprMax output and ``gprmax_h5_to_mat.m`` for creating MATLAB v7.3 MAT files. The utilities support receiver histories, ports, S-parameter studies, NTFF, SAR/radiometry, subgrids, attributes, strings, and complex arrays.

**Can I convert my output file to a text file, e.g. to import it into Microsoft Excel**
Yes, we recommend you download `HDFView <https://support.hdfgroup.org/products/java/hdfview/>`_ which is a free viewer for HDF files. You can then export any of the datasets in the output file to a text (ASCII) file that can be imported into Microsoft Excel. To do so right-click on the dataset in HDFView and choose Export Dataset -> Export Data to Text File.

**How do I choose a spatial resolution for my simulation?**
Spatial resolution should be chosen to mitigate numerical dispersion and to adequately resolve geometry in your simulation. :ref:`A 2D example of modelling a metal cylinder in a dielectric <example-2D-Ascan>` provides guidance on how to determine spatial resolution.

.. _spatial-resolution-diagnostic:

**What does the pre-simulation spatial-resolution diagnostic check?**

The diagnostic estimates the source bandwidth, then samples the material
responses within that band, including known relaxation and resonance scales.
It includes both relative permittivity and relative permeability. In a
lossless, nondispersive isotropic medium the physical phase speed is

.. math::

   v_p = \frac{c}{\sqrt{\varepsilon_r\mu_r}},\qquad
   \lambda = \frac{v_p}{f}.

For lossy or dispersive isotropic media it instead evaluates

.. math::

   k(\omega)=\frac{\omega}{c}
       \sqrt{\varepsilon_r^*(\omega)\mu_r^*(\omega)}
       =\beta-j\alpha,\qquad
   \lambda=\frac{2\pi}{|\beta|},\qquad
   \ell_\alpha=\frac{1}{\alpha}.

Here the star denotes the complex constitutive response, not complex
conjugation. Electric and magnetic conductivities are included once through
their respective responses. The convention is :math:`\exp(j\omega t-jkx)`,
with the attenuating branch selected. The phase velocity is
:math:`\omega/\beta`, not a group velocity or a pulse-arrival velocity. A
purely evanescent response has no finite phase wavelength or phase velocity;
its attenuation length is still reported.

The reported wavelength sampling is :math:`\lambda/\Delta_{\max}`, where
:math:`\Delta_{\max}` is the largest active spatial step. A 2-D TM or TE
model excludes its invariant direction. Sampling below the configured
minimum (three cells by default) stops the run. This minimum is a resolution
guard, not a guarantee of accurate propagation. Decreasing the timestep alone
does not improve spatial sampling. The separate attenuation diagnostic reports
cells per 1/e amplitude-decay length and warns below three cells; this is a
heuristic warning, not an attenuation-error bound. Perfect conductors and
artificial voltage-source material records are excluded from bulk-material
analysis. As in the previous diagnostic, the check considers the grid's
material catalogue, including unused definitions; it is not an occupancy
analysis of each material's geometry.

For a positive, lossless, nondispersive isotropic material, the numerical
phase-velocity estimate uses the homogeneous grid-axis Yee dispersion relation:

.. math::

   k_{\mathrm{num}} = \frac{2}{\Delta_{\max}}
       \sin^{-1}\!\left[
       \frac{\Delta_{\max}}{v_p\Delta t}
       \sin\!\left(\frac{\omega\Delta t}{2}\right)\right],\qquad
   e_v = 100\left(\frac{\omega/k_{\mathrm{num}}}{v_p}-1\right).

The material speed appears consistently in both the physical wavelength and
the discrete relation. See `Schneider, Understanding the FDTD Method,
chapter 7, equations 7.39--7.43
<https://eecs.wsu.edu/~schneidj/ufdtd/chap7.pdf>`_. The diagnostic reports the
largest absolute grid-axis error among the sampled frequencies and eligible
materials, not a guaranteed error for all propagation directions or for a
complete heterogeneous model.

For lossy or dispersive media, continuum wavelength and attenuation estimates
are available, but a numerical phase-error estimate for their discrete
material updates is **not** supplied. Positive, lossless, nondispersive
diagonal anisotropy uses a conservative index bound
:math:`\sqrt{\max_i\varepsilon_{r,i}\,\max_i\mu_{r,i}}`, explicitly labelled
as a bound rather than a scalar material velocity. Complex or indefinite
tensors are not certified by this bound; the diagnostic warns that a full
tensor propagation analysis is unavailable. Scalar constituent checks do not
replace that analysis.

These are setup-only checks shared by CPU and accelerator runs. MPI ranks
exchange compact diagnostic summaries so a failure stops all ranks before
field updates. Subgrids use their own spatial steps and timestep. No timestep,
update coefficient or material parameter is modified. CFL and dispersive
recurrence stability checks remain separate.

Waveforms with exactly zero amplitude are excluded before estimating bandwidth,
including those attached to passive voltage sources or transmission lines. The
sources themselves remain in the model with their normal receiving/loading
behaviour. If all waveforms have zero amplitude, the diagnostic is skipped.
Each nonzero pulse is analysed with its own sampling window, independent of
waveform declaration order. When a waveform cannot be characterised, a warning
identifies the missing estimate; any available bandwidth from other waveforms
is still checked. This partial check does not certify the uncharacterised band.

For an intentional under-resolution experiment, use:

.. code-block:: console

   python -m gprMax model.in --allow-underresolved

or the Python API:

.. code-block:: python

   gprMax.run(scenes=[scene], outputfile='experiment', allow_underresolved=True)

The default is ``False``. The override continues the diagnostic and changes
only the minimum wavelength-sampling rejection into a warning. It applies to
the run's grids and MPI ranks without modifying the fields, geometry or
timestep. It does **not** bypass material validation, CFL/dispersive stability
checks, or SAR/port frequency-output validity limits. Allowing a run does not
make its numerical results accurate.

Bandwidth detection retains its existing limitations: impulse and user-defined
waveforms may prevent an automatic estimate, and sinusoidal waveforms use the
existing conservative four-times-frequency band. A finite set of frequency
samples cannot prove that an arbitrary fitted material has no narrower feature
between samples. The diagnostics do not measure interface staircasing, PML
reflection or subgrid coupling error; convergence studies remain necessary.

**I specified a certain piece of geometry but I don’t see it when I view my geometry file.**
gprMax builds objects in a model in the order the objects were specified in the input file, using a layered canvas approach. This means, for example, a cylinder object which comes after a box object in the input file will overwrite the properties of the box object at any locations where they overlap. This approach allows complex geometries to be created using basic object building blocks.

**Can I run gprMax on my HPC/cluster?**
Yes. gprMax has been parallelised using hybrid MPI + OpenMP and also features a task farm based on MPI. For more information read the :ref:`HPC <hpc>` section.
