# Impedance-surface and eigenmode documentation rewrite

This inventory maps the feature documentation at commit `1cc9e83c` to its
new homes. It is a review aid, not an additional user tutorial. Runtime APIs,
solver code, example model code, and recorded physics results are unchanged.

## Content ownership

| Original material | New home | Treatment |
| --- | --- | --- |
| Surface purpose, quick start, model choices, metal applicability | Surface-impedance practical guide: First copper model; Choosing a surface model | Lead with the existing copper-versus-PEC workflow. Explain ideal resistance and exact PMC afterward. |
| Surface constructor defaults and fit/plot controls | Shared surface argument table, included by practical guide and Python reference | One table supplies both pages. Distinguish bulk conductivity in S/m from surface resistance in ohms. |
| Geometry semantics, supported primitives, ordered overwrites, contacts, symmetry | Surface practical guide: Building geometry; theory: Discrete geometry and contacts | Practical geometry rules and remedies precede exact voxel/quadrant classification, retained masses, and component constraints. |
| Automatic timestep margin | Practical guide: Running reliably; theory: Clipped-circulation stability | Keep command, logging, unused-declaration trigger, reuse behaviour, and time-window consequences in the guide; retain the formula and CFL evidence in theory. |
| Solver restrictions, PML and virtual-guide wall requirements | Surface practical guide: Supported solver configurations; Waveguides and absorbers | Canonical SIBC compatibility rules, including physical-axis padding in 2D. Other references link here. |
| SIBC/PML update, source corrections, independent histories, TE/TM live layers | Surface theory: PML and virtual-guide coupling | Retain discrete equations and the all-orientation/long-run validation drivers. Link to the eigenmode aperture equations. |
| Exact voxel-face PMC | Practical model choice and warning explanation; theory: Exact voxel-face PMC | Preserve zero admittance, retained clipped update, legacy reflection-plane limitation, warning exclusions, and PMC validation location. |
| Continuous boundary conventions, rational realization, metal Foster fitting, trapezoidal ADE, passivity | Surface theory, opening sections | Preserve equations, applicability conditions, fitting details, and citations. |
| Sparse locally implicit update, dispersive retained quadrants, exact FDFD reduction, modal injection, rectangular-guide analysis | Surface theory | Preserve all discrete/continuous qualifications and analytical comparisons. Move supported dispersive-host mathematics out of the future-extension section. |
| Surface benchmarks, sphere/reflection tests, reproduction commands and metrics | Surface theory: Validation and benchmarking | Existing evidence and reports retained; no new simulation claims substituted. |
| Surface HDF5 schema and inspection example | Output reference: Surface-impedance reproducibility metadata | One canonical schema description; links from both guides and theory. |
| Surface troubleshooting, implementation map, future extensions | User remedies in practical guide; internal realization diagnostics, implementation map, and extensions at end of theory | Keep public users out of internal state-space modification instructions. |
| Eigenmode band/port/excitation/virtual-guide APIs | Eigenmode practical guide and shared tables included by Python reference | Explain workflows once, preserve argument details, and retain compact hash equivalents. Full grammar stays in hash reference. |
| First two-port scene and mode inspection | Eigenmode practical guide: First two-port model | Retain a complete Python scene, geometry-only/full-run sequence, output names, and expected results. |
| Port placement, anchors, waveform, 2D setup | Eigenmode practical guide: Configuring ports | Explain output bins versus modal solves, finite recording time, and invariant-axis TE/TM versus propagation-axis waveguide mode names. |
| Virtual-waveguide explanation, diagram, arguments, active/passive setup and NTFF purpose | Eigenmode practical guide: Terminating the feed | Preserve the physical explanation; move coupling equations to theory and link SIBC restrictions to the surface guide. |
| Degenerate-group syntax, physical directions, TE11 examples | Eigenmode practical guide: Selecting modal channels; hash/API/output references | Keep channel selection and plotting instructions in the guide; reconcile missing hash options and group-specific failure semantics. |
| Moment alignment, SVD transport, eigenvalue/residual/conditioning safeguards and metadata | Eigenmode theory: Degenerate subspaces and physical alignment; output reference | Keep all numerical thresholds, phase conventions, full power matrices, and restrictions. |
| Coefficients, validity masks, complete S matrix, study schedule, result lookup, coherent active reflection | Eigenmode practical guide: Measuring a network | Preserve the worked two-channel matrix explanation and Python study example. Link detailed projections and power equations to theory. |
| Seven numbered applications and standard TE11 modal figures | Eigenmode practical guide: Applications and troubleshooting; example READMEs | Retain all examples, expected observations, commands, and figures. Replace repeated full-file listings with source downloads. |
| HSG, backend scope, lossy/cutoff interpretation and accuracy advice | Eigenmode practical guide: Applications and troubleshooting | Distinguish ordinary port support from stricter SIBC and reusable-study support. |
| FDFD 1D/2D operators, Yee shapes, reconstruction, power, phasors, I/Q synthesis, interpolation, staggering, TF/SF injection, DFT/Gram reception, antenna normalization | Eigenmode theory | Preserve derivations and stable cross-references; qualify independent-mode tracking separately from declared groups. |
| Second-order HORIPML guard mathematics in Python reference | Surface theory: Second-order HORIPML profile guard | Keep the existing PML reference label attached to practical error remedies; link the analytical condition and evidence. |

## Consistency and navigation

- Existing feature page URLs and all original explicit reference labels remain available.
- New theory pages follow the practical guides in the documentation navigation.
- Shared tables use a scoped wrapping style so long descriptions fit the HTML content column.
- Correct stale “3D only”, “cannot enter PML”, example-count, bulk discrete-symbol,
  and per-mode fallback descriptions in affected references and example guidance.
- Preserve `plot_fit`'s geometry-only behaviour at `False`, separately from
  `plot_fields=False`, which suppresses modal plots.
- Keep the timestep margin distinct from the custom-PML profile guard.
- Preserve both original TE11 PNG assets without regeneration or modification.
- Explain degeneracy as independent patterns sharing a propagation constant,
  distinguish arbitrary solver-basis rotation from physical beating, and
  explain how group tracking, physical labels, and excitation selection differ.

## Verification

- Compared all displayed `.. math::` blocks from the two original feature
  pages with their destinations after whitespace normalization: all 51
  surface and 44 eigenmode blocks are retained.
- Built baseline and rewritten HTML with Sphinx 9.1.0 and the Read the Docs
  theme. The Windows runtime lacks ReFrame, so both builds use
  `-D autosummary_generate=0`; existing ReFrame import/orphan-page warnings
  and unrelated formatting warnings are recorded separately from this rewrite.
  The final clean build has no errors and no warnings in the changed pages.
  Its 58 remaining warnings comprise 19 ReFrame autodoc imports, 19 ReFrame
  autosummary imports, 19 developer pages outside a toctree, and one unrelated
  Utilities README heading. The baseline had 64 warning/error messages.
- Audited local links, fragments, image assets, and downloads across both
  guides, both theory pages, API/hash/output references, feature/example
  indexes, and the main index. No broken local destinations found.
- Visually inspected guide navigation, wrapped API tables, relocated
  alignment equations, and the standard TE11 profile figures in HTML.
- Geometry-only runs passed for all seven eigenmode examples, both PEC/copper
  comparison cases, and the contact model with and without PMC symmetry.
- Full runs passed for the PEC/copper comparison, straight two-port guide,
  2D TM infinite-resistance virtual guide, and 2D TE fitted-surface virtual guide.
- Copper and straight-guide result plotters passed. The copper check reports
  positive attenuation of 5.43967 dB/m at 140 GHz and a nonzero wall/centre
  electric-field ratio of 0.000243035; the corresponding PEC ratio is zero.
- Parsed seven documented material/degenerate hash snippets in minimal
  appropriate command context; both existing Python/hash equivalence tests passed.
- All eight original explicit reference labels remain defined exactly once;
  all 16 Python snippets in the two practical guides compile successfully.
- The complete physics regression suite was not rerun for this documentation-only change.

Reproduce the HTML check from `docs` with an environment containing the
documentation dependencies:

```console
python -m sphinx -E -a -b html -D autosummary_generate=0 source _build/html
```

Normal documentation environments with ReFrame installed can retain the
configured autosummary generation. Run example commands from the repository
root as listed in the practical guides; use output-path options to keep
generated artifacts separate from checked-in examples.
