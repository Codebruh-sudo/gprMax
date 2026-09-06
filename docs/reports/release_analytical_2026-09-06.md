# Fresh analytical verification, 6 September 2026

All **18 main drivers (8 CPU, 10 CUDA) and 2 additional two-rank MPI drivers completed**. The main campaign has **16 gated drivers and 2 report-only drivers**. **178/178 explicit checks passed: 154 main + 24 MPI**, with no analytical gate failure. This is 161 numerical-versus-physical-reference checks, 6 Mie-reference residual checks, 2 normalization identities, 1 long-time stability ratio and 8 source/mesh/finite-validity checks—not 178 independent physical experiments or pytest cases.

Counts below are driver-named checks/separate threshold predicates. A named validity conjunction counts once; an aggregate `passed` flag or each sample under a maximum gate does not add checks. `N+V` means numerical/error/consistency checks plus validity checks. Fractions in the JSON are converted to percent only where `%` is shown here.

| Case | Driver / resolution | Passed checks | Selected worst result → gate |
| --- | --- | ---: | --- |
| 1 | CPU FDFD; 0.25-mm 1-D, 1-mm 2-D mesh | 4 | Effective-index relative error: circular TE11 1.04305% → 1.5%; other geometries ≤0.0329291% → 0.1%. Four maxima, 30 reported eigenvalues, not 30 gates. |
| 2 | CPU Fresnel, six materials; 0.5 mm, 0.25–8 GHz | 18 | Reflection magnitude RMSE 0.00120417 → 0.005; complex L2 0.244243% → 1%; phase RMSE 0.0203355° → 0.1°. |
| 3 | CPU realistic halfspaces; clay 0.5 mm, water 0.05 mm | 6 | Water is worst: magnitude RMSE 0.00230561 → 0.01; complex L2 0.362488% → 2%; phase RMSE 0.112742° → 0.5°. |
| 4 | CPU R/C/L/RC/RLC TEM sheets; 1 mm, 1–30 GHz | 11+1 | Magnitude RMSE 0.00610431 → <0.02; phase RMSE 0.395095° → <2°; long-inductor late/initial peak ratio 0.00499493 → <0.02. |
| 5 | CPU power normalization; 2 mm, 0.75/1/1.25 GHz | 2 | 4-W/1-W error 0; accepted/incident identity error 6.66e-16 → 5e-12. Algebraic consistency, not independent EM accuracy. |
| 6 | CPU SAR halfspace; 2 mm, fourteen 0.5–7-GHz frequencies | 2+1 | Cell-average SAR profile L2 5.68565% → 6.5%; max pointwise 6.99754% → 8%, both at 7 GHz. |
| 7 | CPU grounded-slab reflection; 0.25 mm, 12-mm slab | 3 | Max magnitude error 2.10e-8 → 1e-6; max phase 0.0552533° → 0.1°; complex L2 0.0309280% → 0.1%. |
| 8 | CPU multilayer; 0.5 mm, five stacks × two constructions | 0 | **Report-only.** Worst averaged/staircased complex L2 0.0812596% / 3.69979%; phase RMSE 0.0262066° / 2.34288°. |
| 9 | CUDA Hertzian; 1 mm, 100³ domain, double | 6 | Direct significant-window near-field L2 0.0335327% → 0.1%; KSIR 0.0166692% → 0.1%; largest far directivity gate 0.000158234 → 0.001. |
| 10 | CUDA PEC Mie; 0.5 mm, radius 16 mm, single | 2 | RCS RMS 0.441689 dB → 0.75 dB; max absolute 0.953946 dB → 1.25 dB, 34 frequencies 0.75–9 GHz. |
| 11 | CUDA dielectric Mie, εr=4; same resolution | 2 | RCS RMS 0.270687 dB → 0.75 dB; max absolute 0.724350 dB → 1.25 dB. |
| 12 | CUDA SAR sphere; 0.75 mm, radius 18 mm, 1 GHz | 2 | Absorbed-power / absorption-cross-section relative errors both 6.36748% → 8%. |
| 13 | CUDA SAR muscle cylinder, TM+TE; 0.4 mm, radius 60 mm, 5.5 GHz | 10+2 | Interior L2 3.45775% → 5%; max pointwise 15.4035% → 20%; integrated power error 0.948949% → 2%. |
| 14 | CUDA SAR fat cylinder; same resolution | 10+2 | Interior L2 0.717463%; max pointwise 11.5225%; power 0.188549%; same limits. |
| 15 | CUDA SAR skin cylinder; same resolution | 10+2 | Interior L2 2.80033%; max pointwise 14.9777%; power 0.732908%; same limits. |
| 16 | CUDA grounded time-domain HED/VED; 0.1 mm | 4 | Peak-normalized waveform max error 0.914764% → <8%; RMS 0.287289% → <3%, reduced-time window 0–100 ps. |
| 17 | CUDA grounded dipole patterns; 1.5 mm, six cases × three frequencies | 54 | Max peak-normalized field 3.83289% → 4%; power 5.06913% → 5.5%; max-directivity relative error 3.18775% → 4%. |
| 18 | CUDA core-shell; 4 mm, 160³, radii 60/100 mm | 0 | **Report-only.** Averaged/staircased RCS L2 10.1187% / 15.1069%; RMS 0.897009 / 1.10408 dB; max absolute 3.33641 / 3.06187 dB. |
| 19 | MPI Hertzian, 2×1×1 ranks, double; case 9 mesh | 6 | Direct/KSIR significant-window near L2 0.0335327% / 0.0166692%; same limits as CUDA. |
| 20 | MPI Fresnel, 2×1×1 ranks, double; case 2 mesh | 18 | All 18 analytical metric values equal the fresh CPU baseline. |

Important scope limits:

- Reflection comparisons use separate incident calibration, the predefined discrete interface and axial Yee propagation de-embedding; frequency/signal masks apply. Relative L2 is `norm(numerical-reference)/norm(reference)`. Mie dB error is `10 log10(RCS_numerical/RCS_reference)`.
- Hertzian near-field gates retain analytical `|Ez| ≥ 1%` of peak. Ungated full-history direct L2 is **0.601176%** (CUDA; MPI 0.601175%). Its prescribed excitation is shared with the independent EM formula. FDFD theory supplies the eigensolver shift guess: this is theory-assisted eigenvalue validation, not blind mode finding.
- Cylinder local gates exclude the two-cell interface band and retain analytical SAR ≥5% of peak. Boundary-including report-only comparisons keep that 5%-peak mask; their errors reach **12.3335% L2 (muscle TE)** and **75.4254% max pointwise (skin TE)**. Integrated power/cross-section are separately gated; worst Mie residual is 6.07e-16 → 1e-12. TE/fat/skin extend the literature muscle-TM case.
- SAR sphere 1-g/10-g peaks (**3.77745e-5 / 3.16196e-5 W/kg**) are report-only, not analytically gated mass averages. The fresh campaign contains no 0.5-mm sphere refinement. Halfspace Yee-collocated report-only errors are 3.05986% L2 / 3.94535% max, distinct from the gated cell-average reference.
- Grounded patterns fit one common complex scale per source/frequency across both cuts: shape/directivity are checked, not absolute source calibration. The grounded-time reference uses saved sampled excitation with finite-difference differentiation and a fixed predelay. Water properties come from the production material generator; network RC uses the driver's documented curve-matching capacitance rather than the paper's printed value.
- Multilayer/core-shell have **no numerical acceptance threshold**. Their zero exits certify completed reporting only. Dense analytical plot curves are not extra simulations: multilayer has 150 FDTD frequency samples per construction; core-shell has 87.

Fresh MPI parity: six of seven Fresnel receiver traces are bitwise identical to CPU; Debye-3-pole relative L2 difference is 1.66e-24. Hertzian MPI/CUDA direct-near trace difference is 2.15e-8 relative L2; this is a reported comparison, not a newly imposed gate. No numerical-accuracy improvement over the historical audit is asserted.

The committed [compact evidence ledger](release_analytical_2026-09-06.json) contains all 20 commands, grouped thresholds, resolutions and caveats. Set `${PYTHON}`, `${MPIEXEC}` and `${OUT}` for the local installation as indicated by the ledger. The [portable MPI diagnostic helper](support/cross_backend_analytical.py) delegates to the same maintained analytical scene builders and acceptance functions; both of its MPI cases were replayed after making its repository path portable. Full raw numerical files, per-check ledger and aggregation scripts are retained locally under `/tmp/gprmax-release-closeout-20260906/analytical/`.
