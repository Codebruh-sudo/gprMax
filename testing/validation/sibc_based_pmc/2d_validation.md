# Exact SIBC PMC in 2D TE/TM and virtual guides

The native `resistance=float('inf')` boundary was checked in all invariant-axis orientations and both propagation directions. It retains electric dual mass and circulation, with exactly zero admittance and surface current.

The data contain 12 independent 3D-extrusion comparisons, 96 physical/virtual comparisons, 4 20,000-step runs, and 24 analytical discrete-mode checks.

For the 16-cell parallel-wall guide, the TM fundamental is the constant Neumann mode with operator index 1. The TE fundamental has transverse symbol `2 sin(pi/32)/dl`, giving `sqrt(1-(kt/operator_k0)^2)`. Maximum relative modal errors are 1.22779e-16 in double and 8.456e-10 in single precision.

See [machine-readable results](results/2d_results.json) and the [general 2D validation report](../impedance_surface/results/2d_sibc/README.md) for field errors, finite-PML reflection, history bounds, and reproduction commands. The 3D reference uses periodic invariant images; no legacy PMC volume is used as an oracle.
