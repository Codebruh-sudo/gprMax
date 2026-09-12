==========================
Surface-impedance examples
==========================

``rectangular_waveguide_comparison`` compares the dominant TE10 mode of the
same rectangular guide with ideal PEC walls and with thick copper walls
represented by a dispersive surface impedance.  It demonstrates the direct
geometry-material syntax, the impedance-aware FDFD mode solve, conductor loss
in the complex effective index, and the non-zero tangential electric field at
a finite-conductivity wall.

Run the commands in the example's local ``README.rst`` from the repository
root.

``virtual_waveguide_2d.py`` demonstrates actively driven TE and TM guides
with infinite-resistance PMC or fitted dispersive impedance walls, including
their uniformly extruded auxiliary PML. ``pec_pmc_contacts.py`` demonstrates
contacts with PEC/legacy PMC volumes and domain symmetry planes.

The user guide starts with the copper example and explains API choices,
geometry, timestep controls, and supported PML configurations. Its separate
surface-impedance theory page contains the discrete updates and validation
evidence.
