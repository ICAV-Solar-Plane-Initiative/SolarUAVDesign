"""Viscous drag of the whole aircraft.

Purpose
-------
Estimate the parasite (zero-lift and lift-dependent viscous) drag: section
profile drag of each lifting surface plus fuselage, tail and interference drag.

Inputs
------
- Aircraft geometry: surface areas and chords [m^2, m].
- Section profile drag data per surface: Cd as a function of Cl and Reynolds
  number (polar table or fitted curve).
- Spanwise Cl distribution from ``lifting_line``.
- Flight condition: airspeed [m/s], air density [kg/m^3], dynamic viscosity [Pa s].
- Fuselage wetted area [m^2] and form factor.

Outputs
-------
- Viscous drag coefficient CD0 referenced to the wing area (float).
- Drag breakdown by component [N], for checking and plotting.
"""
