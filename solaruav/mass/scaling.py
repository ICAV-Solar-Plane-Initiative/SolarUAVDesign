"""Empirical structural mass scaling.

Purpose
-------
First-estimate structural mass of the wing, tail and fuselage from simple
scaling laws, used until the beam-based structural model replaces it.

Inputs
------
- Surface geometry: span [m], area [m^2], aspect ratio, thickness ratio.
- Total aircraft mass [kg] and ultimate load factor.
- Scaling-law coefficients (from literature or past ICAV-SPI aircraft).

Outputs
-------
- Structural mass of each surface and the fuselage [kg].
"""
