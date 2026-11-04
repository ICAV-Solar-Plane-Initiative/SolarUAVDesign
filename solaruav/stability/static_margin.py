"""Neutral point and static margin.

Purpose
-------
Find the longitudinal neutral point and the static margin, the distance
between neutral point and CG as a fraction of the mean aerodynamic chord.

Inputs
------
- Aircraft geometry and surface lift curve slopes (via ``aerodynamics``).
- CG position [m] from ``mass.cg``.
- Reference mean aerodynamic chord [m].

Outputs
-------
- Neutral point x position in body axes [m].
- Static margin (float, fraction of MAC; positive is stable).
"""
