"""Trim solution at the cruise condition.

Purpose
-------
Find the angle of attack and tail incidence that give lift equal to weight and
zero pitching moment about the CG. Solved inside each analysis call rather than
by the optimiser.

Inputs
------
- Aircraft geometry and airfoil data (via ``aerodynamics.lifting_line``).
- Weight [N] and CG position [m] from ``mass``.
- Airspeed [m/s] and air density [kg/m^3].

Outputs
-------
- Trim angle of attack [deg] and tail incidence [deg].
- Trimmed lift coefficient CL and induced drag coefficient CDi.
- A flag or residual showing whether a trim solution was found within limits.
"""
