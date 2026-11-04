"""Wing loads at the design load cases.

Purpose
-------
Build the spanwise shear force, bending moment and torque on each lifting
surface at the limit and ultimate load factors.

Inputs
------
- Spanwise lift distribution from ``aerodynamics.lifting_line`` [N/m].
- Spanwise mass distribution (structure, panels, battery if in the wing) [kg/m].
- Limit load factor and safety factor (floats).

Outputs
-------
- Shear force [N], bending moment [N m] and torque [N m] at each spanwise
  station, as arrays of length n_points.
"""
