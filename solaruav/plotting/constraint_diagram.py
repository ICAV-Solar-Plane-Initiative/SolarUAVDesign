"""Constraint diagram.

Purpose
-------
Plot power loading against wing loading, with a curve for each sizing
constraint (cruise, stall, climb, energy closure), so the feasible design
region can be seen at a glance.

Inputs
------
- Range of wing loadings [N/m^2].
- Aerodynamic, propulsive and energy parameters for each constraint.
- Optional design points to mark.

Outputs
-------
- Matplotlib Figure and Axes with the constraint curves and feasible region.
"""
