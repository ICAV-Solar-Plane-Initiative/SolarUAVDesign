"""Beam-theory analysis of the wing structure.

Purpose
-------
Treat each wing as a cantilever beam to find stresses, deflections and
rigidities, and the structural mass needed to carry the loads.

Inputs
------
- Spanwise loads from ``loads``: shear [N], bending moment [N m], torque [N m].
- Spar and skin section geometry along the span [m].
- Material properties: Young's modulus E and shear modulus G [Pa], density
  [kg/m^3], allowable stress [Pa].

Outputs
-------
- Flexural rigidity EI [N m^2] and torsional rigidity GJ [N m^2] along the span.
- Maximum bending stress [Pa] and tip deflection [m], for the constraints.
- Structural mass [kg], fed back into ``mass.components``.
"""
