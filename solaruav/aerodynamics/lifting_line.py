"""Lifting-line solution for spanwise lift and induced drag.

Purpose
-------
Solve Prandtl's lifting-line equations for each lifting surface to get the
spanwise circulation and lift distribution, the total lift coefficient and the
induced drag.

Inputs
------
- Aircraft geometry (``Aircraft`` / ``LiftingSurface``): stations giving span,
  chord and twist.
- Airfoil data per surface: lift curve slope [1/rad], zero-lift angle [deg]
  and pitching moment coefficient Cm0.
- Flight condition: angle of attack [deg], airspeed [m/s], air density [kg/m^3].
- Number of spanwise collocation points (int).

Outputs
-------
- Spanwise lift coefficient distribution, Cl(s), as an array of length n_points.
- Total lift coefficient CL and induced drag coefficient CDi (floats).
- Span efficiency factor e (float).
- Pitching moment contribution of each surface [N m], for the stability model.
"""
