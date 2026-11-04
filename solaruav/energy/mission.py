"""24-hour mission simulation and the charge margin.

Purpose
-------
Step the battery state of charge through one full day-night cycle and compute
the charge margin, the objective of the optimisation.

Inputs
------
- Solar power in [W] over the day, from ``solar.panels``.
- Cruise electrical power required [W], from ``aerodynamics.power``.
- Battery model from ``battery``.
- Time step [s]; the simulation runs from sunrise to the next sunrise, starting
  at the reserve state of charge.

Outputs
-------
- Time history of power in, power out and state of charge, as arrays (for plotting).
- Charge margin [h]: (energy at the next sunrise - reserve energy) / cruise power.
"""
