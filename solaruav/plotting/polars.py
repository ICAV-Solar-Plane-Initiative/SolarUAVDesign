"""Drag and power polars.

Purpose
-------
Plot the aircraft's drag polar (CD against CL) and power-required curve
(power against airspeed).

Inputs
------
- CL and CD arrays, or results from ``aerodynamics``.
- Airspeed [m/s] and power [W] arrays for the power curve.
- Optional marked points (cruise, minimum power, stall).

Outputs
-------
- Matplotlib Figure and Axes with the polars.
"""
