"""Solar panel power income.

Purpose
-------
Turn irradiance into electrical power, accounting for the panel area, the
angle between the sun and the panels, and cell and packing efficiency. The
detailed box-packing of real panel layouts is done here as a post-processing
step only, never inside the optimisation loop.

Inputs
------
- Irradiance and sun position from ``irradiance`` [W/m^2, deg].
- Panel area [m^2] (wing upper-surface area times panel coverage fraction).
- Panel normal directions, from the surface geometry and aircraft attitude.
- Cell efficiency and packing efficiency (floats, 0-1).

Outputs
-------
- Electrical power from the panels [W] at each time, as an array.
- (Post-processing) a panel layout: number and position of cells on each surface.
"""
