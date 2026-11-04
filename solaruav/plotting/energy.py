"""24-hour energy plot.

Purpose
-------
Plot solar power in, power out and battery state of charge over the day-night
cycle, with sunrise, sunset and the charge margin marked.

Inputs
------
- Time history from ``energy.mission``: times [s], power in and out [W],
  state of charge (0-1).

Outputs
-------
- Matplotlib Figure and Axes with the energy history.
"""
