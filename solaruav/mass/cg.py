"""Centre of gravity.

Purpose
-------
Find the aircraft's centre of gravity from the component mass breakdown.

Inputs
------
- Mass items from ``components``: mass [kg] and body-axis position (x, y, z) [m].

Outputs
-------
- CG position (x, y, z) in body axes [m].
- CG as a fraction of the reference mean aerodynamic chord, for stability.
"""
