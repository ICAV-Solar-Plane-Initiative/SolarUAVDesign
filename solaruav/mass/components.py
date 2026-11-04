"""Component mass breakdown.

Purpose
-------
Collect every mass item on the aircraft (structure, battery, solar panels,
avionics, payload) with its position, and close the total mass.

Inputs
------
- Structural masses from ``scaling`` or ``structures`` [kg].
- Battery mass [kg] (design variable).
- Panel area [m^2] and areal density [kg/m^2].
- Fixed masses: avionics and payload [kg].
- Body-axis position of each item [m].

Outputs
-------
- List of mass items, each with name, mass [kg] and position (x, y, z) [m].
- Total aircraft mass [kg] and weight [N].
"""
