"""Battery model.

Purpose
-------
Describe the battery: capacity from mass, charge and discharge efficiency,
MPPT and charge-controller losses, and state-of-charge limits.

Inputs
------
- Battery mass [kg] and specific energy [J/kg].
- Charge, discharge and MPPT efficiencies (floats, 0-1).
- Reserve state of charge (float, 0-1).
- Power in from the panels and power out to the aircraft [W].

Outputs
-------
- Battery capacity [J].
- Rate of change of stored energy [W], given power in and out, with charging
  smoothly limited at full charge.
"""
