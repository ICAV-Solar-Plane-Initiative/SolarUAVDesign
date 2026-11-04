"""Power required for steady level flight.

Purpose
-------
Convert total drag into the electrical power the aircraft draws in cruise,
P = D V / (eta_prop * eta_motor). Drag comes either from the lifting-line and
viscous drag models or from a user-specified drag polar.

Inputs
------
- Total drag [N], or a drag polar CD(CL) plus wing area [m^2].
- Airspeed [m/s] (float or array for a power-required curve).
- Propeller and motor efficiencies (floats, 0-1).
- Avionics and payload electrical power [W].

Outputs
-------
- Shaft power and electrical power required [W] (float or array matching speed).
- Speeds for minimum power and maximum range [m/s].
"""
