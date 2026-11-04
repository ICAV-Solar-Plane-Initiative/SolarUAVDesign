"""Solar irradiance over the day.

Purpose
-------
Compute the sun's position and the solar irradiance available at the aircraft
as a function of place and time.

Inputs
------
- Latitude [deg] and longitude [deg].
- Date (day of year) and times of day [s from midnight] as an array.
- Altitude [m].

Outputs
-------
- Sun elevation and azimuth [deg] at each time.
- Direct and diffuse irradiance [W/m^2] at each time, as arrays matching the
  input times.
- Sunrise and sunset times [s from midnight].
"""
