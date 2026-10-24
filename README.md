# SolarUAVDesign
Developing conceptual and preliminary design tools for ICAV-SPI's solar plane program.

## Install

```bash
pip install -e ".[dev]"
```

## Geometry

Aircraft are built from 2D lifting surfaces. Each surface is a list of spanwise
stations (spanwise position `s`, leading-edge `x`, chord), joined by straight-tapered
panels. Area, span, aspect ratio, taper, mean aerodynamic chord and sweep are all
derived from the stations.

```python
from solaruav import Aircraft, LiftingSurface

wing = LiftingSurface.from_panels(
    "wing", root_chord=0.32,
    panels=[{"span": 1.5}, {"span": 1.0, "taper": 0.6, "sweep_le_deg": 3}],
)
htail = LiftingSurface.trapezoidal("htail", span=1.0, root_chord=0.2, taper=0.8,
                                   origin=(1.6, 0, 0))
fin = LiftingSurface.trapezoidal("fin", span=0.4, root_chord=0.28, taper=0.6,
                                 sweep_le_deg=25, symmetric=False,
                                 orientation="vertical", origin=(1.58, 0, 0))

ac = Aircraft("Concept A", [wing, htail, fin], reference_surface="wing")
print(ac.summary())
fig, axes = ac.plot()   # top and side views
```

Conventions: body axes are x aft, y to starboard, z up. Each surface has its own
local `(x, s)` plane, placed on the aircraft by `origin` and `orientation`
(`"horizontal"` maps `s` to y, `"vertical"` maps it to z). Symmetric surfaces are
mirrored about their origin, and `span` is tip-to-tip.

There are three ways to define a surface:

- `LiftingSurface.trapezoidal(...)`: single panel from span, root chord, taper and sweep.
- `LiftingSurface.from_panels(...)`: panel by panel from the root outward.
- `LiftingSurface(name, stations=[Station(s, x_le, chord), ...])`: full control.

See `examples/reference_aircraft.py` for a complete example.

## Tests

```bash
pytest
```
