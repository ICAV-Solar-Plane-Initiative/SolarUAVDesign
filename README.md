# SolarUAVDesign
Developing conceptual and preliminary design tools for ICAV-SPI's solar plane program.

## Install

From the repository folder:

```bash
pip install -e ".[dev]"
```

## Geometry

Aircraft are built from flat (2D) lifting surfaces: wings, tailplanes and fins.

Each surface is a list of **stations** running from root to tip. A station records
its position along the span, the position of its leading edge, and its chord.
Between two stations the surface is a straight-tapered panel. Everything else
(area, span, aspect ratio, taper ratio, mean aerodynamic chord, sweep) is
worked out from the stations.

```python
from solaruav import Aircraft, LiftingSurface

wing = LiftingSurface.from_panels(
    "wing",
    root_chord=0.32,
    panels=[
        {"span": 1.5},                                      # rectangular centre section
        {"span": 1.0, "taper": 0.6, "sweep_le_deg": 3.0},   # tapered outer panel
    ],
)
htail = LiftingSurface.trapezoidal(
    "htail", span=1.0, root_chord=0.2, taper=0.8, origin=(1.6, 0.0, 0.0)
)
fin = LiftingSurface.trapezoidal(
    "fin", span=0.4, root_chord=0.28, taper=0.6, sweep_le_deg=25.0,
    symmetric=False, orientation="vertical", origin=(1.58, 0.0, 0.0),
)

aircraft = Aircraft("Concept A", [wing, htail, fin], reference_surface="wing")

print(wing.area, wing.aspect_ratio, wing.mac)   # individual properties
print(aircraft.summary())                       # every surface at once
fig, axes = aircraft.plot()                     # top and side views
```

### Three ways to define a surface

| Constructor | Use it for |
|---|---|
| `LiftingSurface.trapezoidal(...)` | A single tapered panel from span, root chord, taper and sweep |
| `LiftingSurface.from_panels(...)` | Several panels, built one at a time from the root outward |
| `LiftingSurface(name, stations=[Station(...), ...])` | Full control over every station |

### Conventions

- **Units:** SI (m, m²). Angles are in degrees, and their argument names end in `_deg`.
- **Body axes:** x aft, y to starboard, z up.
- **Placing a surface:** each surface is drawn in its own flat plane, then placed
  on the aircraft by `origin` (where its root leading edge sits) and
  `orientation` (`"horizontal"` lays the span along y, `"vertical"` stands it up
  along z).
- **Symmetric surfaces:** the stations describe one side, and the other side is
  mirrored. `span` and `area` cover both sides.

See `examples/reference_aircraft.py` for a complete, runnable example.

## Code layout

```
solaruav/
├── geometry/
│   ├── surface.py    # Station and LiftingSurface: defining a planform and its properties
│   └── aircraft.py   # Aircraft: a set of surfaces, reference values, tail volumes
└── plotting.py       # plot_surface, plot_aircraft: top and side views
tests/                # pytest tests, checked against hand calculations
examples/             # runnable scripts
```

## Tests

```bash
pytest
```
