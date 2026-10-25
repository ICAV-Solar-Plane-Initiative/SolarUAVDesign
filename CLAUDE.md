# CLAUDE.md

Guidance for Claude (and humans) working in this repository.

## Project

SolarUAVDesign: conceptual and preliminary design tools for the ICAV Solar Plane
Initiative (ICAV-SPI), Imperial College London. The end goal is to estimate solar
energy income and endurance for a solar-powered UAV, and to size the aircraft so
that its 24-hour energy balance closes.

Users are aerospace engineering students. Many are confident engineers but newer
programmers, so **readability beats cleverness every time.**

## Commenting and Readbility
- Every class is accompanied by a description of all of it's attributes and methods
- Every function details the type and size of it's inputs and outputs
- **Names say what things are.** `root_chord`, not `cr`; `semispan`, not `b2`.
  Standard aerospace symbols are fine inside a short


## Commands

```bash
pip install -e ".[dev]"            # install in editable mode with test tools
pytest                             # run all tests
python examples/reference_aircraft.py
```

Run `pytest` before every commit. All tests must pass.

## Conventions

- **Units:** SI throughout (m, m², kg, s, W). Angles are degrees at user-facing
  inputs and outputs, radians internally. Put units in names or docstrings when
  not obvious, e.g. `sweep_le_deg`, `area_m2`.
- **Axes:** body axes are x aft, y to starboard, z up. Each surface has its own
  local (x, s) plane placed by `origin` and `orientation`.
- **Single source of truth:** store only the defining inputs (stations). Compute
  derived quantities (area, span, MAC, ...) as properties on demand; never cache
  or duplicate them.
- **Symmetric surfaces:** stations describe one side; `span` and `area` cover
  both sides.






