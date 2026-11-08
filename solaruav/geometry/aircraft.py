"""Aircraft: a named collection of lifting surfaces and fuselages.

Purpose
-------
Group the lifting surfaces and fuselages of one aircraft, pick the reference
surface, and give aircraft-level reference quantities and tail volume
coefficients.

Inputs
------
- Aircraft name (str).
- List of ``LiftingSurface`` objects and list of ``Fuselage`` objects; every
  name must be unique across both.
- Name of the reference surface (normally the main wing).

Outputs
-------
- Reference area [m^2], span [m] and mean aerodynamic chord [m].
- Horizontal and vertical tail volume coefficients (floats).
- Summary tables of the surfaces and of the fuselages (lists of dicts) and plots.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .fuselage import Fuselage
from .surface import LiftingSurface


@dataclass
class Aircraft:
    """A collection of lifting surfaces and fuselages.

    Parameters
    ----------
    name : str
        Aircraft name.
    surfaces : list of LiftingSurface
        Surfaces making up the aircraft.
    reference_surface : str
        Name of the surface used for reference area, span and chord
        (normally the main wing). Defaults to the first surface.
    fuselages : list of Fuselage
        Fuselages, booms and pods. A mirrored ``Fuselage`` counts as one entry
        but describes two bodies (e.g. a twin-fuselage layout).

    Names must be unique across surfaces and fuselages.

    Methods
    -------
    aircraft["name"]
        The surface or fuselage with that name.
    add(component)
        Add a LiftingSurface or Fuselage; returns self for chaining.
    summary(), fuselage_summary()
        Lists of dicts, one per surface / per fuselage.
    plot(**kwargs)
        Three-view plot, see ``solaruav.plotting.plot_aircraft``.
    """

    name: str
    surfaces: list[LiftingSurface] = field(default_factory=list)
    reference_surface: str | None = None
    fuselages: list[Fuselage] = field(default_factory=list)

    def __post_init__(self):
        names = self._component_names()
        if len(names) != len(set(names)):
            raise ValueError(f"Surface and fuselage names must be unique, got {names}")
        surface_names = [s.name for s in self.surfaces]
        if self.reference_surface is None and self.surfaces:
            self.reference_surface = self.surfaces[0].name
        if self.reference_surface is not None and self.reference_surface not in surface_names:
            raise ValueError(f"Reference surface '{self.reference_surface}' not in {surface_names}")

    def _component_names(self) -> list[str]:
        return [s.name for s in self.surfaces] + [f.name for f in self.fuselages]

    def __getitem__(self, name: str) -> LiftingSurface | Fuselage:
        for component in [*self.surfaces, *self.fuselages]:
            if component.name == name:
                return component
        raise KeyError(name)

    def add(self, component: LiftingSurface | Fuselage) -> "Aircraft":
        """Add a surface or fuselage and return self, so calls can be chained."""
        if component.name in self._component_names():
            raise ValueError(f"A component named '{component.name}' already exists")
        if isinstance(component, Fuselage):
            self.fuselages.append(component)
            return self
        self.surfaces.append(component)
        if self.reference_surface is None:
            self.reference_surface = component.name
        return self

    @property
    def reference(self) -> LiftingSurface:
        return self[self.reference_surface]

    @property
    def s_ref(self) -> float:
        """Reference area [m^2]."""
        return self.reference.area

    @property
    def b_ref(self) -> float:
        """Reference span [m]."""
        return self.reference.span

    @property
    def c_ref(self) -> float:
        """Reference chord [m] (MAC of the reference surface)."""
        return self.reference.mac

    def horizontal_tail_volume(self, tail: str) -> float:
        """Horizontal tail volume coefficient V_H = S_t * l_t / (S_ref * c_ref),
        with l_t measured between quarter-MAC points."""
        t = self[tail]
        l_t = t.x_ac_body - self.reference.x_ac_body
        return t.area * l_t / (self.s_ref * self.c_ref)

    def vertical_tail_volume(self, fin: str) -> float:
        """Vertical tail volume coefficient V_V = S_v * l_v / (S_ref * b_ref)."""
        f = self[fin]
        l_v = f.x_ac_body - self.reference.x_ac_body
        return f.area * l_v / (self.s_ref * self.b_ref)

    def summary(self) -> list[dict]:
        """Planform summary of every surface."""
        return [s.summary() for s in self.surfaces]

    def fuselage_summary(self) -> list[dict]:
        """Geometry summary of every fuselage (per body, with the body count)."""
        return [f.summary() for f in self.fuselages]

    def plot(self, **kwargs):
        """Plot the aircraft. See :func:`solaruav.plotting.plot_aircraft`."""
        from ..plotting import plot_aircraft

        return plot_aircraft(self, **kwargs)
