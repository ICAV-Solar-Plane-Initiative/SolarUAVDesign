"""Aircraft: a named collection of 2D lifting surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field

from .surface import LiftingSurface


@dataclass
class Aircraft:
    """A collection of lifting surfaces.

    Parameters
    ----------
    name : str
        Aircraft name.
    surfaces : list of LiftingSurface
        Surfaces making up the aircraft. Names must be unique.
    reference_surface : str
        Name of the surface used for reference area, span and chord
        (normally the main wing). Defaults to the first surface.
    """

    name: str
    surfaces: list[LiftingSurface] = field(default_factory=list)
    reference_surface: str | None = None

    def __post_init__(self):
        names = [s.name for s in self.surfaces]
        if len(names) != len(set(names)):
            raise ValueError(f"Surface names must be unique, got {names}")
        if self.reference_surface is None and self.surfaces:
            self.reference_surface = self.surfaces[0].name
        if self.reference_surface is not None and self.reference_surface not in names:
            raise ValueError(f"Reference surface '{self.reference_surface}' not in {names}")

    def __getitem__(self, name: str) -> LiftingSurface:
        for s in self.surfaces:
            if s.name == name:
                return s
        raise KeyError(name)

    def add(self, surface: LiftingSurface) -> "Aircraft":
        """Add a surface and return self, so calls can be chained."""
        if any(s.name == surface.name for s in self.surfaces):
            raise ValueError(f"A surface named '{surface.name}' already exists")
        self.surfaces.append(surface)
        if self.reference_surface is None:
            self.reference_surface = surface.name
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

    def plot(self, **kwargs):
        """Plot the aircraft. See :func:`solaruav.plotting.plot_aircraft`."""
        from ..plotting import plot_aircraft

        return plot_aircraft(self, **kwargs)
