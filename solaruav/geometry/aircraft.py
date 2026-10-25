"""Aircraft: a named collection of 2D lifting surfaces.

Tail volume coefficients follow the usual conceptual-design definitions (see
D. P. Raymer, *Aircraft Design: A Conceptual Approach*, AIAA, tail sizing
section). Tail moment arms are measured between aerodynamic centres
(quarter-MAC points), which is a common variant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .surface import LiftingSurface


@dataclass
class Aircraft:
    """A collection of lifting surfaces.

    Parameters
    ----------
    name : str
        Aircraft name, used as the plot title.
    surfaces : list of LiftingSurface
        Surfaces making up the aircraft. Names must be unique.
    reference_surface : str, optional
        Name of the surface that sets the reference area, span and chord
        (normally the main wing). Defaults to the first surface.
    """

    name: str
    surfaces: list[LiftingSurface] = field(default_factory=list)
    reference_surface: str | None = None

    def __post_init__(self):
        names = [surface.name for surface in self.surfaces]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Surface names must be unique; repeated: {duplicates}")

        if self.reference_surface is None and self.surfaces:
            self.reference_surface = self.surfaces[0].name
        if self.reference_surface is not None and self.reference_surface not in names:
            raise ValueError(
                f"Reference surface '{self.reference_surface}' is not one of the surfaces {names}"
            )

    def __getitem__(self, name: str) -> LiftingSurface:
        """Look up a surface by name, e.g. ``aircraft["wing"]``."""
        for surface in self.surfaces:
            if surface.name == name:
                return surface
        names = [surface.name for surface in self.surfaces]
        raise KeyError(f"No surface named '{name}'. Surfaces are {names}")

    def add(self, surface: LiftingSurface) -> Aircraft:
        """Add a surface. Returns the aircraft, so calls can be chained."""
        if any(existing.name == surface.name for existing in self.surfaces):
            raise ValueError(f"A surface named '{surface.name}' already exists")
        self.surfaces.append(surface)
        if self.reference_surface is None:
            self.reference_surface = surface.name
        return self

    # ------------------------------------------------------------------ #
    # Reference quantities
    # ------------------------------------------------------------------ #
    @property
    def reference(self) -> LiftingSurface:
        """The reference surface (normally the main wing)."""
        return self[self.reference_surface]

    @property
    def reference_area(self) -> float:
        """Reference area [m^2]: area of the reference surface."""
        return self.reference.area

    @property
    def reference_span(self) -> float:
        """Reference span [m]: span of the reference surface."""
        return self.reference.span

    @property
    def reference_chord(self) -> float:
        """Reference chord [m]: MAC of the reference surface."""
        return self.reference.mac

    def _tail_arm(self, surface_name: str) -> float:
        """Distance in x [m] from the reference surface's aerodynamic centre to
        the named surface's aerodynamic centre."""
        tail = self[surface_name]
        return tail.x_aerodynamic_centre_body - self.reference.x_aerodynamic_centre_body

    def horizontal_tail_volume(self, tail_name: str) -> float:
        """Horizontal tail volume coefficient.

        V_H = S_tail * l_tail / (S_ref * c_ref)

        where S is area, l_tail is the tail arm and c_ref is the reference chord.
        Raymer's typical value for sailplanes is about 0.5.
        """
        tail = self[tail_name]
        return tail.area * self._tail_arm(tail_name) / (self.reference_area * self.reference_chord)

    def vertical_tail_volume(self, fin_name: str) -> float:
        """Vertical tail volume coefficient.

        V_V = S_fin * l_fin / (S_ref * b_ref)

        where b_ref is the reference span.
        Raymer's typical value for sailplanes is about 0.02.
        """
        fin = self[fin_name]
        return fin.area * self._tail_arm(fin_name) / (self.reference_area * self.reference_span)

    # ------------------------------------------------------------------ #
    def summary(self) -> list[dict]:
        """Planform summary of every surface, one dict per surface."""
        return [surface.summary() for surface in self.surfaces]

    def plot(self, **kwargs):
        """Plot the aircraft. Takes the same options as
        :func:`solaruav.plotting.plot_aircraft`."""
        from ..plotting import plot_aircraft  # imported here to avoid a circular import

        return plot_aircraft(self, **kwargs)
