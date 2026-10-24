"""2D planform geometry for lifting surfaces.

A lifting surface is described by a list of spanwise *stations*. Each station
gives the spanwise position, the leading-edge x position and the chord. The
planform between two stations is a straight-tapered (trapezoidal) panel, so a
surface with N stations has N - 1 panels.

Coordinates are in the surface's own 2D plane:

* ``x`` points aft (chordwise, leading edge to trailing edge).
* ``s`` is the spanwise coordinate (outboard for a wing or tailplane, upward
  for a fin).

The surface is placed on the aircraft by ``origin`` (the body-axis position of
the point ``x = 0, s = 0``) and ``orientation`` ("horizontal" maps ``s`` to the
body y-axis, "vertical" maps it to the body z-axis). Body axes are x aft,
y to starboard, z up.

All derived properties (area, span, aspect ratio, mean aerodynamic chord...)
are computed from the stations on demand and never stored, so they cannot fall
out of step with the geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

Orientation = Literal["horizontal", "vertical"]


@dataclass(frozen=True)
class Station:
    """A spanwise station of a lifting surface.

    Parameters
    ----------
    s : float
        Spanwise position [m], measured in the surface's own plane.
    x_le : float
        Leading-edge x position [m] (positive aft).
    chord : float
        Local chord [m]. Must be positive.
    """

    s: float
    x_le: float
    chord: float

    def __post_init__(self):
        if not self.chord > 0:
            raise ValueError(f"Station chord must be positive, got {self.chord}")

    @property
    def x_te(self) -> float:
        """Trailing-edge x position [m]."""
        return self.x_le + self.chord


@dataclass
class LiftingSurface:
    """A 2D lifting surface (wing, tailplane or fin) made of trapezoidal panels.

    Parameters
    ----------
    name : str
        Identifier, e.g. ``"wing"``.
    stations : sequence of Station
        At least two stations, with strictly increasing ``s``.
    symmetric : bool
        If True, the surface is mirrored about its origin plane (``s -> -s``).
        Wings and tailplanes are normally symmetric; fins are not.
    origin : (x, y, z)
        Body-axis position [m] of the surface's local point ``x = 0, s = 0``.
    orientation : "horizontal" or "vertical"
        Whether the spanwise coordinate maps to body y or body z.
    """

    name: str
    stations: Sequence[Station]
    symmetric: bool = True
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: Orientation = "horizontal"
    _arrays: dict = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        self.stations = tuple(self.stations)
        if len(self.stations) < 2:
            raise ValueError(f"{self.name}: a lifting surface needs at least two stations")
        s = np.array([st.s for st in self.stations])
        if np.any(np.diff(s) <= 0):
            raise ValueError(f"{self.name}: station s positions must be strictly increasing")
        if self.symmetric and s[0] < 0:
            raise ValueError(f"{self.name}: symmetric surfaces must start at s >= 0")
        if self.orientation not in ("horizontal", "vertical"):
            raise ValueError(f"{self.name}: orientation must be 'horizontal' or 'vertical'")
        self.origin = tuple(float(v) for v in self.origin)

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def trapezoidal(
        cls,
        name: str,
        span: float,
        root_chord: float,
        taper: float = 1.0,
        sweep_le_deg: float = 0.0,
        **kwargs,
    ) -> "LiftingSurface":
        """Single-panel straight-tapered surface.

        Parameters
        ----------
        span : float
            Tip-to-tip span [m] if symmetric, otherwise root-to-tip (e.g. fin
            height).
        root_chord : float
            Root chord [m].
        taper : float
            Tip chord / root chord.
        sweep_le_deg : float
            Leading-edge sweep angle [deg], positive aft.
        **kwargs
            Passed to the constructor (``symmetric``, ``origin``, ``orientation``).
        """
        symmetric = kwargs.get("symmetric", True)
        semispan = span / 2 if symmetric else span
        return cls(
            name=name,
            stations=[
                Station(s=0.0, x_le=0.0, chord=root_chord),
                Station(
                    s=semispan,
                    x_le=semispan * np.tan(np.radians(sweep_le_deg)),
                    chord=root_chord * taper,
                ),
            ],
            **kwargs,
        )

    @classmethod
    def from_panels(
        cls,
        name: str,
        root_chord: float,
        panels: Sequence[dict],
        **kwargs,
    ) -> "LiftingSurface":
        """Build a multi-panel surface panel by panel, from the root outward.

        Each panel is a dict with keys ``span`` (spanwise length of this panel,
        for one side), ``taper`` (outer chord / inner chord of this panel) and
        optionally ``sweep_le_deg``.

        Example
        -------
        Rectangular centre section and tapered outer panels::

            LiftingSurface.from_panels(
                "wing", root_chord=0.3,
                panels=[{"span": 1.0, "taper": 1.0},
                        {"span": 0.8, "taper": 0.6, "sweep_le_deg": 5}],
            )
        """
        stations = [Station(s=0.0, x_le=0.0, chord=root_chord)]
        for p in panels:
            inner = stations[-1]
            ds = p["span"]
            stations.append(
                Station(
                    s=inner.s + ds,
                    x_le=inner.x_le + ds * np.tan(np.radians(p.get("sweep_le_deg", 0.0))),
                    chord=inner.chord * p.get("taper", 1.0),
                )
            )
        return cls(name=name, stations=stations, **kwargs)

    # ------------------------------------------------------------------ #
    # Station arrays
    # ------------------------------------------------------------------ #
    @property
    def s(self) -> np.ndarray:
        return np.array([st.s for st in self.stations])

    @property
    def x_le(self) -> np.ndarray:
        return np.array([st.x_le for st in self.stations])

    @property
    def chord(self) -> np.ndarray:
        return np.array([st.chord for st in self.stations])

    @property
    def x_te(self) -> np.ndarray:
        return self.x_le + self.chord

    @property
    def n_panels(self) -> int:
        return len(self.stations) - 1

    def chord_at(self, s: float | np.ndarray) -> np.ndarray:
        """Chord [m] at spanwise position(s) ``s`` (one side, local coordinates)."""
        return np.interp(np.abs(s) if self.symmetric else s, self.s, self.chord)

    def x_le_at(self, s: float | np.ndarray) -> np.ndarray:
        """Leading-edge x [m] at spanwise position(s) ``s``."""
        return np.interp(np.abs(s) if self.symmetric else s, self.s, self.x_le)

    # ------------------------------------------------------------------ #
    # Integrals over one side. c and x_le are linear within each panel, so
    # Simpson's rule per panel is exact for every integrand used here
    # (all are at most quadratic in s).
    # ------------------------------------------------------------------ #
    def _integrate(self, f_in: np.ndarray, f_mid: np.ndarray, f_out: np.ndarray) -> float:
        ds = np.diff(self.s)
        return float(np.sum(ds / 6 * (f_in + 4 * f_mid + f_out)))

    def _panel_values(self, attr: str):
        v = getattr(self, attr)
        return v[:-1], (v[:-1] + v[1:]) / 2, v[1:]

    def _side_integral(self, kind: str) -> float:
        c0, cm, c1 = self._panel_values("chord")
        if kind == "c":
            return self._integrate(c0, cm, c1)
        if kind == "c2":
            return self._integrate(c0**2, cm**2, c1**2)
        if kind == "cs":
            s0, sm, s1 = self._panel_values("s")
            return self._integrate(c0 * s0, cm * sm, c1 * s1)
        if kind == "cx":
            x0, xm, x1 = self._panel_values("x_le")
            return self._integrate(c0 * x0, cm * xm, c1 * x1)
        raise ValueError(kind)

    # ------------------------------------------------------------------ #
    # Derived planform properties
    # ------------------------------------------------------------------ #
    @property
    def area(self) -> float:
        """Planform area [m^2], both sides if symmetric."""
        a = self._side_integral("c")
        return 2 * a if self.symmetric else a

    @property
    def span(self) -> float:
        """Tip-to-tip span [m] if symmetric, otherwise root-to-tip extent."""
        return 2 * self.s[-1] if self.symmetric else self.s[-1] - self.s[0]

    @property
    def aspect_ratio(self) -> float:
        """Aspect ratio, span^2 / area. For a fin this is the geometric
        (single-sided) value, without any end-plate correction."""
        return self.span**2 / self.area

    @property
    def taper_ratio(self) -> float:
        """Tip chord / root chord."""
        return self.chord[-1] / self.chord[0]

    @property
    def mac(self) -> float:
        """Mean aerodynamic chord [m]: (1/S) * integral of c^2 ds."""
        return self._side_integral("c2") / self._side_integral("c")

    @property
    def s_mac(self) -> float:
        """Spanwise position [m] of the MAC, local coordinates (one side)."""
        return self._side_integral("cs") / self._side_integral("c")

    @property
    def x_le_mac(self) -> float:
        """Leading-edge x [m] of the MAC, local coordinates."""
        return self._side_integral("cx") / self._side_integral("c")

    @property
    def x_ac(self) -> float:
        """Approximate aerodynamic centre x [m] (quarter-MAC), local coordinates."""
        return self.x_le_mac + 0.25 * self.mac

    def sweep_deg(self, chord_fraction: float = 0.25) -> np.ndarray:
        """Sweep angle [deg] of each panel along the given chord-fraction line."""
        x = self.x_le + chord_fraction * self.chord
        return np.degrees(np.arctan2(np.diff(x), np.diff(self.s)))

    def panel_areas(self) -> np.ndarray:
        """Area [m^2] of each panel, one side."""
        return np.diff(self.s) * (self.chord[:-1] + self.chord[1:]) / 2

    # ------------------------------------------------------------------ #
    # Outline and placement on the aircraft
    # ------------------------------------------------------------------ #
    def outline(self) -> tuple[np.ndarray, np.ndarray]:
        """Closed planform outline in local (x, s) coordinates.

        For symmetric surfaces the outline covers both sides, running round
        the leading edge from the left tip to the right tip and back along the
        trailing edge. A symmetric surface whose first station is off the
        centreline returns two separate outlines, joined with a NaN so they
        plot as two shapes.
        """
        x_right = np.concatenate([self.x_le, self.x_te[::-1], self.x_le[:1]])
        s_right = np.concatenate([self.s, self.s[::-1], self.s[:1]])
        if not self.symmetric:
            return x_right, s_right
        if self.s[0] == 0:
            x = np.concatenate([self.x_le[::-1], self.x_le[1:], self.x_te[::-1], self.x_te[1:], self.x_le[-1:]])
            s = np.concatenate([-self.s[::-1], self.s[1:], self.s[::-1], -self.s[1:], -self.s[-1:]])
            return x, s
        return (
            np.concatenate([x_right, [np.nan], x_right]),
            np.concatenate([s_right, [np.nan], -s_right]),
        )

    def to_body(self, x_local: np.ndarray, s_local: np.ndarray) -> np.ndarray:
        """Map local (x, s) points to body axes. Returns an (n, 3) array."""
        x0, y0, z0 = self.origin
        x_local = np.asarray(x_local, dtype=float)
        s_local = np.asarray(s_local, dtype=float)
        if self.orientation == "horizontal":
            return np.column_stack([x0 + x_local, y0 + s_local, np.full_like(x_local, z0)])
        return np.column_stack([x0 + x_local, np.full_like(x_local, y0), z0 + s_local])

    def outline_body(self) -> np.ndarray:
        """Closed planform outline in body axes, (n, 3)."""
        return self.to_body(*self.outline())

    @property
    def x_ac_body(self) -> float:
        """Approximate aerodynamic centre x [m] in body axes."""
        return self.origin[0] + self.x_ac

    # ------------------------------------------------------------------ #
    def summary(self) -> dict:
        """Key planform properties as a dict."""
        return {
            "name": self.name,
            "area_m2": self.area,
            "span_m": self.span,
            "aspect_ratio": self.aspect_ratio,
            "taper_ratio": self.taper_ratio,
            "mac_m": self.mac,
            "x_ac_body_m": self.x_ac_body,
            "sweep_c4_deg": self.sweep_deg(0.25).tolist(),
        }
