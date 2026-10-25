"""2D planform geometry for lifting surfaces (wings, tailplanes and fins).

How a surface is described
--------------------------
A lifting surface is a list of spanwise *stations*. Each station records three
numbers: where it is along the span, where its leading edge is, and its chord.
Between two neighbouring stations the planform is a straight-tapered
(trapezoidal) *panel*, so a surface with N stations has N - 1 panels.

Each surface lives in its own flat 2D plane with two coordinates:

* ``x``: chordwise, pointing aft (from leading edge towards trailing edge).
* span position: along the span, pointing outboard for a wing or tailplane and
  upward for a fin.

The surface is placed on the aircraft by two settings:

* ``origin``: the body-axis point (x, y, z) where the surface's local
  ``x = 0``, span position ``= 0`` sits.
* ``orientation``: ``"horizontal"`` lays the span along body y (wings and
  tailplanes); ``"vertical"`` stands it along body z (fins).

Body axes are x aft, y to starboard, z up.

Only the stations are stored. Area, span, aspect ratio, mean aerodynamic chord
and so on are computed from them each time they are asked for, so they can never
disagree with the geometry.

Reference for the planform definitions used here (mean aerodynamic chord and its
position, aspect ratio, taper ratio): D. P. Raymer, *Aircraft Design: A
Conceptual Approach*, AIAA, wing geometry section.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

Orientation = Literal["horizontal", "vertical"]

#: Chord fraction of the quarter-chord line. The aerodynamic centre of a
#: subsonic wing section sits close to this point.
QUARTER_CHORD = 0.25

#: Keys accepted in each panel dict passed to ``LiftingSurface.from_panels``.
PANEL_KEYS = {"span", "taper", "sweep_le_deg"}


@dataclass(frozen=True)
class Station:
    """One spanwise station of a lifting surface.

    Parameters
    ----------
    span_position : float
        Position along the span [m], in the surface's own plane.
    x_leading_edge : float
        Chordwise position of the leading edge [m], positive aft.
    chord : float
        Local chord [m]. Must be positive.
    """

    span_position: float
    x_leading_edge: float
    chord: float

    def __post_init__(self):
        if not np.isfinite([self.span_position, self.x_leading_edge, self.chord]).all():
            raise ValueError(f"Station values must be finite numbers, got {self}")
        if self.chord <= 0:
            raise ValueError(f"Station chord must be positive, got chord = {self.chord} m")

    @property
    def x_trailing_edge(self) -> float:
        """Chordwise position of the trailing edge [m]."""
        return self.x_leading_edge + self.chord


def _integrate_product(span_positions: np.ndarray, f: np.ndarray, g: np.ndarray) -> float:
    """Integrate f(s) * g(s) along the span, from the first station to the last.

    ``f`` and ``g`` are values at the stations, and both are assumed to vary
    linearly between stations (true for chord, leading-edge position and span
    position on a straight-tapered panel). For two linear functions on a
    panel of width ds, the integral of their product is exactly

        ds / 6 * (2 f0 g0 + f0 g1 + f1 g0 + 2 f1 g1)

    where 0 and 1 denote the inner and outer ends of the panel.
    """
    ds = np.diff(span_positions)
    f0, f1 = f[:-1], f[1:]
    g0, g1 = g[:-1], g[1:]
    return float(np.sum(ds / 6 * (2 * f0 * g0 + f0 * g1 + f1 * g0 + 2 * f1 * g1)))


@dataclass
class LiftingSurface:
    """A 2D lifting surface (wing, tailplane or fin) made of trapezoidal panels.

    Parameters
    ----------
    name : str
        Identifier, e.g. ``"wing"``. Must be unique within an aircraft.
    stations : sequence of Station
        At least two stations, ordered root to tip with strictly increasing
        span position.
    symmetric : bool
        If True, the stations describe one side and the surface is mirrored to
        the other side. Wings and tailplanes are normally symmetric; a single
        centreline fin is not.
    origin : (x, y, z)
        Body-axis position [m] of the surface's local origin.
    orientation : "horizontal" or "vertical"
        Whether the span lies along body y (horizontal) or body z (vertical).
    """

    name: str
    stations: Sequence[Station]
    symmetric: bool = True
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: Orientation = "horizontal"

    def __post_init__(self):
        self.stations = tuple(self.stations)
        self.origin = tuple(float(value) for value in self.origin)
        self._check_inputs()

    def _check_inputs(self) -> None:
        if len(self.stations) < 2:
            raise ValueError(
                f"Surface '{self.name}': needs at least two stations, got {len(self.stations)}"
            )
        if np.any(np.diff(self.span_positions) <= 0):
            raise ValueError(
                f"Surface '{self.name}': station span positions must strictly increase "
                f"from root to tip, got {self.span_positions.tolist()}"
            )
        if self.symmetric and self.span_positions[0] < 0:
            raise ValueError(
                f"Surface '{self.name}': a symmetric surface describes one side only, so its "
                f"first station must have span position >= 0, got {self.span_positions[0]}"
            )
        if self.orientation not in ("horizontal", "vertical"):
            raise ValueError(
                f"Surface '{self.name}': orientation must be 'horizontal' or 'vertical', "
                f"got '{self.orientation}'"
            )
        if len(self.origin) != 3:
            raise ValueError(
                f"Surface '{self.name}': origin must be (x, y, z), got {self.origin}"
            )

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
        symmetric: bool = True,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        orientation: Orientation = "horizontal",
    ) -> LiftingSurface:
        """Make a single-panel, straight-tapered surface.

        Parameters
        ----------
        name : str
            Identifier, e.g. ``"htail"``.
        span : float
            Tip-to-tip span [m] if symmetric; otherwise root-to-tip length
            (for a fin, its height).
        root_chord : float
            Root chord [m].
        taper : float
            Taper ratio: tip chord / root chord.
        sweep_le_deg : float
            Leading-edge sweep angle [deg], positive aft.
        symmetric, origin, orientation
            As for :class:`LiftingSurface`.
        """
        semispan = span / 2 if symmetric else span
        root = Station(span_position=0.0, x_leading_edge=0.0, chord=root_chord)
        tip = Station(
            span_position=semispan,
            x_leading_edge=semispan * np.tan(np.radians(sweep_le_deg)),
            chord=root_chord * taper,
        )
        return cls(
            name=name,
            stations=[root, tip],
            symmetric=symmetric,
            origin=origin,
            orientation=orientation,
        )

    @classmethod
    def from_panels(
        cls,
        name: str,
        root_chord: float,
        panels: Sequence[dict],
        symmetric: bool = True,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        orientation: Orientation = "horizontal",
    ) -> LiftingSurface:
        """Build a multi-panel surface one panel at a time, from root to tip.

        Parameters
        ----------
        name : str
            Identifier, e.g. ``"wing"``.
        root_chord : float
            Chord at the root [m].
        panels : sequence of dict
            One dict per panel, root first. Keys:

            * ``span`` (required): length of this panel along the span [m],
              for one side.
            * ``taper`` (default 1.0): outer chord / inner chord of this panel.
            * ``sweep_le_deg`` (default 0.0): leading-edge sweep of this
              panel [deg], positive aft.
        symmetric, origin, orientation
            As for :class:`LiftingSurface`.

        Example
        -------
        Rectangular centre section with tapered, slightly swept outer panels::

            LiftingSurface.from_panels(
                "wing",
                root_chord=0.3,
                panels=[
                    {"span": 1.0},
                    {"span": 0.8, "taper": 0.6, "sweep_le_deg": 5.0},
                ],
            )
        """
        stations = [Station(span_position=0.0, x_leading_edge=0.0, chord=root_chord)]

        for panel_number, panel in enumerate(panels, start=1):
            unknown_keys = set(panel) - PANEL_KEYS
            if unknown_keys:
                raise ValueError(
                    f"Surface '{name}', panel {panel_number}: unknown key(s) "
                    f"{sorted(unknown_keys)}. Allowed keys are {sorted(PANEL_KEYS)}."
                )
            if "span" not in panel:
                raise ValueError(f"Surface '{name}', panel {panel_number}: 'span' is required")

            inner = stations[-1]
            panel_span = panel["span"]
            sweep_rad = np.radians(panel.get("sweep_le_deg", 0.0))
            outer = Station(
                span_position=inner.span_position + panel_span,
                x_leading_edge=inner.x_leading_edge + panel_span * np.tan(sweep_rad),
                chord=inner.chord * panel.get("taper", 1.0),
            )
            stations.append(outer)

        return cls(
            name=name,
            stations=stations,
            symmetric=symmetric,
            origin=origin,
            orientation=orientation,
        )

    # ------------------------------------------------------------------ #
    # Station values as arrays (one side, local coordinates)
    # ------------------------------------------------------------------ #
    @property
    def span_positions(self) -> np.ndarray:
        """Span position of each station [m]."""
        return np.array([station.span_position for station in self.stations])

    @property
    def x_leading_edges(self) -> np.ndarray:
        """Leading-edge x of each station [m]."""
        return np.array([station.x_leading_edge for station in self.stations])

    @property
    def chords(self) -> np.ndarray:
        """Chord of each station [m]."""
        return np.array([station.chord for station in self.stations])

    @property
    def x_trailing_edges(self) -> np.ndarray:
        """Trailing-edge x of each station [m]."""
        return self.x_leading_edges + self.chords

    @property
    def n_panels(self) -> int:
        """Number of trapezoidal panels on one side."""
        return len(self.stations) - 1

    def chord_at(self, span_position: float | np.ndarray) -> np.ndarray:
        """Chord [m] at any span position(s), interpolated between stations.

        For a symmetric surface, negative span positions (the mirrored side)
        give the same chord as the matching positive ones.
        """
        if self.symmetric:
            span_position = np.abs(span_position)
        return np.interp(span_position, self.span_positions, self.chords)

    def x_leading_edge_at(self, span_position: float | np.ndarray) -> np.ndarray:
        """Leading-edge x [m] at any span position(s), interpolated between stations."""
        if self.symmetric:
            span_position = np.abs(span_position)
        return np.interp(span_position, self.span_positions, self.x_leading_edges)

    # ------------------------------------------------------------------ #
    # Planform properties
    # ------------------------------------------------------------------ #
    @property
    def _one_side_area(self) -> float:
        """Area of one side [m^2]: integral of chord along the span."""
        ones = np.ones(len(self.stations))
        return _integrate_product(self.span_positions, self.chords, ones)

    @property
    def area(self) -> float:
        """Planform area [m^2], both sides if symmetric."""
        return 2 * self._one_side_area if self.symmetric else self._one_side_area

    @property
    def span(self) -> float:
        """Span [m]: tip to tip if symmetric, otherwise root to tip."""
        if self.symmetric:
            return 2 * self.span_positions[-1]
        return self.span_positions[-1] - self.span_positions[0]

    @property
    def aspect_ratio(self) -> float:
        """Aspect ratio, span^2 / area.

        For a fin this is the plain geometric value, with no correction for the
        end-plate effect of the fuselage or tailplane.
        """
        return self.span**2 / self.area

    @property
    def taper_ratio(self) -> float:
        """Taper ratio: tip chord / root chord."""
        return self.chords[-1] / self.chords[0]

    @property
    def mac(self) -> float:
        """Mean aerodynamic chord [m].

        MAC = (integral of chord^2 along the span) / (integral of chord along the span),
        both taken over one side.
        """
        integral_chord_squared = _integrate_product(self.span_positions, self.chords, self.chords)
        return integral_chord_squared / self._one_side_area

    @property
    def mac_span_position(self) -> float:
        """Span position of the MAC [m], one side, local coordinates.

        This is the area-weighted mean span position (the centroid of one side).
        """
        integral = _integrate_product(self.span_positions, self.chords, self.span_positions)
        return integral / self._one_side_area

    @property
    def mac_x_leading_edge(self) -> float:
        """Leading-edge x of the MAC [m], local coordinates."""
        integral = _integrate_product(self.span_positions, self.chords, self.x_leading_edges)
        return integral / self._one_side_area

    @property
    def x_aerodynamic_centre(self) -> float:
        """Approximate aerodynamic centre x [m], local coordinates.

        Taken as the quarter-chord point of the MAC, a standard estimate for
        unswept or moderately swept wings in subsonic flow.
        """
        return self.mac_x_leading_edge + QUARTER_CHORD * self.mac

    @property
    def x_aerodynamic_centre_body(self) -> float:
        """Approximate aerodynamic centre x [m], body axes."""
        return self.origin[0] + self.x_aerodynamic_centre

    def sweep_deg(self, chord_fraction: float = QUARTER_CHORD) -> np.ndarray:
        """Sweep angle [deg] of each panel along a chord-fraction line.

        Parameters
        ----------
        chord_fraction : float
            0 for the leading edge, 0.25 for the quarter-chord line (default),
            1 for the trailing edge.
        """
        x_line = self.x_leading_edges + chord_fraction * self.chords
        sweep_rad = np.arctan2(np.diff(x_line), np.diff(self.span_positions))
        return np.degrees(sweep_rad)

    def panel_areas(self) -> np.ndarray:
        """Area of each panel on one side [m^2]."""
        panel_widths = np.diff(self.span_positions)
        mean_chords = (self.chords[:-1] + self.chords[1:]) / 2
        return panel_widths * mean_chords

    # ------------------------------------------------------------------ #
    # Outline and placement on the aircraft
    # ------------------------------------------------------------------ #
    def outline(self) -> tuple[np.ndarray, np.ndarray]:
        """Closed planform outline in local coordinates.

        Returns
        -------
        x, span_position : ndarray
            Points around the outline, ending where they started. For a
            symmetric surface the outline covers both sides. If a symmetric
            surface does not reach the centreline (first station off-centre),
            its two halves are separate shapes; they are returned joined by a
            NaN, which matplotlib draws as a gap.
        """
        # One side: out along the leading edge, back along the trailing edge, close.
        x_one_side = np.concatenate(
            [self.x_leading_edges, self.x_trailing_edges[::-1], self.x_leading_edges[:1]]
        )
        s_one_side = np.concatenate(
            [self.span_positions, self.span_positions[::-1], self.span_positions[:1]]
        )

        if not self.symmetric:
            return x_one_side, s_one_side

        if self.span_positions[0] > 0:
            # Two separate halves, mirror images of each other.
            x_both = np.concatenate([x_one_side, [np.nan], x_one_side])
            s_both = np.concatenate([s_one_side, [np.nan], -s_one_side])
            return x_both, s_both

        # Halves meet on the centreline: trace one continuous loop.
        # Leading edge from left tip to right tip...
        x_le_left_to_right = np.concatenate([self.x_leading_edges[::-1], self.x_leading_edges[1:]])
        s_le_left_to_right = np.concatenate([-self.span_positions[::-1], self.span_positions[1:]])
        # ...then trailing edge from right tip back to left tip...
        x_te_right_to_left = np.concatenate([self.x_trailing_edges[::-1], self.x_trailing_edges[1:]])
        s_te_right_to_left = np.concatenate([self.span_positions[::-1], -self.span_positions[1:]])
        # ...then close at the left-tip leading edge.
        x_loop = np.concatenate([x_le_left_to_right, x_te_right_to_left, x_le_left_to_right[:1]])
        s_loop = np.concatenate([s_le_left_to_right, s_te_right_to_left, s_le_left_to_right[:1]])
        return x_loop, s_loop

    def to_body(self, x_local: np.ndarray, span_position: np.ndarray) -> np.ndarray:
        """Convert local (x, span position) points to body axes.

        Returns
        -------
        ndarray, shape (n, 3)
            Body-axis (x, y, z) of each point [m].
        """
        x0, y0, z0 = self.origin
        x_local = np.asarray(x_local, dtype=float)
        span_position = np.asarray(span_position, dtype=float)

        x_body = x0 + x_local
        if self.orientation == "horizontal":
            y_body = y0 + span_position
            z_body = np.full_like(x_local, z0)
        else:
            y_body = np.full_like(x_local, y0)
            z_body = z0 + span_position
        return np.column_stack([x_body, y_body, z_body])

    def outline_body(self) -> np.ndarray:
        """Closed planform outline in body axes, shape (n, 3)."""
        return self.to_body(*self.outline())

    # ------------------------------------------------------------------ #
    def summary(self) -> dict:
        """Main planform properties as a dict, with units in the key names."""
        return {
            "name": self.name,
            "area_m2": self.area,
            "span_m": self.span,
            "aspect_ratio": self.aspect_ratio,
            "taper_ratio": self.taper_ratio,
            "mac_m": self.mac,
            "x_aerodynamic_centre_body_m": self.x_aerodynamic_centre_body,
            "sweep_quarter_chord_deg": self.sweep_deg(QUARTER_CHORD).tolist(),
        }
