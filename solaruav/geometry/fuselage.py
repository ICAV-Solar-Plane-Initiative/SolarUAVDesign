"""3D geometry of fuselages, booms and pods.

A fuselage is described by a list of cross-sections along its length. Each
section is a rectangle given by its x position, width, height and the height
of its centre. Between two sections the width, height and centre height vary
linearly, so each segment's four faces (top, bottom and two sides) are flat
trapezoids: a fuselage is a chain of trapezoidal 3D segments. A section of
zero width or height closes the body to a point or an edge (e.g. a nose).

Coordinates are the fuselage's own: ``x`` aft along the body, ``z`` up; the
body is symmetric about its own x-z plane. ``origin`` places the local point
(x = 0, z = 0) in body axes. With ``symmetric=True`` the fuselage is mirrored
about the aircraft centreline (body y -> -y), giving a pair of identical
bodies, e.g. the two fuselages of a twin-fuselage aircraft.

All derived quantities (length, volume, wetted area, ...) are computed from
the sections on demand and never stored.

Inputs
------
- Sections: x [m], width [m], height [m] and centre z [m], for one body
  (length N >= 2, x strictly increasing).
- Placement: ``origin`` (x, y, z) in body axes [m]; whether it is mirrored.

Outputs
-------
- Derived geometry of one body: length [m], maximum width and height [m],
  frontal area [m^2], volume [m^3], wetted area [m^2]; and the number of
  bodies (1, or 2 when mirrored off the centreline).
- Width, height and centre height at any x, by linear interpolation.
- Top-view and side-view outlines in body axes, for plotting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class FuselageSection:
    """One rectangular cross-section of a fuselage.

    Attributes
    ----------
    x : float
        Position along the fuselage [m], local axes (aft positive).
    width : float
        Width of the section [m] (body y direction). Zero closes the body.
    height : float
        Height of the section [m] (body z direction). Zero closes the body.
    z_centre : float
        Height of the section's centre [m], local axes. Lets the body droop
        or rise along its length, e.g. an upswept tail boom.
    """

    x: float
    width: float
    height: float
    z_centre: float = 0.0

    def __post_init__(self):
        if self.width < 0 or self.height < 0:
            raise ValueError(f"Section at x = {self.x}: width and height must be >= 0")


@dataclass(frozen=True)
class Fuselage:
    """A fuselage, boom or pod built from rectangular cross-sections.

    Attributes
    ----------
    name : str
        Unique name, e.g. "fuselage" or "boom".
    sections : sequence of FuselageSection
        At least two sections with strictly increasing x.
    origin : tuple of float
        Body-axis position (x, y, z) [m] of the local point x = 0, z = 0.
    symmetric : bool
        If True, the body is mirrored about the aircraft centreline, making
        a pair (needs origin y != 0).

    Properties (computed on demand, for one body)
    ---------------------------------------------
    x, width, height, z_centre : arrays of the section values, shape (N,)
    n_bodies : 1, or 2 when mirrored
    length [m], max_width [m], max_height [m], frontal_area_m2 [m^2],
    volume_m3 [m^3], wetted_area_m2 [m^2] (four side faces; flat end faces
    are not counted)

    Methods
    -------
    from_segments(name, nose_width, nose_height, segments, ...)
        Build segment by segment from the nose aft (class method).
    width_at(x), height_at(x), z_centre_at(x)
        Interpolated values at local x [m]; float or array in, array out.
    outline_top_body(), outline_side_body()
        Closed outlines in body axes for plotting, each an array of shape (M, 3).
    summary()
        Dict of the main derived quantities.
    """

    name: str
    sections: Sequence[FuselageSection]
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    symmetric: bool = False

    def __post_init__(self):
        object.__setattr__(self, "sections", tuple(self.sections))
        if len(self.sections) < 2:
            raise ValueError(f"{self.name}: need at least two sections")
        if np.any(np.diff(self.x) <= 0):
            raise ValueError(f"{self.name}: section x positions must strictly increase")
        if self.symmetric and self.origin[1] == 0:
            raise ValueError(f"{self.name}: a mirrored fuselage needs origin y != 0")

    @classmethod
    def from_segments(
        cls,
        name: str,
        nose_width: float,
        nose_height: float,
        segments: Sequence[dict],
        nose_z_centre: float = 0.0,
        **kwargs,
    ) -> "Fuselage":
        """Build a fuselage segment by segment, from the nose aft.

        Each segment is a dict with ``length`` [m] and the ``width`` and
        ``height`` [m] at its aft end, and optionally ``z_centre`` [m] there
        (defaults to the previous section's). Other keyword arguments
        (``origin``, ``symmetric``) are passed to the constructor.

        Example
        -------
        A pod with a pointed nose and a thin tail boom::

            Fuselage.from_segments(
                "fuselage", nose_width=0.0, nose_height=0.0,
                segments=[{"length": 0.15, "width": 0.12, "height": 0.14},
                          {"length": 0.60, "width": 0.12, "height": 0.14},
                          {"length": 0.25, "width": 0.03, "height": 0.03},
                          {"length": 1.20, "width": 0.02, "height": 0.02}],
            )
        """
        sections = [FuselageSection(x=0.0, width=nose_width, height=nose_height, z_centre=nose_z_centre)]
        for segment in segments:
            previous = sections[-1]
            sections.append(
                FuselageSection(
                    x=previous.x + segment["length"],
                    width=segment["width"],
                    height=segment["height"],
                    z_centre=segment.get("z_centre", previous.z_centre),
                )
            )
        return cls(name=name, sections=sections, **kwargs)

    # ------------------------------------------------------------------ #
    # Section arrays and interpolation
    # ------------------------------------------------------------------ #
    @property
    def x(self) -> np.ndarray:
        return np.array([section.x for section in self.sections])

    @property
    def width(self) -> np.ndarray:
        return np.array([section.width for section in self.sections])

    @property
    def height(self) -> np.ndarray:
        return np.array([section.height for section in self.sections])

    @property
    def z_centre(self) -> np.ndarray:
        return np.array([section.z_centre for section in self.sections])

    def width_at(self, x: float | np.ndarray) -> np.ndarray:
        """Width [m] at local x [m]."""
        return np.interp(x, self.x, self.width)

    def height_at(self, x: float | np.ndarray) -> np.ndarray:
        """Height [m] at local x [m]."""
        return np.interp(x, self.x, self.height)

    def z_centre_at(self, x: float | np.ndarray) -> np.ndarray:
        """Centre height [m] at local x [m]."""
        return np.interp(x, self.x, self.z_centre)

    # ------------------------------------------------------------------ #
    # Derived geometry, one body
    # ------------------------------------------------------------------ #
    @property
    def n_bodies(self) -> int:
        """Number of bodies this object describes: 2 if mirrored, else 1."""
        return 2 if self.symmetric else 1

    @property
    def length(self) -> float:
        return float(self.x[-1] - self.x[0])

    @property
    def max_width(self) -> float:
        return float(self.width.max())

    @property
    def max_height(self) -> float:
        return float(self.height.max())

    @property
    def frontal_area_m2(self) -> float:
        """Largest cross-sectional area, width x height [m^2], one body."""
        return float(np.max(self.width * self.height))

    @property
    def volume_m3(self) -> float:
        """Enclosed volume of one body [m^3].

        Cross-sectional area w(x) h(x) is quadratic within each segment, so
        Simpson's rule per segment is exact.
        """
        x, width, height = self.x, self.width, self.height
        area_start = width[:-1] * height[:-1]
        area_end = width[1:] * height[1:]
        area_mid = 0.25 * (width[:-1] + width[1:]) * (height[:-1] + height[1:])
        return float(np.sum(np.diff(x) / 6 * (area_start + 4 * area_mid + area_end)))

    @property
    def wetted_area_m2(self) -> float:
        """Area of the top, bottom and side faces of one body [m^2].

        Each face of a segment is a flat trapezoid: its parallel sides are the
        section edges at either end, and its height is the slant distance
        between them. Flat end faces (a blunt nose or tail) are not counted.
        """
        dx = np.diff(self.x)
        width, height, z_centre = self.width, self.height, self.z_centre
        z_top = z_centre + height / 2
        z_bottom = z_centre - height / 2

        top = 0.5 * (width[:-1] + width[1:]) * np.hypot(dx, np.diff(z_top))
        bottom = 0.5 * (width[:-1] + width[1:]) * np.hypot(dx, np.diff(z_bottom))
        side_slant = np.hypot(dx, np.diff(width) / 2)
        one_side = 0.5 * (height[:-1] + height[1:]) * side_slant
        return float(np.sum(top + bottom + 2 * one_side))

    # ------------------------------------------------------------------ #
    # Outlines for plotting, body axes
    # ------------------------------------------------------------------ #
    def outline_top_body(self, mirrored: bool = False) -> np.ndarray:
        """Closed top-view outline (x, y, z at the centreline height), shape (2N + 1, 3).

        With ``mirrored``, the outline of the mirror-image body (y -> -y).
        """
        x0, y0, z0 = self.origin
        y_centre = -y0 if mirrored else y0
        x = np.concatenate([self.x, self.x[::-1], self.x[:1]]) + x0
        half_width = np.concatenate([self.width, -self.width[::-1], self.width[:1]]) / 2
        z = np.full_like(x, z0)
        return np.column_stack([x, y_centre + half_width, z])

    def outline_side_body(self) -> np.ndarray:
        """Closed side-view outline (x, y, z), shape (2N + 1, 3)."""
        x0, y0, z0 = self.origin
        x = np.concatenate([self.x, self.x[::-1], self.x[:1]]) + x0
        z_top = self.z_centre + self.height / 2
        z_bottom = self.z_centre - self.height / 2
        z = np.concatenate([z_top, z_bottom[::-1], z_top[:1]]) + z0
        return np.column_stack([x, np.full_like(x, y0), z])

    def summary(self) -> dict:
        """Main derived quantities of one body, and the number of bodies."""
        return {
            "name": self.name,
            "n_bodies": self.n_bodies,
            "length_m": self.length,
            "max_width_m": self.max_width,
            "max_height_m": self.max_height,
            "frontal_area_m2": self.frontal_area_m2,
            "volume_m3": self.volume_m3,
            "wetted_area_m2": self.wetted_area_m2,
        }
