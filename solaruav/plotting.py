"""Plotting of 2D aircraft geometry: top and side views.

Each view is a 2D projection of the aircraft's body axes:

* Top view: across the page is body y (starboard to the right); up the page is
  body x, flipped so the nose points up.
* Side view: across the page is body x (nose on the left); up the page is
  body z.

A surface that lies flat in the view (a wing seen from above, a fin seen from
the side) is drawn as a filled outline. A surface seen edge-on is drawn as a
thick line covering its chordwise extent.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .geometry.aircraft import Aircraft
from .geometry.surface import QUARTER_CHORD, LiftingSurface

VIEWS = ("top", "side")

# Line styles, gathered here so the look can be changed in one place.
OUTLINE_WIDTH = 1.5
FILL_OPACITY = 0.15
QUARTER_CHORD_WIDTH = 0.8
MAC_BAR_WIDTH = 2.5
EDGE_ON_WIDTH = 3.0
AERODYNAMIC_CENTRE_MARKER_SIZE = 5


def _body_to_view(points: np.ndarray, view: str) -> tuple[np.ndarray, np.ndarray]:
    """Project body-axis points, shape (n, 3), onto a view's
    (across-the-page, up-the-page) coordinates."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    if view == "top":
        return y, x
    return x, z


def _lies_flat_in_view(surface: LiftingSurface, view: str) -> bool:
    """True if the surface lies in the view plane, False if seen edge-on."""
    if view == "top":
        return surface.orientation == "horizontal"
    return surface.orientation == "vertical"


def _sides(surface: LiftingSurface) -> list[int]:
    """Span direction(s) to draw: +1 for the stored side, -1 for its mirror."""
    return [1, -1] if surface.symmetric else [1]


def _plot_flat(
    surface: LiftingSurface, view: str, ax: plt.Axes, color: str | None, label: str, show_mac: bool
) -> None:
    """Draw a surface that lies flat in the view: outline, quarter-chord line,
    and optionally the MAC and aerodynamic centre."""
    # Filled outline
    across, up = _body_to_view(surface.outline_body(), view)
    (outline_line,) = ax.plot(across, up, color=color, lw=OUTLINE_WIDTH, label=label)
    color = outline_line.get_color()  # resolve the automatic colour, if none was given
    ax.fill(across, up, color=color, alpha=FILL_OPACITY, lw=0)

    # Quarter-chord line (dashed)
    x_quarter_chord = surface.x_leading_edges + QUARTER_CHORD * surface.chords
    for side in _sides(surface):
        points = surface.to_body(x_quarter_chord, side * surface.span_positions)
        ax.plot(*_body_to_view(points, view), color=color, lw=QUARTER_CHORD_WIDTH, ls="--")

    if not show_mac:
        return

    # MAC as a thick bar at its span position
    mac_x_ends = np.array([surface.mac_x_leading_edge, surface.mac_x_leading_edge + surface.mac])
    for side in _sides(surface):
        mac_span_positions = np.full(2, side * surface.mac_span_position)
        points = surface.to_body(mac_x_ends, mac_span_positions)
        ax.plot(*_body_to_view(points, view), color=color, lw=MAC_BAR_WIDTH, solid_capstyle="butt")

    # Aerodynamic centre, drawn on the surface's root line
    centre = surface.to_body([surface.x_aerodynamic_centre], [0.0])
    ax.plot(
        *_body_to_view(centre, view),
        marker="o",
        ms=AERODYNAMIC_CENTRE_MARKER_SIZE,
        color=color,
        mfc="white",
        zorder=5,
    )


def _plot_edge_on(
    surface: LiftingSurface, view: str, ax: plt.Axes, color: str | None, label: str
) -> None:
    """Draw a surface seen edge-on as a thick line over its chordwise extent."""
    x_front = surface.origin[0] + surface.x_leading_edges.min()
    x_back = surface.origin[0] + surface.x_trailing_edges.max()
    _, y0, z0 = surface.origin
    line_ends = np.array([[x_front, y0, z0], [x_back, y0, z0]])
    ax.plot(
        *_body_to_view(line_ends, view),
        color=color,
        lw=EDGE_ON_WIDTH,
        label=label,
        solid_capstyle="butt",
        zorder=4,  # on top of filled outlines
    )


def plot_surface(
    surface: LiftingSurface,
    view: str = "top",
    ax: plt.Axes | None = None,
    show_mac: bool = True,
    color: str | None = None,
    label: str | None = None,
) -> plt.Axes:
    """Draw one lifting surface in the top or side view.

    Parameters
    ----------
    surface : LiftingSurface
        Surface to draw.
    view : "top" or "side"
        Which view to draw.
    ax : matplotlib Axes, optional
        Axes to draw on. A new figure is made if not given.
    show_mac : bool
        If True, draw the mean aerodynamic chord and aerodynamic centre.
    color : str, optional
        Matplotlib colour. Picked automatically if not given.
    label : str, optional
        Legend label. Defaults to the surface name.

    Returns
    -------
    matplotlib Axes
        The axes drawn on.
    """
    if view not in VIEWS:
        raise ValueError(f"view must be one of {VIEWS}, got '{view}'")
    if ax is None:
        _, ax = plt.subplots()
    if label is None:
        label = surface.name

    if _lies_flat_in_view(surface, view):
        _plot_flat(surface, view, ax, color, label, show_mac)
    else:
        _plot_edge_on(surface, view, ax, color, label)
    return ax


def _label_axes(ax: plt.Axes, view: str) -> None:
    """Set titles, axis labels and orientation for a view."""
    if view == "top":
        ax.invert_yaxis()  # so the nose points up the page
        ax.set_title("Top view")
        ax.set_xlabel("y [m]")
        ax.set_ylabel("x [m]")
    else:
        ax.set_title("Side view")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")


def plot_aircraft(
    aircraft: Aircraft,
    views: Sequence[str] = ("top", "side"),
    show_mac: bool = True,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot an aircraft in one or more views, side by side.

    Quarter-chord lines are dashed. With ``show_mac``, each surface's mean
    aerodynamic chord is drawn as a thick bar and its aerodynamic centre
    (quarter-MAC point) as an open circle on the surface's root line.

    Parameters
    ----------
    aircraft : Aircraft
        Aircraft to plot.
    views : sequence of "top" / "side"
        Views to draw, left to right.
    show_mac : bool
        If True, draw MACs and aerodynamic centres.
    figsize : (width, height), optional
        Figure size [inches]. Defaults to 6 inches square per view.

    Returns
    -------
    fig : matplotlib Figure
    axes : ndarray of matplotlib Axes
        One axes per view, in the order given.
    """
    views = list(views)
    for view in views:
        if view not in VIEWS:
            raise ValueError(f"view must be one of {VIEWS}, got '{view}'")
    if figsize is None:
        figsize = (6 * len(views), 6)

    fig, axes = plt.subplots(1, len(views), figsize=figsize, squeeze=False)
    axes = axes[0]
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, view in zip(axes, views):
        for index, surface in enumerate(aircraft.surfaces):
            colour = colours[index % len(colours)]  # same colour for a surface in every view
            plot_surface(surface, view=view, ax=ax, show_mac=show_mac, color=colour)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, lw=0.3, alpha=0.5)
        _label_axes(ax, view)

    axes[0].legend(loc="best", fontsize="small")
    fig.suptitle(aircraft.name)
    fig.tight_layout()
    return fig, axes
