"""Plotting of 2D aircraft geometry."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .geometry.aircraft import Aircraft
from .geometry.surface import LiftingSurface

_VIEWS = ("top", "side")


def _extent_line(surface: LiftingSurface) -> tuple[float, float]:
    """Body-axis x extent of a surface (for drawing it edge-on)."""
    x0 = surface.origin[0]
    return x0 + surface.x_le.min(), x0 + surface.x_te.max()


def plot_surface(
    surface: LiftingSurface,
    view: str = "top",
    ax: plt.Axes | None = None,
    show_mac: bool = True,
    color: str | None = None,
    label: str | None = None,
) -> plt.Axes:
    """Draw one lifting surface in the top or side view.

    Surfaces lying in the view plane are drawn as filled outlines; surfaces
    seen edge-on are drawn as a thick line along their chordwise extent.

    Top view axes: horizontal = body y (starboard to the right), vertical =
    body x with the nose at the top. Side view axes: horizontal = body x
    (nose to the left), vertical = body z.
    """
    if view not in _VIEWS:
        raise ValueError(f"view must be one of {_VIEWS}")
    if ax is None:
        _, ax = plt.subplots()
    label = surface.name if label is None else label
    in_plane = (view == "top") == (surface.orientation == "horizontal")
    x0, y0, z0 = surface.origin

    def to_plot(xb, yb, zb):
        return (yb, xb) if view == "top" else (xb, zb)

    if in_plane:
        pts = surface.outline_body()
        h, v = to_plot(pts[:, 0], pts[:, 1], pts[:, 2])
        (line,) = ax.plot(h, v, color=color, lw=1.5, label=label)
        ax.fill(h, v, color=line.get_color(), alpha=0.15, lw=0)
        c = line.get_color()

        # Quarter-chord line
        s = surface.s
        x_qc = surface.x_le + 0.25 * surface.chord
        sides = [1, -1] if surface.symmetric else [1]
        for sign in sides:
            qc = surface.to_body(x_qc, sign * s)
            ax.plot(*to_plot(qc[:, 0], qc[:, 1], qc[:, 2]), color=c, lw=0.8, ls="--")

        if show_mac:
            for sign in sides:
                xs = np.array([surface.x_le_mac, surface.x_le_mac + surface.mac])
                ss = np.full(2, sign * surface.s_mac)
                m = surface.to_body(xs, ss)
                ax.plot(*to_plot(m[:, 0], m[:, 1], m[:, 2]), color=c, lw=2.5, solid_capstyle="butt")
            ac = surface.to_body([surface.x_ac], [0.0])[0]
            ax.plot(*to_plot(*ac), marker="o", ms=5, color=c, mfc="white", zorder=5)
    else:
        xa, xb = _extent_line(surface)
        if view == "top":
            ax.plot([y0, y0], [xa, xb], color=color, lw=3, label=label, solid_capstyle="butt", zorder=4)
        else:
            ax.plot([xa, xb], [z0, z0], color=color, lw=3, label=label, solid_capstyle="butt", zorder=4)
    return ax


def plot_aircraft(
    aircraft: Aircraft,
    views: Sequence[str] = ("top", "side"),
    show_mac: bool = True,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot an aircraft in one or more views.

    Quarter-chord lines are dashed. With ``show_mac``, the mean aerodynamic
    chord of each in-plane surface is drawn as a thick bar and its
    quarter-MAC point (approximate aerodynamic centre, on the centreline) as
    an open circle.

    Returns
    -------
    fig, axes
        The figure and an array of axes, one per view.
    """
    views = list(views)
    for v in views:
        if v not in _VIEWS:
            raise ValueError(f"view must be one of {_VIEWS}, got '{v}'")
    if figsize is None:
        figsize = (6 * len(views), 6)
    fig, axes = plt.subplots(1, len(views), figsize=figsize, squeeze=False)
    axes = axes[0]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, view in zip(axes, views):
        for i, surface in enumerate(aircraft.surfaces):
            plot_surface(surface, view=view, ax=ax, show_mac=show_mac, color=colors[i % len(colors)])
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, lw=0.3, alpha=0.5)
        if view == "top":
            ax.invert_yaxis()  # nose at the top
            ax.set_xlabel("y [m]")
            ax.set_ylabel("x [m]")
            ax.set_title("Top view")
        else:
            ax.set_xlabel("x [m]")
            ax.set_ylabel("z [m]")
            ax.set_title("Side view")
    axes[0].legend(loc="best", fontsize="small")
    fig.suptitle(aircraft.name)
    fig.tight_layout()
    return fig, axes
