"""Plots of the wing and fuselage loads.

Purpose
-------
``plot_wing_loads``: the running load, shear force, bending moment and torque
along one side of the wing at the design load cases, with the cruise lift for
comparison; works for any lifting surface, e.g. the tailplane.
``plot_fuselage_loads``: shear force, bending moment and required section
modulus along one fuselage body, with the mass items and tail force marked.

Four stacked panels against spanwise position, root to tip:

1. Running load: lift - n x surface weight [N/m] per case, and cruise lift only.
2. Shear force V [N].
3. Bending moment M [N m].
4. Torque T about the elastic axis [N m].

Point loads away from the root (fuselage mounts on a wing, boom supports
under a tailplane) are marked by dotted lines.

Inputs
------
- ``DesignLoadCases`` from ``structures.design_load_cases``.
- Optionally, a cruise ``LiftingLineResult`` to draw its lift distribution.

Outputs
-------
- Matplotlib Figure and an array of Axes for each plot.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from ..aerodynamics.lifting_line import LiftingLineResult
from ..structures.beam import RequiredFuselageSectionProperties
from ..structures.fuselage_loads import FuselageLoads
from ..structures.loads import DesignLoadCases

# Same case colours as the beam plot: limit blue, ultimate orange.
CASE_COLOURS = ["#2a78d6", "#eb6834"]
REFERENCE_COLOUR = "#3d3d3a"
LINE_WIDTH = 2.0


def plot_wing_loads(cases: DesignLoadCases, cruise: LiftingLineResult | None = None):
    """Plot the wing loads at the limit and ultimate cases.

    Parameters
    ----------
    cases : DesignLoadCases
        Limit and ultimate wing loads.
    cruise : LiftingLineResult, optional
        Cruise solution of the same wing; its lift per span is drawn dashed
        in the running-load panel.

    Returns
    -------
    fig : matplotlib Figure
    axes : ndarray of Axes, shape (4,)
    """
    fig, axes = plt.subplots(4, 1, figsize=(7, 10), sharex=True)
    if cruise is not None:
        axes[0].plot(cruise.s_m, cruise.lift_per_span_N_m, color=REFERENCE_COLOUR, linestyle="--",
                     linewidth=1.5, label="cruise, n = 1: lift only")

    case_list = [("limit", cases.limit), ("ultimate", cases.ultimate)]
    for (name, loads), colour in zip(case_list, CASE_COLOURS):
        label = f"{name}, n = {loads.load_factor:.1f}"
        style = dict(color=colour, linewidth=LINE_WIDTH)
        axes[0].plot(loads.s_m, loads.net_load_per_span_N_m, label=f"{label}: lift − n × own weight", **style)
        axes[1].plot(loads.s_m, loads.shear_force_N, label=label, **style)
        axes[2].plot(loads.s_m, loads.bending_moment_N_m, label=label, **style)
        axes[3].plot(loads.s_m, loads.torque_N_m, label=label, **style)

    axis_fraction = cases.limit.elastic_axis_chord_fraction
    ylabels = ["Running load [N/m]", "Shear force [N]", "Bending moment [N m]",
               f"Torque about {axis_fraction:.0%} chord [N m]"]
    for ax, ylabel in zip(axes, ylabels):
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.legend(fontsize=8)
    for point_load in cases.limit.point_loads:
        if point_load.s_m == 0:
            continue  # acts at the root: no effect on the wing's internal loads
        for ax in axes:
            ax.axvline(point_load.s_m, color=REFERENCE_COLOUR, linewidth=0.8, linestyle=":")
        direction = "down" if point_load.downward_force_N >= 0 else "up"
        axes[1].annotate(f"{point_load.name}\n{abs(point_load.downward_force_N):.1f} N {direction} at limit",
                         (point_load.s_m, 0), xytext=(4, 4), textcoords="offset points",
                         fontsize=8, color=REFERENCE_COLOUR)
    axes[-1].set_xlabel("Spanwise position s [m]")
    axes[0].set_title(f"Loads: {cases.limit.surface_name} (one side)")
    fig.text(
        0.5, 0.005,
        "Lift is this surface's trimmed lift (wing + tail = n × total weight). Its own mass relieves the "
        "load along the span; point loads (dotted lines: fuselage mounts, boom supports) step the loads "
        "where they act; whatever remains (e.g. a central fuselage) reacts at the root, twice the root shear.",
        ha="center", fontsize=8, wrap=True,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig, axes


def plot_fuselage_loads(limit: FuselageLoads, required: RequiredFuselageSectionProperties | None = None):
    """Plot the loads along one fuselage body, and its required section modulus.

    Panels against body-axis x, nose to tail: shear force, bending moment
    (limit, and ultimate if ``required`` is given), required section modulus
    Z [mm^3] (if ``required`` is given). The wing mount and the mass items are
    marked with dotted lines.

    Parameters
    ----------
    limit : FuselageLoads
        Loads at the limit load factor.
    required : RequiredFuselageSectionProperties, optional
        Required properties from the ultimate loads.

    Returns
    -------
    fig : matplotlib Figure
    axes : ndarray of Axes, shape (2,) or (3,)
    """
    n_panels = 2 if required is None else 3
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 3 * n_panels + 1), sharex=True)
    cases = [(limit, CASE_COLOURS[0], "limit")]
    if required is not None:
        cases.append((required.ultimate_loads, CASE_COLOURS[1], "ultimate"))
    for loads, colour, name in cases:
        label = f"{name}, n = {loads.load_factor:.1f}"
        axes[0].plot(loads.x_m, loads.shear_force_N, color=colour, linewidth=LINE_WIDTH, label=label)
        axes[1].plot(loads.x_m, loads.bending_moment_N_m, color=colour, linewidth=LINE_WIDTH, label=label)
    axes[0].set_ylabel("Shear force [N]")
    axes[1].set_ylabel("Bending moment [N m]\n(+ hogging)")
    if required is not None:
        axes[2].plot(required.x_m, required.required_section_modulus_m3 * 1e9, color=CASE_COLOURS[1],
                     linewidth=LINE_WIDTH, label=f"required Z, {required.allowable_stress_Pa / 1e6:.0f} MPa allowable")
        axes[2].set_ylabel("Section modulus Z [mm^3]")
        axes[2].set_ylim(bottom=0)

    markers = [("mount", limit.mount_x_m)] + [(item.name, item.x_m) for item in limit.mass_items]
    markers += [(f"{force.name} lift {force.upward_force_N:+.1f} N", force.x_m) for force in limit.point_forces]
    for name, x in markers:
        for ax in axes:
            ax.axvline(x, color=REFERENCE_COLOUR, linewidth=0.8, linestyle=":")
        axes[0].annotate(name, (x, 0), xytext=(3, 3), textcoords="offset points", fontsize=8,
                         color=REFERENCE_COLOUR, rotation=90)
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Body-axis x [m] (nose to tail)")
    axes[0].set_title(
        f"Fuselage loads: {limit.fuselage_name} (one body)   mount reaction at limit: "
        f"{limit.mount_force_N:.0f} N down, {limit.mount_moment_N_m:+.1f} N m nose-up"
    )
    fig.tight_layout()
    return fig, axes
