"""Spanwise plots of the wing loads.

Purpose
-------
Show the running load, shear force, bending moment and torque along one side
of the wing at the design load cases, with the cruise lift for comparison.

Four stacked panels against spanwise position, root to tip:

1. Running load: lift - n x wing weight [N/m] per case, and cruise lift only.
2. Shear force V [N].
3. Bending moment M [N m].
4. Torque T about the elastic axis [N m].

Inputs
------
- ``DesignLoadCases`` from ``structures.design_load_cases``.
- Optionally, a cruise ``LiftingLineResult`` to draw its lift distribution.

Outputs
-------
- Matplotlib Figure and an array of Axes, shape (4,).
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from ..aerodynamics.lifting_line import LiftingLineResult
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
        axes[0].plot(loads.s_m, loads.net_load_per_span_N_m, label=f"{label}: lift − n × wing weight", **style)
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
    axes[-1].set_xlabel("Spanwise position s [m]")
    axes[0].set_title(f"Wing loads: {cases.limit.surface_name} (one side)")
    fig.text(
        0.5, 0.005,
        "Lift is n × total weight. Wing-borne mass relieves the load along the span; "
        "the fuselage weight (n × W_fuselage / 2 per side) reacts at the root, "
        "where it equals the root shear.",
        ha="center", fontsize=8, wrap=True,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig, axes
