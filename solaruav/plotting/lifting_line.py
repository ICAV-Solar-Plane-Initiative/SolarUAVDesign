"""Spanwise plots of a lifting-line solution.

Purpose
-------
Show what the lifting-line solver found for one surface, so the lift
distribution can be judged at a glance against the ideal (elliptic) loading
for the same total lift.

Six panels, each against spanwise position (root to tip) except the last:

1. Lift per unit span L'(s), with the elliptic distribution of equal lift.
2. Section lift coefficient Cl(s), with the surface CL.
3. Angles: geometric section angle (alpha + twist), induced angle alpha_i,
   and effective angle (geometric - alpha_i).
4. Downwash velocity w(s), with the uniform downwash of the elliptic wing.
5. Induced drag per unit span D_i'(s), with the elliptic distribution.
6. Fourier coefficients |A_n / A_1|: how far the loading is from elliptic.
   The span efficiency is e = 1 / (1 + delta), delta = sum_{n>1} n (A_n/A_1)^2.

Thin vertical lines mark the stations (panel joints) of the surface.

Inputs
------
- A ``LiftingLineResult`` from ``aerodynamics.solve_lifting_line``.
- The ``LiftingSurface`` it was solved for (for twist and station positions).

Outputs
-------
- Matplotlib Figure and an array of Axes, shape (3, 2).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..aerodynamics.lifting_line import LiftingLineResult
from ..geometry.surface import LiftingSurface

# Fixed series colours (validated categorical order) and a neutral reference.
SOLUTION_COLOUR = "#2a78d6"  # blue: the lifting-line solution
SECOND_COLOUR = "#eb6834"    # orange
THIRD_COLOUR = "#1baf7a"     # aqua
REFERENCE_COLOUR = "#8a8a86"  # grey, dashed: elliptic / ideal references
LINE_WIDTH = 2.0


def plot_lifting_line(result: LiftingLineResult, surface: LiftingSurface):
    """Plot the spanwise lifting-line solution for one surface.

    Parameters
    ----------
    result : LiftingLineResult
        The solution to plot.
    surface : LiftingSurface
        The surface ``result`` was solved for.

    Returns
    -------
    fig : matplotlib Figure
    axes : ndarray of Axes, shape (3, 2)
    """
    fig, axes = plt.subplots(3, 2, figsize=(11, 10))
    elliptic = _elliptic_reference(result)

    _plot_lift(axes[0, 0], result, elliptic)
    _plot_section_cl(axes[0, 1], result)
    _plot_angles(axes[1, 0], result, surface)
    _plot_downwash(axes[1, 1], result, elliptic)
    _plot_induced_drag(axes[2, 0], result, elliptic)
    _plot_fourier_coefficients(axes[2, 1], result)

    for ax in axes.flat[:5]:
        for station_s in surface.s[1:-1]:
            ax.axvline(station_s, color=REFERENCE_COLOUR, linewidth=0.6, alpha=0.5)
        ax.set_xlabel("Spanwise position s [m]")
        ax.set_xlim(0, result.reference_span_m / 2)
    for ax in axes.flat:
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        f"Lifting line: {result.surface_name}   "
        f"V = {result.airspeed_m_s:.1f} m/s,  alpha = {result.alpha_deg:.2f} deg,  "
        f"CL = {result.lift_coefficient:.3f},  CDi = {result.induced_drag_coefficient:.4f},  "
        f"e = {result.span_efficiency:.3f},  L = {result.lift_N:.1f} N,  Di = {result.induced_drag_N:.2f} N"
    )
    fig.tight_layout()
    return fig, axes


def _elliptic_reference(result: LiftingLineResult) -> dict:
    """The elliptic wing of the same span, lift and airspeed, at result.s_m.

    Output: dict of arrays, shape (n_output,): lift_per_span [N/m],
    downwash [m/s] (uniform), induced_drag_per_span [N/m].
    """
    semispan = result.reference_span_m / 2
    root_lift_per_span = 4 * result.lift_N / (np.pi * result.reference_span_m)
    lift_per_span = root_lift_per_span * np.sqrt(np.clip(1 - (result.s_m / semispan) ** 2, 0, None))
    induced_angle_rad = result.lift_coefficient / (np.pi * result.aspect_ratio)
    return {
        "lift_per_span": lift_per_span,
        "downwash": np.full_like(result.s_m, result.airspeed_m_s * induced_angle_rad),
        "induced_drag_per_span": lift_per_span * induced_angle_rad,
    }


def _plot_lift(ax, result: LiftingLineResult, elliptic: dict) -> None:
    ax.plot(result.s_m, result.lift_per_span_N_m, color=SOLUTION_COLOUR, linewidth=LINE_WIDTH,
            label="lifting line")
    ax.plot(result.s_m, elliptic["lift_per_span"], color=REFERENCE_COLOUR, linestyle="--",
            linewidth=1.5, label="elliptic, same lift")
    ax.set_ylabel("Lift per span L' [N/m]")
    ax.set_title("Lift distribution")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)


def _plot_section_cl(ax, result: LiftingLineResult) -> None:
    ax.plot(result.s_m, result.local_lift_coefficient, color=SOLUTION_COLOUR, linewidth=LINE_WIDTH,
            label="section Cl")
    ax.axhline(result.lift_coefficient, color=REFERENCE_COLOUR, linestyle="--", linewidth=1.5,
               label=f"surface CL = {result.lift_coefficient:.3f}")
    ax.set_ylabel("Section lift coefficient Cl")
    ax.set_title("Section lift coefficient (check against Cl_max)")
    ax.legend(fontsize=8)


def _plot_angles(ax, result: LiftingLineResult, surface: LiftingSurface) -> None:
    geometric_deg = result.alpha_deg + surface.twist_deg_at(result.s_m)
    effective_deg = geometric_deg - result.induced_angle_deg
    ax.plot(result.s_m, geometric_deg, color=SOLUTION_COLOUR, linewidth=LINE_WIDTH,
            label="geometric (alpha + twist)")
    ax.plot(result.s_m, result.induced_angle_deg, color=SECOND_COLOUR, linewidth=LINE_WIDTH,
            label="induced alpha_i")
    ax.plot(result.s_m, effective_deg, color=THIRD_COLOUR, linewidth=LINE_WIDTH,
            label="effective (geometric - alpha_i)")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Angle [deg]")
    ax.set_title("Section angles of attack")
    ax.legend(fontsize=8)


def _plot_downwash(ax, result: LiftingLineResult, elliptic: dict) -> None:
    ax.plot(result.s_m, result.downwash_m_s, color=SOLUTION_COLOUR, linewidth=LINE_WIDTH,
            label="lifting line")
    ax.plot(result.s_m, elliptic["downwash"], color=REFERENCE_COLOUR, linestyle="--",
            linewidth=1.5, label="elliptic (uniform)")
    ax.set_ylabel("Downwash w [m/s]")
    ax.set_title("Downwash at the lifting line (positive down)")
    ax.legend(fontsize=8)


def _plot_induced_drag(ax, result: LiftingLineResult, elliptic: dict) -> None:
    ax.plot(result.s_m, result.induced_drag_per_span_N_m, color=SOLUTION_COLOUR,
            linewidth=LINE_WIDTH, label=f"lifting line, Di = {result.induced_drag_N:.3f} N")
    elliptic_drag_N = result.lift_N * result.lift_coefficient / (np.pi * result.aspect_ratio)
    ax.plot(result.s_m, elliptic["induced_drag_per_span"], color=REFERENCE_COLOUR, linestyle="--",
            linewidth=1.5, label=f"elliptic, Di = {elliptic_drag_N:.3f} N")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Induced drag per span Di' [N/m]")
    ax.set_title("Induced drag distribution")
    ax.legend(fontsize=8)


def _plot_fourier_coefficients(ax, result: LiftingLineResult) -> None:
    ratio = np.abs(result.fourier_coefficients / result.fourier_coefficients[0])
    higher = result.harmonics > 1
    delta = float(np.sum(result.harmonics[higher] * ratio[higher] ** 2))
    ax.semilogy(result.harmonics, ratio, "o", color=SOLUTION_COLOUR, markersize=5)
    ax.set_xlabel("Harmonic n")
    ax.set_ylabel("|A_n / A_1|")
    ax.set_title(f"Fourier coefficients: delta = {delta:.3f}, e = 1/(1+delta) = {1 / (1 + delta):.3f}")
