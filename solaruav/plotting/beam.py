"""Spanwise plot of the required section properties of the wing beam.

Purpose
-------
Show the minimum section properties one side of the wing needs to carry its
design loads, so a spar or skin can be sized to them.

Three stacked panels against spanwise position, root to tip:

1. Required section modulus Z = I / c [mm^3] (strength, ultimate load).
2. Required flexural rigidity EI [N m^2] (tip deflection limit, limit load):
   the constant-curvature distribution, and the uniform value as a dashed line.
3. Required torsional rigidity GJ [N m^2] (tip twist limit, limit load):
   the constant-twist-rate distribution, and the uniform value as a dashed line.

Inputs
------
- A ``RequiredSectionProperties`` from ``structures.required_section_properties``.

Outputs
-------
- Matplotlib Figure and an array of Axes, shape (3,).
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from ..structures.beam import RequiredSectionProperties

DISTRIBUTION_COLOUR = "#2a78d6"  # blue: spanwise requirement
UNIFORM_COLOUR = "#8a8a86"       # grey, dashed: uniform value
LINE_WIDTH = 2.0
MM3_PER_M3 = 1e9


def plot_section_properties(properties: RequiredSectionProperties):
    """Plot the required section modulus, EI and GJ along one side of the wing.

    Parameters
    ----------
    properties : RequiredSectionProperties

    Returns
    -------
    fig : matplotlib Figure
    axes : ndarray of Axes, shape (3,)
    """
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    s = properties.s_m
    limit_n = properties.limit_loads.load_factor
    ultimate_n = properties.ultimate_loads.load_factor

    axes[0].plot(s, properties.required_section_modulus_m3 * MM3_PER_M3, color=DISTRIBUTION_COLOUR,
                 linewidth=LINE_WIDTH, label="required Z")
    axes[0].set_ylabel("Section modulus Z = I / c [mm^3]")
    axes[0].set_title(
        f"Strength: ultimate load (n = {ultimate_n:.1f}), "
        f"allowable stress {properties.allowable_stress_Pa / 1e6:.0f} MPa"
    )

    axes[1].plot(s, properties.required_flexural_rigidity_N_m2, color=DISTRIBUTION_COLOUR,
                 linewidth=LINE_WIDTH, label="constant-curvature distribution")
    axes[1].axhline(properties.required_uniform_flexural_rigidity_N_m2, color=UNIFORM_COLOUR,
                    linestyle="--", linewidth=1.5,
                    label=f"uniform: {properties.required_uniform_flexural_rigidity_N_m2:.0f} N m^2")
    axes[1].set_ylabel("Flexural rigidity EI [N m^2]")
    axes[1].set_title(
        f"Bending stiffness: limit load (n = {limit_n:.1f}), tip deflection "
        f"<= {properties.max_tip_deflection_m:.3f} m ({properties.max_tip_deflection_fraction:.0%} of b/2)"
    )

    axes[2].plot(s, properties.required_torsional_rigidity_N_m2, color=DISTRIBUTION_COLOUR,
                 linewidth=LINE_WIDTH, label="constant-twist-rate distribution")
    axes[2].axhline(properties.required_uniform_torsional_rigidity_N_m2, color=UNIFORM_COLOUR,
                    linestyle="--", linewidth=1.5,
                    label=f"uniform: {properties.required_uniform_torsional_rigidity_N_m2:.1f} N m^2")
    axes[2].set_ylabel("Torsional rigidity GJ [N m^2]")
    axes[2].set_title(
        f"Torsional stiffness: limit load (n = {limit_n:.1f}), "
        f"tip twist <= {properties.max_tip_twist_deg:.1f} deg"
    )
    axes[2].set_xlabel("Spanwise position s [m]")

    for ax in axes:
        ax.set_ylim(bottom=0)
        ax.set_xlim(0, properties.semispan_m)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"Required section properties: {properties.surface_name} (one side)")
    fig.tight_layout()
    return fig, axes
