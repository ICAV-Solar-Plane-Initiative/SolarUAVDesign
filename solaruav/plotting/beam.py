"""Spanwise plots of the wing beam analysis.

Purpose
-------
Show the spar's section properties, stiffness, stresses and deformation along
one side of the wing, for one or more load cases, so the structure can be
checked against its strength and stiffness constraints at a glance.

Eight panels against spanwise position, root to tip:

1. Spar outer diameter [mm]          2. Spar wall thickness [mm]
3. Flexural rigidity EI [N m^2]      4. Torsional rigidity GJ [N m^2]
5. Bending stress [MPa] vs allowable 6. Shear stress [MPa] vs allowable
7. Vertical deflection [m]           8. Elastic twist [deg]

Section properties (1-4) are the same for every case and drawn once in dark
ink; each load case gets its own colour in panels 5-8, the same in all of them.

Inputs
------
- A list of ``BeamResult`` (one per load case) for the same surface and spar.

Outputs
-------
- Matplotlib Figure and an array of Axes, shape (4, 2).
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt

from ..structures.beam import BeamResult

# Fixed load-case colours, in order (validated categorical palette), so a case
# keeps its colour in every panel. Section properties use dark ink; allowables
# and references use grey dashes.
CASE_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SECTION_COLOUR = "#3d3d3a"
REFERENCE_COLOUR = "#8a8a86"
LINE_WIDTH = 2.0
MM_PER_M = 1e3
PA_PER_MPA = 1e6


def plot_beam(results: Sequence[BeamResult]):
    """Plot the beam analysis of one surface for one or more load cases.

    Parameters
    ----------
    results : sequence of BeamResult
        Load cases of the same surface and spar, e.g. from
        ``structures.analyse_design_cases``. At most four.

    Returns
    -------
    fig : matplotlib Figure
    axes : ndarray of Axes, shape (4, 2)
    """
    if not 1 <= len(results) <= len(CASE_COLOURS):
        raise ValueError(f"Plot between 1 and {len(CASE_COLOURS)} load cases")
    first = results[0]
    fig, axes = plt.subplots(4, 2, figsize=(11, 12), sharex=True)

    _plot_section(axes[0, 0], first, first.outer_diameter_m * MM_PER_M, "Outer diameter [mm]", "Spar outer diameter")
    _plot_section(axes[0, 1], first, first.wall_thickness_m * MM_PER_M, "Wall thickness [mm]", "Spar wall thickness")
    _plot_section(axes[1, 0], first, first.flexural_rigidity_N_m2, "EI [N m^2]", "Flexural rigidity EI")
    _plot_section(axes[1, 1], first, first.torsional_rigidity_N_m2, "GJ [N m^2]", "Torsional rigidity GJ")
    _plot_stress(axes[2, 0], results, "bending")
    _plot_stress(axes[2, 1], results, "shear")
    _plot_deformation(axes[3, 0], results, "deflection")
    _plot_deformation(axes[3, 1], results, "twist")

    for ax in axes.flat:
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0, first.semispan_m)
    for ax in axes[-1]:
        ax.set_xlabel("Spanwise position s [m]")

    fig.suptitle(
        f"Spar beam: {first.surface_name}   {first.spar.material.name},   "
        f"spar mass {first.spar_mass_kg:.3f} kg (both sides)"
    )
    fig.tight_layout()
    return fig, axes


def _case_label(result: BeamResult) -> str:
    return f"{result.load_case}, n = {result.load_factor:.1f}"


def _plot_section(ax, result: BeamResult, values, ylabel: str, title: str) -> None:
    ax.plot(result.s_m, values, color=SECTION_COLOUR, linewidth=LINE_WIDTH)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(bottom=0)


def _plot_stress(ax, results: Sequence[BeamResult], kind: str) -> None:
    """Bending or shear stress for each case, with the allowable as a dashed line."""
    material = results[0].spar.material
    if kind == "bending":
        allowable_Pa, title = material.allowable_stress_Pa, "Bending stress |M| r / I"
    else:
        allowable_Pa, title = material.allowable_shear_stress_Pa, "Shear stress |T| r / J + 2|V| / A"

    peak_indices = []
    for result, colour in zip(results, CASE_COLOURS):
        stress_Pa = result.bending_stress_Pa if kind == "bending" else result.shear_stress_Pa
        failure_index = result.bending_failure_index if kind == "bending" else result.shear_failure_index
        peak_indices.append(f"{result.load_case} {failure_index.max():.2f}")
        ax.plot(result.s_m, stress_Pa / PA_PER_MPA, color=colour, linewidth=LINE_WIDTH,
                label=_case_label(result))
    ax.axhline(allowable_Pa / PA_PER_MPA, color=REFERENCE_COLOUR, linestyle="--", linewidth=1.5,
               label=f"allowable {allowable_Pa / PA_PER_MPA:.0f} MPa")
    ax.set_ylabel(f"{kind.capitalize()} stress [MPa]")
    ax.set_title(f"{title}\npeak failure index: " + ", ".join(peak_indices), fontsize=10)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)


def _plot_deformation(ax, results: Sequence[BeamResult], kind: str) -> None:
    """Vertical deflection or elastic twist for each case."""
    for result, colour in zip(results, CASE_COLOURS):
        if kind == "deflection":
            values = result.deflection_m
            label = f"{_case_label(result)}: tip {result.tip_deflection_m:.3f} m ({result.tip_deflection_fraction:.1%} of b/2)"
        else:
            values = result.twist_deg
            label = f"{_case_label(result)}: tip {result.tip_twist_deg:.2f} deg"
        ax.plot(result.s_m, values, color=colour, linewidth=LINE_WIDTH, label=label)
    ax.axhline(0, color="black", linewidth=0.6)
    if kind == "deflection":
        ax.set_ylabel("Deflection w [m], up")
        ax.set_title("Vertical deflection (clamped at the root)")
    else:
        ax.set_ylabel("Twist [deg], nose-up")
        ax.set_title("Elastic twist about the spar")
    ax.legend(fontsize=8)
