"""Required section properties of the wing beam, from the wing loads.

Purpose
-------
Treat one side of a wing as a cantilever beam, clamped at the root (s = 0),
and work out the minimum structural section properties it needs at each
spanwise station to carry the design loads. No section geometry is assumed:
the results say what any spar, tube, box or skin must provide, and are the
target for sizing the real structure. Write them to a CSV file named after
the surface.

Method (Euler-Bernoulli bending and St Venant torsion, small deflections)
------
Strength, at the ultimate load. The peak bending stress is |M| / Z, where the
section modulus Z = I / c (c = distance from the neutral axis to the extreme
fibre). Keeping it below the allowable stress needs

    Z_required(s) = |M_ultimate(s)| / allowable stress           [m^3]

Stiffness, at the limit load. Tip deflection and twist depend on the whole
stiffness distribution, so a tip limit does not fix EI or GJ at a station by
itself. Two well-defined answers are given:

1. Constant-rate distribution: every station bends (and twists) at the same
   rate, so no part of the beam is stiffer than it needs to be. Constant
   curvature kappa = M / EI gives tip deflection kappa L^2 / 2, and constant
   twist rate T / GJ gives tip twist rate x L, so

       EI_required(s) = |M_limit(s)| / kappa,   kappa = 2 w_tip,max / L^2
       GJ_required(s) = |T_limit(s)| / rate,    rate  = phi_tip,max / L

2. Uniform value: the single EI (or GJ) that, held constant along the span,
   just meets the limit (unit-load method):

       EI_uniform = |integral_0^L M_limit(s) (L - s) ds| / w_tip,max
       GJ_uniform = |integral_0^L T_limit(s) ds| / phi_tip,max

Assumptions: linear elastic, small deflections; the root is rigidly clamped
on the centreline; bending about the chordwise axis only.

Inputs
------
- Limit and ultimate spanwise loads from ``loads`` (``DesignLoadCases``):
  shear [N], bending moment [N m] and torque [N m], arrays of length n_points.
- Allowable bending stress of the intended material [Pa].
- Stiffness limits: tip deflection as a fraction of the semispan, tip twist [deg].

Outputs
-------
- A ``RequiredSectionProperties``: Z [m^3], EI and GJ [N m^2] required at each
  spanwise station (arrays of length n_points) and the uniform EI and GJ.
- A CSV file ``<output_dir>/<surface name>.csv`` with one row per spanwise
  station: the design loads and the required section properties.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .loads import DesignLoadCases, WingLoads


@dataclass(frozen=True)
class RequiredSectionProperties:
    """Minimum section properties of one side of a wing at each spanwise station.

    Only the inputs are stored; every result is computed on demand from them.
    All arrays have shape (n_points,) and run from the root (s = 0) to the tip.

    Attributes
    ----------
    limit_loads : WingLoads
        Loads at the limit load factor; set the stiffness requirements.
    ultimate_loads : WingLoads
        Loads at the ultimate load factor; set the strength requirement.
    allowable_stress_Pa : float
        Allowable bending stress of the intended material [Pa].
    max_tip_deflection_fraction : float
        Largest allowed tip deflection at limit load, as a fraction of the semispan.
    max_tip_twist_deg : float
        Largest allowed elastic tip twist at limit load [deg].

    Properties
    ----------
    surface_name (str), s_m [m], span_fraction (s / semispan), semispan_m [m],
    max_tip_deflection_m [m],
    required_section_modulus_m3 : array, Z = I / c needed for strength [m^3]
    design_curvature_per_m : float, constant curvature kappa [1/m]
    required_flexural_rigidity_N_m2 : array, constant-curvature EI [N m^2]
    required_uniform_flexural_rigidity_N_m2 : float, constant EI along the span [N m^2]
    design_twist_rate_rad_per_m : float, constant twist rate [rad/m]
    required_torsional_rigidity_N_m2 : array, constant-twist-rate GJ [N m^2]
    required_uniform_torsional_rigidity_N_m2 : float, constant GJ along the span [N m^2]
    """

    limit_loads: WingLoads
    ultimate_loads: WingLoads
    allowable_stress_Pa: float
    max_tip_deflection_fraction: float
    max_tip_twist_deg: float

    @property
    def surface_name(self) -> str:
        return self.limit_loads.surface_name

    @property
    def s_m(self) -> np.ndarray:
        return self.limit_loads.s_m

    @property
    def semispan_m(self) -> float:
        return float(self.s_m[-1])

    @property
    def span_fraction(self) -> np.ndarray:
        return self.s_m / self.semispan_m

    @property
    def max_tip_deflection_m(self) -> float:
        return self.max_tip_deflection_fraction * self.semispan_m

    # ------------------------------------------------------------------ #
    # Strength
    # ------------------------------------------------------------------ #
    @property
    def required_section_modulus_m3(self) -> np.ndarray:
        """Z = I / c needed so the ultimate bending stress stays within the allowable [m^3]."""
        return np.abs(self.ultimate_loads.bending_moment_N_m) / self.allowable_stress_Pa

    # ------------------------------------------------------------------ #
    # Bending stiffness
    # ------------------------------------------------------------------ #
    @property
    def design_curvature_per_m(self) -> float:
        """Constant curvature that gives the allowed tip deflection, 2 w_tip / L^2 [1/m]."""
        return 2 * self.max_tip_deflection_m / self.semispan_m**2

    @property
    def required_flexural_rigidity_N_m2(self) -> np.ndarray:
        """EI giving constant curvature along the span and the allowed tip deflection [N m^2]."""
        return np.abs(self.limit_loads.bending_moment_N_m) / self.design_curvature_per_m

    @property
    def required_uniform_flexural_rigidity_N_m2(self) -> float:
        """The constant EI that gives the allowed tip deflection [N m^2]."""
        lever_to_tip = self.semispan_m - self.s_m
        tip_deflection_times_EI = _integrate(self.limit_loads.bending_moment_N_m * lever_to_tip, self.s_m)
        return abs(tip_deflection_times_EI) / self.max_tip_deflection_m

    # ------------------------------------------------------------------ #
    # Torsional stiffness
    # ------------------------------------------------------------------ #
    @property
    def design_twist_rate_rad_per_m(self) -> float:
        """Constant twist rate that gives the allowed tip twist [rad/m]."""
        return np.radians(self.max_tip_twist_deg) / self.semispan_m

    @property
    def required_torsional_rigidity_N_m2(self) -> np.ndarray:
        """GJ giving a constant twist rate along the span and the allowed tip twist [N m^2]."""
        return np.abs(self.limit_loads.torque_N_m) / self.design_twist_rate_rad_per_m

    @property
    def required_uniform_torsional_rigidity_N_m2(self) -> float:
        """The constant GJ that gives the allowed tip twist [N m^2]."""
        tip_twist_times_GJ = _integrate(self.limit_loads.torque_N_m, self.s_m)
        return abs(tip_twist_times_GJ) / np.radians(self.max_tip_twist_deg)


def required_section_properties(
    cases: DesignLoadCases,
    allowable_stress_Pa: float,
    max_tip_deflection_fraction: float = 0.10,
    max_tip_twist_deg: float = 2.0,
) -> RequiredSectionProperties:
    """Minimum section properties of a wing beam to carry its design loads.

    Parameters
    ----------
    cases : DesignLoadCases
        Limit and ultimate loads from ``loads.design_load_cases``.
    allowable_stress_Pa : float
        Allowable bending stress of the intended material [Pa]: the lower of
        its tension and compression allowables.
    max_tip_deflection_fraction : float
        Largest allowed tip deflection at limit load, as a fraction of the
        semispan. 0.1 is also roughly where small-deflection theory stops
        being reliable.
    max_tip_twist_deg : float
        Largest allowed elastic tip twist at limit load [deg].

    Returns
    -------
    RequiredSectionProperties
    """
    if not (allowable_stress_Pa > 0 and max_tip_deflection_fraction > 0 and max_tip_twist_deg > 0):
        raise ValueError("Allowable stress and the stiffness limits must be positive")
    limit, ultimate = cases.limit, cases.ultimate
    starts_at_root = abs(limit.s_m[0]) <= 1e-9 * limit.s_m[-1]  # allow rounding error
    if not starts_at_root or np.any(np.diff(limit.s_m) <= 0):
        raise ValueError("Loads must start at the root (s = 0) and increase towards the tip")
    if not np.array_equal(limit.s_m, ultimate.s_m):
        raise ValueError("Limit and ultimate loads must be at the same stations")
    return RequiredSectionProperties(
        limit_loads=limit,
        ultimate_loads=ultimate,
        allowable_stress_Pa=float(allowable_stress_Pa),
        max_tip_deflection_fraction=float(max_tip_deflection_fraction),
        max_tip_twist_deg=float(max_tip_twist_deg),
    )


def write_section_properties_csv(
    properties: RequiredSectionProperties, output_dir: str | Path = "output"
) -> Path:
    """Write the required section properties to ``<output_dir>/<surface name>.csv``.

    One row per spanwise station: position, the design loads, then the
    required section modulus, EI and GJ. Units are in the column names and
    values are written to 8 significant figures. The output folder is created
    if it does not exist, and an existing file of the same name is overwritten.
    The uniform EI and GJ are single numbers, so they are not in the file;
    read them from the ``RequiredSectionProperties``.

    Parameters
    ----------
    properties : RequiredSectionProperties
        Result of ``required_section_properties``.
    output_dir : str or Path
        Folder to write into. Defaults to ``output`` in the current folder.

    Returns
    -------
    Path
        The path of the written file.
    """
    limit, ultimate = properties.limit_loads, properties.ultimate_loads
    columns = {
        "s_m": properties.s_m,
        "span_fraction": properties.span_fraction,
        "limit_bending_moment_N_m": limit.bending_moment_N_m,
        "limit_torque_N_m": limit.torque_N_m,
        "ultimate_shear_force_N": ultimate.shear_force_N,
        "ultimate_bending_moment_N_m": ultimate.bending_moment_N_m,
        "ultimate_torque_N_m": ultimate.torque_N_m,
        "required_section_modulus_m3": properties.required_section_modulus_m3,
        "required_EI_N_m2": properties.required_flexural_rigidity_N_m2,
        "required_GJ_N_m2": properties.required_torsional_rigidity_N_m2,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{_safe_file_name(properties.surface_name)}.csv"
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns.keys())
        for row in zip(*columns.values()):
            writer.writerow([f"{value:.8g}" for value in row])
    return path


def _safe_file_name(name: str) -> str:
    """Replace anything but letters, digits, '-' and '_' with '_', e.g. 'main wing' -> 'main_wing'."""
    return re.sub(r"[^\w\-]+", "_", name).strip("_") or "surface"


def _integrate(values: np.ndarray, s: np.ndarray) -> float:
    """Trapezoidal integral of ``values`` over spanwise positions ``s`` (both shape (M,))."""
    return float(np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(s)))
