"""Beam-theory analysis of the wing structure.

Purpose
-------
Treat one side of a wing as a cantilever beam, clamped at the root (s = 0),
to find its stiffness, stresses, deflection and twist under the spanwise
loads from ``loads``, and the mass of the spar that carries them. Write the
results to a CSV file named after the surface.

Method (Euler-Bernoulli bending and St Venant torsion, small deflections)
------
Rigidities from the spar section and material:

    EI(s) = E I(s)          flexural rigidity  [N m^2]
    GJ(s) = G J(s)          torsional rigidity [N m^2]

Stresses in a circular tube of outer radius r = D/2:

    bending stress   sigma = |M| r / I            (peak fibre stress; tension on
                                                   one side, compression on the other)
    shear stress     tau = |T| r / J + 2 |V| / A  (torsion plus the peak transverse
                                                   shear of a thin tube, added for
                                                   a conservative estimate)

Failure indices compare these with the material allowables; a value above 1
means the section fails (the reserve factor is 1 / failure index):

    bending failure index = sigma / allowable stress
    shear failure index   = tau / allowable shear stress

Deflection and twist, integrated from the clamped root, where slope,
deflection and twist are all zero:

    slope       dw/ds = integral from root of M / EI ds   [rad]
    deflection  w(s)  = integral from root of dw/ds ds    [m], positive up
    twist       phi(s) = integral from root of T / GJ ds  [rad], positive nose-up

Assumptions: linear elastic, small deflections (check the tip deflection is a
small fraction of the semispan); the spar alone carries all loads (skin, ribs
and panels add stiffness and mass that are not counted here); the root is
rigidly clamped on the centreline.

Inputs
------
- Spanwise loads from ``loads``: a ``WingLoads`` for each load case, with
  shear [N], bending moment [N m] and torque [N m], arrays of length n_points.
- The spar: a ``TubeSpar`` from ``sections`` (geometry along the span and
  material: E, G [Pa], density [kg/m^3], allowable stresses [Pa]).

Outputs
-------
- A ``BeamResult`` per load case: EI and GJ [N m^2], stresses [Pa], failure
  indices, slope [rad], deflection [m] and twist [deg] at each spanwise station
  (arrays of length n_points), and the totals the constraints need: maximum
  failure index, tip deflection [m] and spar mass [kg].
- A CSV file ``<output_dir>/<surface name>.csv`` with one row per spanwise
  station: the spar section properties, then every quantity above for each
  load case, with the case name as a column prefix.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .loads import DesignLoadCases, WingLoads
from .sections import TubeSpar


@dataclass(frozen=True)
class BeamResult:
    """Beam analysis of one side of a wing at one load case.

    Only the inputs are stored; every result is computed on demand from them.
    All arrays have shape (n_points,) and run from the root (s = 0) to the tip.

    Attributes
    ----------
    load_case : str
        Name of the load case, e.g. "limit" or "ultimate". Used as the column
        prefix in the CSV file.
    loads : WingLoads
        The spanwise loads of this case.
    spar : TubeSpar
        The spar that carries them.

    Properties: spanwise arrays
    ---------------------------
    surface_name (str), load_factor (float), s_m [m], span_fraction (s / semispan),
    outer_diameter_m [m], wall_thickness_m [m], area_m2 [m^2],
    second_moment_m4 [m^4], polar_moment_m4 [m^4], mass_per_span_kg_m [kg/m],
    flexural_rigidity_N_m2 (EI) [N m^2], torsional_rigidity_N_m2 (GJ) [N m^2],
    bending_stress_Pa [Pa], shear_stress_Pa [Pa],
    bending_failure_index, shear_failure_index, failure_index (the larger of the two),
    slope_rad [rad], deflection_m [m], twist_deg [deg].

    Properties: totals
    ------------------
    max_bending_stress_Pa [Pa], max_shear_stress_Pa [Pa], max_failure_index,
    critical_s_m (where the failure index is largest) [m],
    tip_deflection_m [m], tip_deflection_fraction (of the semispan),
    tip_twist_deg [deg], spar_mass_kg (both sides) [kg].
    """

    load_case: str
    loads: WingLoads
    spar: TubeSpar

    # ------------------------------------------------------------------ #
    # Positions and section properties
    # ------------------------------------------------------------------ #
    @property
    def surface_name(self) -> str:
        return self.loads.surface_name

    @property
    def load_factor(self) -> float:
        return self.loads.load_factor

    @property
    def s_m(self) -> np.ndarray:
        return self.loads.s_m

    @property
    def semispan_m(self) -> float:
        return float(self.s_m[-1])

    @property
    def span_fraction(self) -> np.ndarray:
        return self.s_m / self.semispan_m

    @property
    def outer_diameter_m(self) -> np.ndarray:
        return self.spar.outer_diameter_m(self.span_fraction)

    @property
    def wall_thickness_m(self) -> np.ndarray:
        return self.spar.wall_thickness_m(self.span_fraction)

    @property
    def area_m2(self) -> np.ndarray:
        return self.spar.area_m2(self.span_fraction)

    @property
    def second_moment_m4(self) -> np.ndarray:
        return self.spar.second_moment_m4(self.span_fraction)

    @property
    def polar_moment_m4(self) -> np.ndarray:
        return self.spar.polar_moment_m4(self.span_fraction)

    @property
    def mass_per_span_kg_m(self) -> np.ndarray:
        return self.spar.mass_per_span_kg_m(self.span_fraction)

    @property
    def flexural_rigidity_N_m2(self) -> np.ndarray:
        """EI [N m^2]."""
        return self.spar.material.youngs_modulus_Pa * self.second_moment_m4

    @property
    def torsional_rigidity_N_m2(self) -> np.ndarray:
        """GJ [N m^2]."""
        return self.spar.material.shear_modulus_Pa * self.polar_moment_m4

    # ------------------------------------------------------------------ #
    # Stresses
    # ------------------------------------------------------------------ #
    @property
    def bending_stress_Pa(self) -> np.ndarray:
        """Peak fibre bending stress |M| r / I [Pa]."""
        outer_radius = self.outer_diameter_m / 2
        return np.abs(self.loads.bending_moment_N_m) * outer_radius / self.second_moment_m4

    @property
    def shear_stress_Pa(self) -> np.ndarray:
        """Peak shear stress |T| r / J + 2 |V| / A [Pa]."""
        outer_radius = self.outer_diameter_m / 2
        torsion = np.abs(self.loads.torque_N_m) * outer_radius / self.polar_moment_m4
        transverse = 2 * np.abs(self.loads.shear_force_N) / self.area_m2
        return torsion + transverse

    @property
    def bending_failure_index(self) -> np.ndarray:
        return self.bending_stress_Pa / self.spar.material.allowable_stress_Pa

    @property
    def shear_failure_index(self) -> np.ndarray:
        return self.shear_stress_Pa / self.spar.material.allowable_shear_stress_Pa

    @property
    def failure_index(self) -> np.ndarray:
        """The larger of the bending and shear failure indices at each station."""
        return np.maximum(self.bending_failure_index, self.shear_failure_index)

    # ------------------------------------------------------------------ #
    # Deflection and twist
    # ------------------------------------------------------------------ #
    @property
    def slope_rad(self) -> np.ndarray:
        """Bending slope dw/ds [rad], zero at the root."""
        return _integrate_from_root(self.loads.bending_moment_N_m / self.flexural_rigidity_N_m2, self.s_m)

    @property
    def deflection_m(self) -> np.ndarray:
        """Vertical deflection w [m], positive up, zero at the root."""
        return _integrate_from_root(self.slope_rad, self.s_m)

    @property
    def twist_deg(self) -> np.ndarray:
        """Elastic twist [deg], positive nose-up, zero at the root."""
        return np.degrees(_integrate_from_root(self.loads.torque_N_m / self.torsional_rigidity_N_m2, self.s_m))

    # ------------------------------------------------------------------ #
    # Totals for the constraints
    # ------------------------------------------------------------------ #
    @property
    def max_bending_stress_Pa(self) -> float:
        return float(self.bending_stress_Pa.max())

    @property
    def max_shear_stress_Pa(self) -> float:
        return float(self.shear_stress_Pa.max())

    @property
    def max_failure_index(self) -> float:
        return float(self.failure_index.max())

    @property
    def critical_s_m(self) -> float:
        """Spanwise position [m] of the largest failure index."""
        return float(self.s_m[np.argmax(self.failure_index)])

    @property
    def tip_deflection_m(self) -> float:
        return float(self.deflection_m[-1])

    @property
    def tip_deflection_fraction(self) -> float:
        """Tip deflection as a fraction of the semispan (small-deflection theory
        is reasonable below about 0.1)."""
        return self.tip_deflection_m / self.semispan_m

    @property
    def tip_twist_deg(self) -> float:
        return float(self.twist_deg[-1])

    @property
    def spar_mass_kg(self) -> float:
        """Mass of the spar, both sides [kg]."""
        return 2 * float(_integrate_from_root(self.mass_per_span_kg_m, self.s_m)[-1])


def analyse_beam(loads: WingLoads, spar: TubeSpar, load_case: str) -> BeamResult:
    """Analyse one side of a wing as a cantilever beam at one load case.

    Parameters
    ----------
    loads : WingLoads
        Spanwise loads, arrays of shape (n_points,) from root to tip.
    spar : TubeSpar
        The spar section and material.
    load_case : str
        Name of the case, e.g. "limit". Letters, digits and underscores only,
        since it becomes a CSV column prefix.

    Returns
    -------
    BeamResult
    """
    if not re.fullmatch(r"\w+", load_case):
        raise ValueError(f"Load case name '{load_case}' must be letters, digits and underscores only")
    starts_at_root = abs(loads.s_m[0]) <= 1e-9 * loads.s_m[-1]  # allow rounding error
    if not starts_at_root or np.any(np.diff(loads.s_m) <= 0):
        raise ValueError("Loads must start at the root (s = 0) and increase towards the tip")
    return BeamResult(load_case=load_case, loads=loads, spar=spar)


def analyse_design_cases(cases: DesignLoadCases, spar: TubeSpar) -> list[BeamResult]:
    """Analyse the beam at the limit and ultimate load cases.

    Use the limit result for the stiffness constraint (tip deflection) and the
    ultimate result for the strength constraint (failure index <= 1).

    Inputs: design load cases from ``loads.design_load_cases``; the spar.
    Output: list of two BeamResults, [limit, ultimate].
    """
    return [analyse_beam(cases.limit, spar, "limit"), analyse_beam(cases.ultimate, spar, "ultimate")]


def write_beam_csv(results: Sequence[BeamResult], output_dir: str | Path = "output") -> Path:
    """Write beam results for one surface to ``<output_dir>/<surface name>.csv``.

    One row per spanwise station. The first columns are the spar section
    properties (the same for every case); then, for each result, its loads,
    stresses, failure indices, deflection and twist, each column prefixed with
    the load case name, e.g. ``ultimate_bending_stress_Pa``. Units are in the
    column names, and values are written to 8 significant figures. The output
    folder is created if it does not exist, and an existing file of the same
    name is overwritten.

    Parameters
    ----------
    results : sequence of BeamResult
        One or more load cases of the same surface, at the same stations.
    output_dir : str or Path
        Folder to write into. Defaults to ``output`` in the current folder.

    Returns
    -------
    Path
        The path of the written file.
    """
    if not results:
        raise ValueError("Need at least one BeamResult to write")
    first = results[0]
    for result in results[1:]:
        if result.surface_name != first.surface_name or not np.array_equal(result.s_m, first.s_m):
            raise ValueError("All results in one CSV must be for the same surface and stations")
    case_names = [result.load_case for result in results]
    if len(set(case_names)) != len(case_names):
        raise ValueError(f"Load case names must be unique, got {case_names}")

    columns = _section_columns(first)
    for result in results:
        columns.update(_load_case_columns(result))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{_safe_file_name(first.surface_name)}.csv"
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns.keys())
        for row in zip(*columns.values()):
            writer.writerow([f"{value:.8g}" for value in row])
    return path


def _section_columns(result: BeamResult) -> dict[str, np.ndarray]:
    """CSV columns that depend only on the spar: name -> array, shape (n_points,)."""
    return {
        "s_m": result.s_m,
        "span_fraction": result.span_fraction,
        "outer_diameter_m": result.outer_diameter_m,
        "wall_thickness_m": result.wall_thickness_m,
        "area_m2": result.area_m2,
        "second_moment_m4": result.second_moment_m4,
        "polar_moment_m4": result.polar_moment_m4,
        "EI_N_m2": result.flexural_rigidity_N_m2,
        "GJ_N_m2": result.torsional_rigidity_N_m2,
        "spar_mass_per_span_kg_m": result.mass_per_span_kg_m,
    }


def _load_case_columns(result: BeamResult) -> dict[str, np.ndarray]:
    """CSV columns for one load case, prefixed with its name: name -> array, shape (n_points,)."""
    columns = {
        "load_factor": np.full_like(result.s_m, result.load_factor),
        "shear_force_N": result.loads.shear_force_N,
        "bending_moment_N_m": result.loads.bending_moment_N_m,
        "torque_N_m": result.loads.torque_N_m,
        "bending_stress_Pa": result.bending_stress_Pa,
        "shear_stress_Pa": result.shear_stress_Pa,
        "bending_failure_index": result.bending_failure_index,
        "shear_failure_index": result.shear_failure_index,
        "slope_rad": result.slope_rad,
        "deflection_m": result.deflection_m,
        "twist_deg": result.twist_deg,
    }
    return {f"{result.load_case}_{name}": values for name, values in columns.items()}


def _safe_file_name(name: str) -> str:
    """Replace anything but letters, digits, '-' and '_' with '_', e.g. 'main wing' -> 'main_wing'."""
    return re.sub(r"[^\w\-]+", "_", name).strip("_") or "surface"


def _integrate_from_root(values: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Running integral of ``values`` from the root to each station.

    Inputs: values and spanwise positions s [m], shape (M,), root to tip.
    Output: shape (M,), element i is the trapezoidal integral from s[0] to s[i].
    """
    strip_integrals = 0.5 * (values[1:] + values[:-1]) * np.diff(s)
    return np.concatenate([[0.0], np.cumsum(strip_integrals)])
