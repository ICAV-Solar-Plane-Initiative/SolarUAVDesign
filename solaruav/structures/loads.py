"""Wing loads at the design load cases.

Purpose
-------
Build the spanwise shear force, bending moment and torque on a wing at the
limit and ultimate load factors, from the lifting-line lift distribution and
the wing's own mass distribution.

Method
------
In a symmetric pull-up at load factor n the wing carries a lift distribution
L'(s) whose total is n W, while every kilogram of wing-borne mass m'(s) pulls
down with n g m'(s). The net upward running load on one side is

    w(s) = L'(s) - n g m'(s)                                   [N/m]

Integrating from the free tip (where everything is zero) towards the root:

    shear force      V(s) = integral from s to tip of w ds      [N]
    bending moment   M(s) = integral from s to tip of V ds      [N m]
    torque           T(s) = integral from s to tip of t ds      [N m]

where t(s) is the running twisting moment about the elastic axis (the spar
line, at a fixed chord fraction), positive nose-up:

    t(s) = L'(s) (x_ea - x_qc) + Cm0 q c^2 - n g m'(s) (x_ea - x_mass)

Lift acts at the local quarter chord x_qc, wing mass at its own chord
fraction x_mass. A positive bending moment bends the tip up (upper skin in
compression).

Assumptions: the wing carries all the aircraft's lift (tail load neglected);
loads are taken in the wing's local (x, s) plane, so small sweep and dihedral
are ignored; drag and in-plane loads are not included; mass not on the wing
(fuselage, battery, payload) adds to the lift required but gives no bending
relief.

Inputs
------
- Spanwise lift distribution from ``aerodynamics.lifting_line`` [N/m].
- Spanwise mass distribution of the wing (structure, panels, anything else
  carried in the wing) [kg/m], as a function of s.
- Limit load factor and safety factor (floats).

Outputs
-------
- A ``WingLoads`` for each load case: shear force [N], bending moment [N m]
  and torque [N m] at each spanwise station, as arrays of length n_points
  from root to tip, along with the running loads that produce them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from ..aerodynamics.airfoil import Airfoil
from ..aerodynamics.lifting_line import LiftingLineResult, solve_lifting_line_for_lift
from ..geometry.surface import LiftingSurface

STANDARD_GRAVITY_M_S2 = 9.80665

# A spanwise mass distribution: takes spanwise positions s [m] (array, any
# shape) and returns the mass per unit span [kg/m] there (same shape).
MassPerSpan = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class WingLoads:
    """Spanwise loads on one side of a wing at one load case.

    All arrays have shape (n_points,) and run from the root (s = 0) to the tip.

    Attributes
    ----------
    surface_name : str
        Name of the wing.
    load_factor : float
        Load factor n of this case (e.g. limit or ultimate).
    elastic_axis_chord_fraction : float
        Chordwise position of the elastic axis, as a fraction of local chord,
        about which the torque is taken.
    s_m : ndarray
        Spanwise positions [m].
    lift_per_span_N_m : ndarray
        Aerodynamic lift per unit span L'(s) [N/m].
    inertial_load_per_span_N_m : ndarray
        Downward inertial load of the wing mass per unit span, n g m'(s) [N/m].
    shear_force_N : ndarray
        Vertical shear force V(s) [N], positive up on the outboard part.
    bending_moment_N_m : ndarray
        Bending moment M(s) [N m], positive tip-up.
    torque_N_m : ndarray
        Torque T(s) about the elastic axis [N m], positive nose-up.

    Properties (computed on demand)
    -------------------------------
    net_load_per_span_N_m
        L' - n g m' [N/m], array.
    root_shear_force_N, root_bending_moment_N_m, root_torque_N_m
        Values at the root (s = 0), one side [N], [N m], [N m].

    Methods
    -------
    scaled(factor)
        A copy with every load multiplied by ``factor`` (e.g. the safety factor).
    """

    surface_name: str
    load_factor: float
    elastic_axis_chord_fraction: float
    s_m: np.ndarray
    lift_per_span_N_m: np.ndarray
    inertial_load_per_span_N_m: np.ndarray
    shear_force_N: np.ndarray
    bending_moment_N_m: np.ndarray
    torque_N_m: np.ndarray

    @property
    def net_load_per_span_N_m(self) -> np.ndarray:
        """Net upward running load w(s) = L'(s) - n g m'(s) [N/m]."""
        return self.lift_per_span_N_m - self.inertial_load_per_span_N_m

    @property
    def root_shear_force_N(self) -> float:
        """Shear force at the root, one side [N]."""
        return float(self.shear_force_N[0])

    @property
    def root_bending_moment_N_m(self) -> float:
        """Bending moment at the root, one side [N m]."""
        return float(self.bending_moment_N_m[0])

    @property
    def root_torque_N_m(self) -> float:
        """Torque at the root, one side [N m]."""
        return float(self.torque_N_m[0])

    def scaled(self, factor: float) -> "WingLoads":
        """A copy with every load (and the load factor) multiplied by ``factor`` (float)."""
        return replace(
            self,
            load_factor=self.load_factor * factor,
            lift_per_span_N_m=self.lift_per_span_N_m * factor,
            inertial_load_per_span_N_m=self.inertial_load_per_span_N_m * factor,
            shear_force_N=self.shear_force_N * factor,
            bending_moment_N_m=self.bending_moment_N_m * factor,
            torque_N_m=self.torque_N_m * factor,
        )


@dataclass(frozen=True)
class DesignLoadCases:
    """The wing loads at the limit and ultimate load factors.

    Attributes
    ----------
    limit : WingLoads
        Loads at the limit load factor (highest load expected in service).
        Use for the stiffness constraint (tip deflection).
    ultimate : WingLoads
        The limit loads multiplied by the safety factor. Use for the strength
        constraint (stress).
    limit_aero : LiftingLineResult
        The lifting-line solution at the limit load, for checking (e.g. peak
        section Cl against Cl_max) and plotting.
    """

    limit: WingLoads
    ultimate: WingLoads
    limit_aero: LiftingLineResult


def wing_loads(
    aero: LiftingLineResult,
    surface: LiftingSurface,
    airfoil: Airfoil,
    load_factor: float,
    mass_per_span_kg_m: MassPerSpan,
    elastic_axis_chord_fraction: float = 0.25,
    mass_chord_fraction: float = 0.40,
) -> WingLoads:
    """Shear force, bending moment and torque along one side of a wing.

    Parameters
    ----------
    aero : LiftingLineResult
        Lifting-line solution for this wing at this load case. Its total lift
        should be n W (see ``solve_lifting_line_for_lift``).
    surface : LiftingSurface
        The wing geometry that ``aero`` was solved for.
    airfoil : Airfoil
        Section properties; Cm0 adds a torque.
    load_factor : float
        Load factor n of this case. Scales the inertial load of the wing mass.
    mass_per_span_kg_m : function of s
        Wing-borne mass per unit span [kg/m], one side. Takes an array of s [m]
        and returns an array of the same shape. ``chord_proportional_mass``
        builds a simple one.
    elastic_axis_chord_fraction : float
        Elastic axis (spar) position as a fraction of local chord, from the
        leading edge. Torque is taken about it.
    mass_chord_fraction : float
        Chordwise position of the wing mass centroid, fraction of local chord.

    Returns
    -------
    WingLoads
        Arrays of shape (n_output of ``aero``,), root to tip.
    """
    if aero.surface_name != surface.name:
        raise ValueError(f"Lifting-line result is for '{aero.surface_name}', not '{surface.name}'")

    s = aero.s_m
    chord = surface.chord_at(s)
    lift_per_span = aero.lift_per_span_N_m
    inertial_load = load_factor * STANDARD_GRAVITY_M_S2 * np.asarray(mass_per_span_kg_m(s), dtype=float)
    net_load = lift_per_span - inertial_load

    # Chordwise lever arms from each load to the elastic axis [m]. Positive
    # means the load acts ahead of the axis, so an upward load twists nose-up.
    lift_ahead_of_axis = (elastic_axis_chord_fraction - 0.25) * chord
    mass_ahead_of_axis = (elastic_axis_chord_fraction - mass_chord_fraction) * chord
    section_moment = airfoil.cm0 * aero.dynamic_pressure_Pa * chord**2
    twisting_moment_per_span = (
        lift_per_span * lift_ahead_of_axis + section_moment - inertial_load * mass_ahead_of_axis
    )

    shear_force = _integrate_from_tip(net_load, s)
    return WingLoads(
        surface_name=surface.name,
        load_factor=float(load_factor),
        elastic_axis_chord_fraction=float(elastic_axis_chord_fraction),
        s_m=s,
        lift_per_span_N_m=lift_per_span,
        inertial_load_per_span_N_m=inertial_load,
        shear_force_N=shear_force,
        bending_moment_N_m=_integrate_from_tip(shear_force, s),
        torque_N_m=_integrate_from_tip(twisting_moment_per_span, s),
    )


def design_load_cases(
    surface: LiftingSurface,
    airfoil: Airfoil,
    aircraft_mass_kg: float,
    mass_per_span_kg_m: MassPerSpan,
    airspeed_m_s: float,
    air_density_kg_m3: float,
    limit_load_factor: float,
    safety_factor: float = 1.5,
    elastic_axis_chord_fraction: float = 0.25,
    mass_chord_fraction: float = 0.40,
    n_terms: int = 30,
    n_points: int = 201,
) -> DesignLoadCases:
    """Wing loads at the limit and ultimate load factors (symmetric pull-up).

    The limit case is a real flight condition: the lifting line is solved at
    the angle of attack that gives L = n_limit W at the manoeuvre airspeed,
    so the shape of the lift distribution is right for that lift (twist makes
    it change with angle of attack). The ultimate loads are the limit loads
    times the safety factor; they are a strength margin, not a flight
    condition, so the aerodynamics are not re-solved for them.

    Parameters
    ----------
    surface : LiftingSurface
        The wing (symmetric, horizontal, root on the centreline).
    airfoil : Airfoil
        Section properties of the wing.
    aircraft_mass_kg : float
        Total aircraft mass [kg]. The wing lifts n times its weight.
    mass_per_span_kg_m : function of s
        Wing-borne mass per unit span [kg/m], one side (see ``wing_loads``).
    airspeed_m_s : float
        Airspeed of the limit manoeuvre [m/s]. Must be high enough that the
        wing can reach n_limit W without stalling (at or above the manoeuvre
        speed V_A = V_stall sqrt(n_limit)).
    air_density_kg_m3 : float
        Air density [kg/m^3].
    limit_load_factor : float
        Limit load factor n_limit.
    safety_factor : float
        Ultimate load factor = limit_load_factor * safety_factor.
    elastic_axis_chord_fraction, mass_chord_fraction : float
        As for ``wing_loads``.
    n_terms : int
        Number of Fourier terms in the lifting-line solution.
    n_points : int
        Number of spanwise output points, root to tip.

    Returns
    -------
    DesignLoadCases
    """
    if not (aircraft_mass_kg > 0 and limit_load_factor > 0 and safety_factor >= 1):
        raise ValueError("Need aircraft mass > 0, limit load factor > 0 and safety factor >= 1")

    weight_N = aircraft_mass_kg * STANDARD_GRAVITY_M_S2
    limit_aero = solve_lifting_line_for_lift(
        surface, limit_load_factor * weight_N, airspeed_m_s, air_density_kg_m3,
        airfoil=airfoil, n_terms=n_terms, n_output=n_points,
    )
    limit_loads = wing_loads(
        limit_aero, surface, airfoil, limit_load_factor, mass_per_span_kg_m,
        elastic_axis_chord_fraction=elastic_axis_chord_fraction,
        mass_chord_fraction=mass_chord_fraction,
    )
    return DesignLoadCases(
        limit=limit_loads,
        ultimate=limit_loads.scaled(safety_factor),
        limit_aero=limit_aero,
    )


def chord_proportional_mass(surface: LiftingSurface, mass_kg: float) -> MassPerSpan:
    """A wing mass distribution in proportion to local chord.

    A reasonable first guess for a constant-thickness-ratio wing covered in
    solar panels: m'(s) = mass_kg * c(s) / S, which integrates over both
    sides to ``mass_kg``.

    Parameters
    ----------
    surface : LiftingSurface
        The wing.
    mass_kg : float
        Total wing-borne mass, both sides [kg].

    Returns
    -------
    function
        Takes s [m] (array, any shape), returns mass per span [kg/m] (same shape).
    """
    area_m2 = surface.area

    def mass_per_span_kg_m(s: np.ndarray) -> np.ndarray:
        return mass_kg * surface.chord_at(s) / area_m2

    return mass_per_span_kg_m


def _integrate_from_tip(values: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Running integral of ``values`` from each station out to the tip.

    Inputs: values and spanwise positions s [m], shape (M,), root to tip.
    Output: shape (M,), element i is the trapezoidal integral from s[i] to s[-1].
    """
    strip_integrals = 0.5 * (values[1:] + values[:-1]) * np.diff(s)
    from_tip = np.concatenate([np.cumsum(strip_integrals[::-1])[::-1], [0.0]])
    return from_tip
