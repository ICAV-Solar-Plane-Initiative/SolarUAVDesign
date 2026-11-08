"""Trim in steady symmetric flight: how the lift is split between wing and tail.

Purpose
-------
Find the wing lift and tail lift that together support n times the weight
with zero pitching moment about the centre of gravity, at a given load factor
n: level cruise (n = 1) or a steady pull-up (n > 1). Return the lifting-line
solution of each surface at its trimmed lift, for the aerodynamic and
structural analyses.

Method
------
Two conditions, two unknowns (wing lift L_w and tail lift L_t):

    vertical force     L_w + L_t = n W
    pitching moment    M_w(L_w) + M_t(L_t) = 0      (about the CG)

Each surface's pitching moment about the CG comes from its lifting-line
solution: its spanwise lift acting at the local quarter chords, plus the
sections' Cm0. In the linear lifting line both lift and moment vary linearly
with angle of attack, so a surface's moment is a straight line in its lift,
M = a + b L. Two solutions per surface give a and b; the two conditions are
then solved directly:

    L_t = -(a_w + a_t + b_w n W) / (b_t - b_w),     L_w = n W - L_t

Assumptions: only the wing and horizontal tail carry lift or pitching moment
(fuselage, propeller and drag moments are neglected); each surface is solved
in isolation, so the wing's downwash at the tail is ignored. Downwash changes
the angle the tail must be set at, not the tail load needed for trim, so the
loads are unaffected but the tail angle of attack in the result is not the
tail incidence.

Inputs
------
- The wing and horizontal tail (``LiftingSurface``) and their airfoils.
- Aircraft mass [kg] and CG body-axis x [m].
- Load factor n, airspeed [m/s] and air density [kg/m^3].

Outputs
-------
- A ``TrimmedFlight``: the wing and tail ``LiftingLineResult`` at their
  trimmed lifts (moments about the CG), and the lift split [N].
"""

from __future__ import annotations

from dataclasses import dataclass

from ..aerodynamics.airfoil import Airfoil
from ..aerodynamics.lifting_line import LiftingLineResult, solve_lifting_line_for_lift
from ..geometry.surface import LiftingSurface

STANDARD_GRAVITY_M_S2 = 9.80665


@dataclass(frozen=True)
class TrimmedFlight:
    """The trimmed lift split in steady symmetric flight at one load factor.

    Attributes
    ----------
    load_factor : float
        Load factor n (1 for level cruise).
    aircraft_mass_kg : float
        Total aircraft mass [kg].
    cg_x_m : float
        Body-axis x of the centre of gravity [m].
    wing : LiftingLineResult
        Wing solution at its trimmed lift, pitching moment about the CG.
    tail : LiftingLineResult
        Horizontal tail solution at its trimmed lift, pitching moment about the CG.

    Properties
    ----------
    required_lift_N : float, n W [N]
    wing_lift_N, tail_lift_N : float, lift of each surface [N] (tail positive up)
    pitching_moment_N_m : float, total about the CG [N m] (zero when trimmed)
    """

    load_factor: float
    aircraft_mass_kg: float
    cg_x_m: float
    wing: LiftingLineResult
    tail: LiftingLineResult

    @property
    def required_lift_N(self) -> float:
        return self.load_factor * self.aircraft_mass_kg * STANDARD_GRAVITY_M_S2

    @property
    def wing_lift_N(self) -> float:
        return self.wing.lift_N

    @property
    def tail_lift_N(self) -> float:
        return self.tail.lift_N

    @property
    def pitching_moment_N_m(self) -> float:
        return self.wing.pitching_moment_N_m + self.tail.pitching_moment_N_m


def trim_symmetric_flight(
    wing: LiftingSurface,
    wing_airfoil: Airfoil,
    tail: LiftingSurface,
    tail_airfoil: Airfoil,
    aircraft_mass_kg: float,
    cg_x_m: float,
    load_factor: float,
    airspeed_m_s: float,
    air_density_kg_m3: float,
    n_terms: int = 30,
    n_points: int = 201,
) -> TrimmedFlight:
    """Split n W between wing and tail so the pitching moment about the CG is zero.

    Parameters
    ----------
    wing, tail : LiftingSurface
        Main wing and horizontal tail (symmetric, horizontal, root on the centreline).
    wing_airfoil, tail_airfoil : Airfoil
        Section properties of each.
    aircraft_mass_kg : float
        Total aircraft mass [kg].
    cg_x_m : float
        Body-axis x of the centre of gravity [m].
    load_factor : float
        Load factor n: 1 for level flight, n_limit for the limit pull-up.
    airspeed_m_s, air_density_kg_m3 : float
        Flight condition [m/s], [kg/m^3].
    n_terms : int
        Number of Fourier terms in each lifting-line solution.
    n_points : int
        Number of spanwise output points in each solution.

    Returns
    -------
    TrimmedFlight
    """
    if not (aircraft_mass_kg > 0 and load_factor > 0):
        raise ValueError("Need aircraft mass > 0 and load factor > 0")
    required_lift_N = load_factor * aircraft_mass_kg * STANDARD_GRAVITY_M_S2

    def solve(surface, airfoil, lift_N):
        return solve_lifting_line_for_lift(
            surface, lift_N, airspeed_m_s, air_density_kg_m3, airfoil=airfoil,
            n_terms=n_terms, n_output=n_points, moment_reference_x_m=cg_x_m,
        )

    def moment_line(surface, airfoil):
        """Intercept a [N m] and slope b [m] of M = a + b L for one surface."""
        moment_at_zero = solve(surface, airfoil, 0.0).pitching_moment_N_m
        moment_at_required = solve(surface, airfoil, required_lift_N).pitching_moment_N_m
        return moment_at_zero, (moment_at_required - moment_at_zero) / required_lift_N

    wing_intercept, wing_slope = moment_line(wing, wing_airfoil)
    tail_intercept, tail_slope = moment_line(tail, tail_airfoil)
    if abs(tail_slope - wing_slope) < 1e-9:
        raise ValueError("Wing and tail act at the same point relative to the CG; the tail cannot trim")
    tail_lift_N = -(wing_intercept + tail_intercept + wing_slope * required_lift_N) / (tail_slope - wing_slope)

    return TrimmedFlight(
        load_factor=float(load_factor),
        aircraft_mass_kg=float(aircraft_mass_kg),
        cg_x_m=float(cg_x_m),
        wing=solve(wing, wing_airfoil, required_lift_N - tail_lift_N),
        tail=solve(tail, tail_airfoil, tail_lift_N),
    )
