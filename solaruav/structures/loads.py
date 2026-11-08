"""Wing loads at the design load cases.

Purpose
-------
Build the spanwise shear force, bending moment and torque on a wing at the
limit and ultimate load factors, from the lifting-line lift distribution, the
wing's own mass distribution, and concentrated loads where fuselages are
mounted on the wing (see ``fuselage_loads``).

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

Mount loads. A fuselage mounted on the wing at s_i pushes down on it with a
force F_i (its share of n W) and a pitching moment M_i about the elastic axis,
both found from the fuselage's own equilibrium in ``fuselage_loads``. Every
station inboard of s_i carries them:

    V(s) += F_i              for s < s_i
    M(s) += F_i (s_i - s)    for s < s_i
    T(s) += M_i              for s < s_i

So the fuselages of a twin-fuselage aircraft, mounted at s = +-y_f, relieve
the wing's bending inboard of y_f. A central fuselage mounts at s = 0 and
loads only the root. Any mass that is neither in the wing distribution nor
on a mounted fuselage also hangs at the root: it adds to the lift required
but gives no relief.

The same routine gives the loads on the tailplane, from its own trimmed lift,
held up where it meets the boom(s) (``support_s_m`` in ``design_load_cases``).

Assumptions: the wing and tail lift come from the trimmed lifting-line
solutions (fuselage lift neglected; the lifting line ignores fuselage
interference);
loads are taken in the wing's local (x, s) plane, so small sweep and dihedral
are ignored; drag and in-plane loads are not included.

Inputs
------
- Spanwise lift distribution from ``aerodynamics.lifting_line`` [N/m], at
  the trimmed lift (``stability.trim_symmetric_flight``).
- Spanwise mass distribution of the wing (structure, panels, anything else
  carried in the wing) [kg/m], as a function of s.
- Mount loads from fuselages: spanwise position [m], downward force [N] and
  nose-up moment [N m], already at the case's load factor.
- Limit load factor and safety factor (floats).

Outputs
-------
- A ``WingLoads`` for each load case: shear force [N], bending moment [N m]
  and torque [N m] at each spanwise station, as arrays of length n_points
  from root to tip, along with the running loads that produce them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Sequence

import numpy as np

from ..aerodynamics.airfoil import Airfoil
from ..aerodynamics.lifting_line import LiftingLineResult
from ..geometry.surface import LiftingSurface

STANDARD_GRAVITY_M_S2 = 9.80665

# A spanwise mass distribution: takes spanwise positions s [m] (array, any
# shape) and returns the mass per unit span [kg/m] there (same shape).
MassPerSpan = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class WingPointLoad:
    """A concentrated load on one side of a symmetric wing, e.g. a fuselage mount.

    The values are at the load case's load factor. On a symmetric wing the
    same load is mirrored on the other side, so a pair of fuselages at
    y = +-0.8 m is one WingPointLoad with s_m = 0.8 and the load of one body.
    Build one from a fuselage with ``FuselageLoads.wing_point_load``.

    Attributes
    ----------
    name : str
        What applies it, e.g. "fuselage".
    s_m : float
        Spanwise position, one side [m], 0 (root) to b/2. A load at s = 0
        acts at the root and adds nothing to the wing's internal loads.
    downward_force_N : float
        Force pushing the wing down [N].
    nose_up_moment_N_m : float
        Pitching moment applied to the wing about its elastic axis [N m],
        positive nose-up.

    Methods
    -------
    scaled(factor)
        A copy with the force and moment multiplied by ``factor``.
    """

    name: str
    s_m: float
    downward_force_N: float
    nose_up_moment_N_m: float = 0.0

    def __post_init__(self):
        if self.s_m < 0:
            raise ValueError(f"{self.name}: spanwise position must be >= 0")

    def scaled(self, factor: float) -> "WingPointLoad":
        return replace(self, downward_force_N=self.downward_force_N * factor,
                       nose_up_moment_N_m=self.nose_up_moment_N_m * factor)


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
    point_loads : tuple of WingPointLoad
        Concentrated loads included (e.g. fuselage mounts; empty if none).

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
    point_loads: tuple[WingPointLoad, ...] = ()

    @property
    def net_load_per_span_N_m(self) -> np.ndarray:
        """Net upward running load w(s) = L'(s) - n g m'(s) [N/m]. Point loads
        are concentrated, so they are not in this distributed load."""
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
            point_loads=tuple(point_load.scaled(factor) for point_load in self.point_loads),
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
    point_loads: Sequence[WingPointLoad] = (),
) -> WingLoads:
    """Shear force, bending moment and torque along one side of a wing.

    Parameters
    ----------
    aero : LiftingLineResult
        Lifting-line solution for this wing at this load case, at its trimmed
        lift (see ``stability.trim_symmetric_flight``).
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
    point_loads : sequence of WingPointLoad
        Concentrated loads on one side of the wing (mirrored on the other), at
        this case's load factor, e.g. fuselage mounts.

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
    point_shear, point_moment, point_torque = _point_load_effects(point_loads, s)
    return WingLoads(
        surface_name=surface.name,
        load_factor=float(load_factor),
        elastic_axis_chord_fraction=float(elastic_axis_chord_fraction),
        s_m=s,
        lift_per_span_N_m=lift_per_span,
        inertial_load_per_span_N_m=inertial_load,
        shear_force_N=shear_force + point_shear,
        bending_moment_N_m=_integrate_from_tip(shear_force, s) + point_moment,
        torque_N_m=_integrate_from_tip(twisting_moment_per_span, s) + point_torque,
        point_loads=tuple(point_loads),
    )


def _point_load_effects(
    point_loads: Sequence[WingPointLoad], s: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shear [N], bending moment [N m] and torque [N m] from concentrated loads.

    Inputs: point loads; spanwise stations s [m], shape (M,), root to tip.
    Outputs: three arrays, shape (M,). Exact, since each is a step or a ramp.
    """
    shear, moment, torque = np.zeros_like(s), np.zeros_like(s), np.zeros_like(s)
    for point_load in point_loads:
        if point_load.s_m > s[-1]:
            raise ValueError(f"{point_load.name} at s = {point_load.s_m} m is beyond the tip")
        inboard = s < point_load.s_m
        # The load pushes down, so it reduces the upward shear and tip-up moment.
        shear[inboard] -= point_load.downward_force_N
        moment[inboard] -= point_load.downward_force_N * (point_load.s_m - s[inboard])
        torque[inboard] += point_load.nose_up_moment_N_m
    return shear, moment, torque


def elastic_axis_x_m(surface: LiftingSurface, s_m: float, chord_fraction: float) -> float:
    """Body-axis x [m] of a wing's elastic axis at spanwise position s_m [m].

    Use it as the mount x of a fuselage attached to the wing at s_m, so the
    mount moment is taken about the same axis as the wing torque.
    """
    return float(surface.origin[0] + surface.x_le_at(s_m) + chord_fraction * surface.chord_at(s_m))


def design_load_cases(
    limit_aero: LiftingLineResult,
    surface: LiftingSurface,
    airfoil: Airfoil,
    limit_load_factor: float,
    mass_per_span_kg_m: MassPerSpan,
    safety_factor: float = 1.5,
    elastic_axis_chord_fraction: float = 0.25,
    mass_chord_fraction: float = 0.40,
    point_loads: Sequence[WingPointLoad] = (),
    support_s_m: float | None = None,
) -> DesignLoadCases:
    """Loads on a lifting surface at the limit and ultimate load factors.

    The limit case is a real flight condition: ``limit_aero`` is the surface's
    lifting-line solution in the trimmed limit pull-up (see
    ``stability.trim_symmetric_flight``), so its lift and its distribution are
    right for that condition. The ultimate loads are the limit loads times the
    safety factor; they are a strength margin, not a flight condition, so the
    aerodynamics are not re-solved for them.

    How the surface is held:

    - ``support_s_m=None`` (a wing): the root (s = 0) carries whatever the
      lift, the surface's own mass and the point loads leave over.
    - ``support_s_m`` given (e.g. a tailplane on booms): the surface is held
      up at that station, which reacts the net force and torque of its side,
      so the centreline carries no shear. At 0 this is the same as None.

    Parameters
    ----------
    limit_aero : LiftingLineResult
        Lifting-line solution of this surface in the limit condition.
    surface : LiftingSurface
        The surface ``limit_aero`` was solved for.
    airfoil : Airfoil
        Section properties of the surface.
    limit_load_factor : float
        Limit load factor n_limit; scales the inertial load of the surface mass.
    mass_per_span_kg_m : function of s
        Surface-borne mass per unit span [kg/m], one side (see ``wing_loads``).
    safety_factor : float
        Ultimate load factor = limit_load_factor * safety_factor.
    elastic_axis_chord_fraction, mass_chord_fraction : float
        As for ``wing_loads``.
    point_loads : sequence of WingPointLoad
        Concentrated loads at the limit load factor (e.g. fuselage mounts from
        ``FuselageLoads.wing_point_load``); scaled with the safety factor for
        the ultimate case.
    support_s_m : float or None
        Spanwise station [m] where the surface is held up, or None.

    Returns
    -------
    DesignLoadCases
    """
    if not (limit_load_factor > 0 and safety_factor >= 1):
        raise ValueError("Need limit load factor > 0 and safety factor >= 1")

    def loads_with(extra_point_loads):
        return wing_loads(
            limit_aero, surface, airfoil, limit_load_factor, mass_per_span_kg_m,
            elastic_axis_chord_fraction=elastic_axis_chord_fraction,
            mass_chord_fraction=mass_chord_fraction,
            point_loads=[*point_loads, *extra_point_loads],
        )

    limit_loads = loads_with([])
    if support_s_m is not None and support_s_m > 0:
        # The support takes the force and torque the root would otherwise
        # carry, so the centreline is left with none. Root shear is the net
        # upward load on the side, so the support pushes down by that much
        # (or up, when the net load is downward).
        support = WingPointLoad(
            name="support",
            s_m=support_s_m,
            downward_force_N=limit_loads.root_shear_force_N,
            nose_up_moment_N_m=-limit_loads.root_torque_N_m,
        )
        limit_loads = loads_with([support])

    return DesignLoadCases(
        limit=limit_loads,
        ultimate=limit_loads.scaled(safety_factor),
        limit_aero=limit_aero,
    )


def wing_mass_centroid_x_m(
    surface: LiftingSurface, mass_per_span_kg_m: MassPerSpan, mass_chord_fraction: float = 0.40
) -> float:
    """Body-axis x [m] of the centre of mass of a surface's distributed mass.

    The mass at each station sits at ``mass_chord_fraction`` of the local
    chord. Inputs: the surface; its mass per span [kg/m] (function of s);
    the chordwise fraction. Output: x [m] (float).
    """
    s = np.linspace(0.0, surface.s[-1], 2001)
    mass_per_span = np.asarray(mass_per_span_kg_m(s), dtype=float)
    x_mass = surface.origin[0] + surface.x_le_at(s) + mass_chord_fraction * surface.chord_at(s)
    first_moment = _integrate_from_tip(mass_per_span * x_mass, s)[0]
    total = _integrate_from_tip(mass_per_span, s)[0]
    if total <= 0:
        raise ValueError(f"{surface.name}: the mass distribution has no mass")
    return float(first_moment / total)


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
