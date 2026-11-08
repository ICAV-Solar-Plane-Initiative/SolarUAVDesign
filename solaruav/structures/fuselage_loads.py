"""Fuselage loads at a design load case, and the reaction at its wing mount.

Purpose
-------
Treat one fuselage body as a beam supported at its wing mount, carrying its
own structure and the items mounted in or on it (battery, payload, avionics,
tail surfaces) in a symmetric pull-up at load factor n, and the aerodynamic
force of the tail where the tail is attached. Find its shear force
and bending moment along its length, and the force and pitching moment it
puts into the wing at the mount.

Method
------
Every mass pulls down with n g m: the fuselage structure as a distributed
load q(x) = n g m'(x), with m'(x) proportional to the local perimeter of the
cross-section (i.e. to wetted area), and each mass item i as a point load
P_i = n g m_i at its x_i. An aerodynamic force (the body's share of the
trimmed tail lift L_t) acts at its x as a point load -L_t: an upward tail lift
pushes the boom up. The mount, at x_mount on the wing's elastic axis,
holds the body up. Each side of the mount is a cantilever, integrated from
its free end (nose or tail) towards the mount:

    shear force      V(x) = total downward load between x and the free end   [N]
    bending moment   M(x) = moment of that load about x                      [N m]

M is positive hogging (the free ends droop; upper surface in tension). At
the mount the shear and moment jump, so the mount station appears twice in
the output: first the nose-side value, then the tail-side value.

The body pushes down on the wing at the mount with

    force   F = integral of q dx + sum P_i                   (its weight x n, less any tail lift)
    moment  M_mount = integral of q (x - x_mount) dx + sum P_i (x_i - x_mount)

positive nose-up: load aft of the mount tips the body tail-down, nose-up.
F and M_mount go into the wing loads as a ``WingPointLoad``.

Assumptions: symmetric pull-up, so no side loads or torsion; one mount per
body, rigid; the fuselage's own aerodynamic forces are neglected, and the
tail force acts at a point (the tail's aerodynamic centre). The loads are for
one body; a mirrored ``Fuselage`` describes two identical bodies with the same
loads.

Inputs
------
- A ``Fuselage`` (geometry from ``solaruav.geometry``).
- Mass items on one body: name, mass [kg], body-axis x [m].
- Aerodynamic point forces on one body (its share of the tail lift): name,
  upward force [N] at the load case, body-axis x [m].
- Structural mass of one body [kg].
- Mount position: body-axis x [m] (the wing elastic axis there) and the
  spanwise station of the mount on the wing [m].
- Load factor (float).

Outputs
-------
- A ``FuselageLoads``: shear force [N] and bending moment [N m] at stations
  along the body (arrays), and the mount force [N] and moment [N m].
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from ..geometry.fuselage import Fuselage
from .loads import STANDARD_GRAVITY_M_S2, WingPointLoad


@dataclass(frozen=True)
class MassItem:
    """A concentrated mass carried by one fuselage body.

    Attributes
    ----------
    name : str
        What it is, e.g. "battery".
    mass_kg : float
        Mass [kg]. Must be positive.
    x_m : float
        Body-axis x of its centre of mass [m]. Must lie within the body.
    """

    name: str
    mass_kg: float
    x_m: float

    def __post_init__(self):
        if not self.mass_kg > 0:
            raise ValueError(f"{self.name}: mass must be positive")


@dataclass(frozen=True)
class PointForce:
    """An aerodynamic point force on one fuselage body, e.g. its share of the tail lift.

    The value is at the load case it was computed for (it is not scaled by
    the load factor, but is scaled with the other loads by ``FuselageLoads.scaled``).

    Attributes
    ----------
    name : str
        What applies it, e.g. "htail".
    upward_force_N : float
        Force on the body [N], positive up (a tail download is negative).
    x_m : float
        Body-axis x where it acts [m]. Must lie within the body.
    """

    name: str
    upward_force_N: float
    x_m: float


@dataclass(frozen=True)
class FuselageLoads:
    """Loads along one fuselage body at one load case.

    Arrays have shape (n_stations,) and run from nose to tail in body-axis x;
    the mount station appears twice (nose-side value, then tail-side value).

    Attributes
    ----------
    fuselage_name : str
        Name of the fuselage.
    load_factor : float
        Load factor n of this case.
    mount_x_m : float
        Body-axis x of the wing mount [m].
    mount_s_m : float
        Spanwise station of the mount on the wing [m] (0 for a central fuselage).
    x_m : ndarray
        Stations along the body, body-axis x [m].
    shear_force_N : ndarray
        Downward load between each station and its free end [N].
    bending_moment_N_m : ndarray
        Bending moment [N m], positive hogging.
    mount_force_N : float
        Downward force the body puts on the wing at the mount [N].
    mount_moment_N_m : float
        Pitching moment the body puts on the wing at the mount [N m], nose-up positive.
    mass_items : tuple of MassItem
        The concentrated masses included.
    structure_mass_kg : float
        Distributed structural mass of the body [kg].
    point_forces : tuple of PointForce
        Aerodynamic point forces included, at this case's load level.

    Properties
    ----------
    total_mass_kg : float, structure plus items [kg]
    max_bending_moment_N_m : float, largest |M| [N m]

    Methods
    -------
    scaled(factor)
        A copy with every load multiplied by ``factor`` (e.g. the safety factor).
    wing_point_load()
        The mount reaction as a ``WingPointLoad`` for ``structures.loads``.
    """

    fuselage_name: str
    load_factor: float
    mount_x_m: float
    mount_s_m: float
    x_m: np.ndarray
    shear_force_N: np.ndarray
    bending_moment_N_m: np.ndarray
    mount_force_N: float
    mount_moment_N_m: float
    mass_items: tuple[MassItem, ...]
    structure_mass_kg: float
    point_forces: tuple[PointForce, ...] = ()

    @property
    def total_mass_kg(self) -> float:
        return self.structure_mass_kg + sum(item.mass_kg for item in self.mass_items)

    @property
    def max_bending_moment_N_m(self) -> float:
        return float(np.max(np.abs(self.bending_moment_N_m)))

    def scaled(self, factor: float) -> "FuselageLoads":
        """A copy with every load (and the load factor) multiplied by ``factor`` (float)."""
        return replace(
            self,
            load_factor=self.load_factor * factor,
            shear_force_N=self.shear_force_N * factor,
            bending_moment_N_m=self.bending_moment_N_m * factor,
            mount_force_N=self.mount_force_N * factor,
            mount_moment_N_m=self.mount_moment_N_m * factor,
            point_forces=tuple(replace(force, upward_force_N=force.upward_force_N * factor)
                               for force in self.point_forces),
        )

    def wing_point_load(self) -> WingPointLoad:
        """The mount reaction, as a load on the wing at ``mount_s_m``."""
        return WingPointLoad(
            name=self.fuselage_name,
            s_m=self.mount_s_m,
            downward_force_N=self.mount_force_N,
            nose_up_moment_N_m=self.mount_moment_N_m,
        )


def fuselage_loads(
    fuselage: Fuselage,
    mass_items: Sequence[MassItem],
    structure_mass_kg: float,
    mount_x_m: float,
    load_factor: float,
    point_forces: Sequence[PointForce] = (),
    n_points: int = 201,
) -> FuselageLoads:
    """Shear force, bending moment and mount reaction of one fuselage body.

    Parameters
    ----------
    fuselage : Fuselage
        The body's geometry. Its mount station on the wing is taken as
        |origin y| (0 for a central fuselage, y_f for a mirrored pair).
    mass_items : sequence of MassItem
        Concentrated masses on this body (one body of a mirrored pair).
    structure_mass_kg : float
        Distributed structural mass of this body [kg], spread in proportion
        to the cross-section perimeter. May be 0.
    mount_x_m : float
        Body-axis x of the wing mount [m]: the wing's elastic axis at the
        mount station (see ``loads.elastic_axis_x_m``).
    load_factor : float
        Load factor n.
    point_forces : sequence of PointForce
        Aerodynamic point forces on this body at this load case, e.g. its
        share of the trimmed tail lift.
    n_points : int
        Number of evenly spaced stations along the body (item and mount
        positions are added to these).

    Returns
    -------
    FuselageLoads
    """
    nose_x, tail_x = fuselage.origin[0] + fuselage.x[0], fuselage.origin[0] + fuselage.x[-1]
    positions = [("mount", mount_x_m)] + [(item.name, item.x_m) for item in mass_items]
    positions += [(force.name, force.x_m) for force in point_forces]
    for position_name, x in positions:
        if not nose_x <= x <= tail_x:
            raise ValueError(f"{position_name} at x = {x} m is outside {fuselage.name} ({nose_x} to {tail_x} m)")
    if structure_mass_kg < 0:
        raise ValueError("Structural mass must be >= 0")

    weight_factor = load_factor * STANDARD_GRAVITY_M_S2
    # Every concentrated load as (downward force [N], x [m]): weights of the
    # items, and the aerodynamic forces with their sign flipped.
    point_loads = [(weight_factor * item.mass_kg, item.x_m) for item in mass_items]
    point_loads += [(-force.upward_force_N, force.x_m) for force in point_forces]
    item_positions = [x for _, x in point_loads]
    nose_side = np.unique(np.concatenate([np.linspace(nose_x, mount_x_m, n_points // 2),
                                          [p for p in item_positions if p <= mount_x_m]]))
    tail_side = np.unique(np.concatenate([np.linspace(mount_x_m, tail_x, n_points - n_points // 2),
                                          [p for p in item_positions if p >= mount_x_m]]))
    mass_per_length = _structure_mass_per_length(fuselage, structure_mass_kg)

    # Each side as a cantilever from its free end. The tail side is mirrored
    # (distance measured from the tail) so the same routine handles both.
    nose_shear, nose_moment = _cantilever_from_free_end(
        nose_side - nose_x, weight_factor * mass_per_length(nose_side),
        [(force, x - nose_x) for force, x in point_loads if x < mount_x_m],
    )
    tail_reversed = tail_side[::-1]
    tail_shear, tail_moment = _cantilever_from_free_end(
        tail_x - tail_reversed, weight_factor * mass_per_length(tail_reversed),
        [(force, tail_x - x) for force, x in point_loads if x > mount_x_m],
    )
    loads_at_mount_N = sum(force for force, x in point_loads if x == mount_x_m)

    return FuselageLoads(
        fuselage_name=fuselage.name,
        load_factor=float(load_factor),
        mount_x_m=float(mount_x_m),
        mount_s_m=abs(float(fuselage.origin[1])),
        x_m=np.concatenate([nose_side, tail_side]),
        shear_force_N=np.concatenate([nose_shear, tail_shear[::-1]]),
        bending_moment_N_m=np.concatenate([nose_moment, tail_moment[::-1]]),
        mount_force_N=float(nose_shear[-1] + tail_shear[-1] + loads_at_mount_N),
        mount_moment_N_m=float(tail_moment[-1] - nose_moment[-1]),
        mass_items=tuple(mass_items),
        structure_mass_kg=float(structure_mass_kg),
        point_forces=tuple(point_forces),
    )


def structure_centroid_x_m(fuselage: Fuselage) -> float:
    """Body-axis x [m] of the centre of the fuselage's distributed structural mass.

    The structure is spread in proportion to the cross-section perimeter, as
    in ``fuselage_loads``. Input: a Fuselage. Output: x [m] (float).
    """
    x = np.linspace(fuselage.origin[0] + fuselage.x[0], fuselage.origin[0] + fuselage.x[-1], 2001)
    mass_per_length = _structure_mass_per_length(fuselage, 1.0)(x)
    first_moment = np.sum(0.5 * (mass_per_length[1:] * x[1:] + mass_per_length[:-1] * x[:-1]) * np.diff(x))
    total = np.sum(0.5 * (mass_per_length[1:] + mass_per_length[:-1]) * np.diff(x))
    return float(first_moment / total)


def _structure_mass_per_length(fuselage: Fuselage, structure_mass_kg: float):
    """Structural mass per length [kg/m] in proportion to the section perimeter.

    Returns a function of body-axis x [m] (array in, array of the same shape
    out) that integrates over the body to ``structure_mass_kg``.
    """
    x0 = fuselage.origin[0]

    def perimeter(x_body):
        x_local = np.asarray(x_body) - x0
        return 2 * (fuselage.width_at(x_local) + fuselage.height_at(x_local))

    fine_x = np.linspace(x0 + fuselage.x[0], x0 + fuselage.x[-1], 2001)
    fine_perimeter = perimeter(fine_x)
    total_perimeter_area = np.sum(0.5 * (fine_perimeter[1:] + fine_perimeter[:-1]) * np.diff(fine_x))
    scale = structure_mass_kg / total_perimeter_area if total_perimeter_area > 0 else 0.0
    return lambda x_body: scale * perimeter(x_body)


def _cantilever_from_free_end(
    distance: np.ndarray, load_per_length: np.ndarray, point_loads: list[tuple[float, float]]
) -> tuple[np.ndarray, np.ndarray]:
    """Shear and moment of a cantilever, measured from its free end.

    Inputs: distance from the free end [m], increasing, shape (M,); downward
    distributed load there [N/m], shape (M,); point loads as (force [N],
    distance from the free end [m]).
    Outputs: shear [N] and moment [N m] at each distance, shape (M,): the load
    between the free end and that point, and its moment about that point.
    Point loads are added exactly; the distributed load by the trapezoidal rule.
    """
    strips = 0.5 * (load_per_length[1:] + load_per_length[:-1]) * np.diff(distance)
    shear = np.concatenate([[0.0], np.cumsum(strips)])
    moment_strips = 0.5 * (shear[1:] + shear[:-1]) * np.diff(distance)
    moment = np.concatenate([[0.0], np.cumsum(moment_strips)])
    for force_N, position in point_loads:
        beyond = distance > position
        shear[beyond] += force_N
        moment[beyond] += force_N * (distance[beyond] - position)
    return shear, moment
