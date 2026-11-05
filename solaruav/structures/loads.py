"""Spanwise shear force and bending moment for a cantilever wing.

The calculation is performed on one side of the aircraft, from wing root to
tip. Lift and mass per unit span are linearly interpolated between stations.
The wing is treated as a cantilever fixed at its root and free at its tip.

Inputs
------
- Spanwise stations from root to tip [m].
- Lift per unit span [N/m] and optional mass per unit span [kg/m].
- Limit load factor and safety factor.

Outputs
-------
- Limit and ultimate shear force [N] and bending moment [N m] at each station.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class WingLoads:
    """Shear-force and bending-moment distributions for one wing semispan.

    All array fields have one value per input spanwise station, ordered from
    root to tip. Positive net load acts upward; positive root bending moment
    is the moment caused by upward lift.

    Attributes
    ----------
    span_m : ndarray
        Spanwise station positions [m].
    net_load_per_span_limit_N_m : ndarray
        Upward net distributed load at the limit load factor [N/m].
    shear_force_limit_N : ndarray
        Internal shear force at the limit load factor [N].
    bending_moment_limit_N_m : ndarray
        Internal bending moment at the limit load factor [N m].
    shear_force_ultimate_N : ndarray
        Internal shear force at the ultimate load factor [N].
    bending_moment_ultimate_N_m : ndarray
        Internal bending moment at the ultimate load factor [N m].
    limit_load_factor : float
        Load factor used for the limit-load distributions.
    safety_factor : float
        Multiplier from limit load to ultimate load.
    """

    span_m: NDArray[np.float64]
    net_load_per_span_limit_N_m: NDArray[np.float64]
    shear_force_limit_N: NDArray[np.float64]
    bending_moment_limit_N_m: NDArray[np.float64]
    shear_force_ultimate_N: NDArray[np.float64]
    bending_moment_ultimate_N_m: NDArray[np.float64]
    limit_load_factor: float
    safety_factor: float


def calculate_wing_loads(
    span_m: ArrayLike,
    lift_per_span_N_m: ArrayLike,
    mass_per_span_kg_m: ArrayLike | float = 0.0,
    *,
    limit_load_factor: float = 1.0,
    safety_factor: float = 1.5,
    gravity_m_s2: float = 9.80665,
) -> WingLoads:
    """Calculate cantilever-wing shear force and bending moment.

    Lift and mass distributions are defined on one semispan at the supplied
    stations. Between stations, each distribution is assumed to vary linearly.
    Distributed weight is subtracted from lift. The limit-load distributions
    are then multiplied by ``safety_factor`` to obtain ultimate loads.

    Parameters
    ----------
    span_m : array-like of float, shape (N,)
        Spanwise positions [m], strictly increasing from root to tip, N >= 2.
        The first position is treated as the wing root.
    lift_per_span_N_m : array-like of float, shape (N,)
        Upward aerodynamic lift per unit span [N/m] on this semispan.
    mass_per_span_kg_m : array-like of float or float, optional
        Mass per unit span [kg/m] on this semispan. A scalar is applied at all
        stations. Defaults to zero.
    limit_load_factor : float, optional
        Positive limit load factor. Defaults to 1.
    safety_factor : float, optional
        Positive multiplier from limit to ultimate load. Defaults to 1.5.
    gravity_m_s2 : float, optional
        Gravitational acceleration [m/s^2]. Defaults to standard gravity.

    Returns
    -------
    WingLoads
        Spanwise net load, shear force and bending moment at limit and
        ultimate load factors.

    Raises
    ------
    ValueError
        If the input distributions have inconsistent lengths, invalid values,
        or span stations that are not strictly increasing.
    """
    span = np.asarray(span_m, dtype=float)
    lift = np.asarray(lift_per_span_N_m, dtype=float)
    if span.ndim != 1 or span.size < 2:
        raise ValueError("span_m must be a one-dimensional array with at least two stations")
    if lift.shape != span.shape:
        raise ValueError("lift_per_span_N_m must have the same shape as span_m")

    mass = np.asarray(mass_per_span_kg_m, dtype=float)
    if mass.ndim == 0:
        mass = np.full(span.shape, float(mass))
    if mass.shape != span.shape:
        raise ValueError("mass_per_span_kg_m must be a scalar or have the same shape as span_m")

    if not np.all(np.isfinite(span)) or not np.all(np.isfinite(lift)) or not np.all(np.isfinite(mass)):
        raise ValueError("span and load distributions must contain only finite values")
    if np.any(np.diff(span) <= 0):
        raise ValueError("span_m positions must be strictly increasing from root to tip")
    if np.any(mass < 0):
        raise ValueError("mass_per_span_kg_m cannot be negative")
    if not np.isfinite(limit_load_factor) or limit_load_factor <= 0:
        raise ValueError("limit_load_factor must be finite and positive")
    if not np.isfinite(safety_factor) or safety_factor <= 0:
        raise ValueError("safety_factor must be finite and positive")
    if not np.isfinite(gravity_m_s2) or gravity_m_s2 <= 0:
        raise ValueError("gravity_m_s2 must be finite and positive")

    net_load = limit_load_factor * (lift - mass * gravity_m_s2)
    shear = np.zeros_like(span)
    bending = np.zeros_like(span)

    for index in range(span.size - 2, -1, -1):
        segment_length = span[index + 1] - span[index]
        load_at_root = net_load[index]
        load_at_tip = net_load[index + 1]
        segment_shear = segment_length * (load_at_root + load_at_tip) / 2
        segment_moment_at_root = segment_length**2 * (load_at_root / 6 + load_at_tip / 3)

        shear[index] = shear[index + 1] + segment_shear
        bending[index] = (
            bending[index + 1]
            + segment_length * shear[index + 1]
            + segment_moment_at_root
        )

    return WingLoads(
        span_m=span.copy(),
        net_load_per_span_limit_N_m=net_load,
        shear_force_limit_N=shear,
        bending_moment_limit_N_m=bending,
        shear_force_ultimate_N=shear * safety_factor,
        bending_moment_ultimate_N_m=bending * safety_factor,
        limit_load_factor=float(limit_load_factor),
        safety_factor=float(safety_factor),
    )
