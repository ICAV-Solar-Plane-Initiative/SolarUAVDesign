"""Two-dimensional airfoil section properties for the lifting-line model.

Purpose
-------
Hold the thin-airfoil properties the lifting-line solver needs for a surface's
section: lift curve slope, zero-lift angle and zero-lift pitching moment.
For now one set of properties applies along the whole span of a surface.

Inputs
------
- Lift curve slope a0 [1/rad] (2 pi for a thin airfoil).
- Zero-lift angle of attack [deg] (negative for a cambered airfoil).
- Zero-lift pitching moment coefficient Cm0 about the quarter chord
  (negative for a conventionally cambered airfoil).

Outputs
-------
- An ``Airfoil`` object passed to ``lifting_line.solve_lifting_line``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Airfoil:
    """Section properties of an airfoil, valid in the linear lift range.

    Attributes
    ----------
    lift_slope_per_rad : float
        Section lift curve slope dCl/dalpha [1/rad]. Must be positive.
        Defaults to the thin-airfoil value, 2 pi.
    alpha_zero_lift_deg : float
        Section angle of attack at zero lift [deg]. Defaults to 0 (symmetric section).
    cm0 : float
        Section pitching moment coefficient about the quarter chord,
        positive nose-up. Defaults to 0 (symmetric section).

    Methods
    -------
    alpha_zero_lift_rad
        Zero-lift angle in radians (float).
    """

    lift_slope_per_rad: float = 2 * np.pi
    alpha_zero_lift_deg: float = 0.0
    cm0: float = 0.0

    def __post_init__(self):
        if not self.lift_slope_per_rad > 0:
            raise ValueError(f"Lift curve slope must be positive, got {self.lift_slope_per_rad}")

    @property
    def alpha_zero_lift_rad(self) -> float:
        """Zero-lift angle of attack [rad]."""
        return float(np.radians(self.alpha_zero_lift_deg))
