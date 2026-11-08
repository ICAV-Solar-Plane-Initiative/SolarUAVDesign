"""Aerodynamic analysis: lift distribution, drag and power required."""

from .airfoil import Airfoil
from .lifting_line import (
    LiftingLineResult,
    effective_lift_slope_per_rad,
    solve_lifting_line,
    solve_lifting_line_for_lift,
)

__all__ = [
    "Airfoil", "LiftingLineResult", "effective_lift_slope_per_rad",
    "solve_lifting_line", "solve_lifting_line_for_lift",
]
