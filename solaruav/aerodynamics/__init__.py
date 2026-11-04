"""Aerodynamic analysis: lift distribution, drag and power required."""

from .airfoil import Airfoil
from .lifting_line import LiftingLineResult, solve_lifting_line

__all__ = ["Airfoil", "LiftingLineResult", "solve_lifting_line"]
