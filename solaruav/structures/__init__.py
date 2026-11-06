"""Structural model: wing loads, stresses and stiffness."""

from .loads import DesignLoadCases, WingLoads, chord_proportional_mass, design_load_cases, wing_loads

__all__ = ["DesignLoadCases", "WingLoads", "chord_proportional_mass", "design_load_cases", "wing_loads"]
