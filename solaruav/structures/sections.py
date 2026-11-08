"""Spar cross-sections and their materials, for the wing beam model.

Purpose
-------
Describe what the wing beam is made of and the shape of its cross-section
along the span, and give the section properties beam theory needs: area,
second moment of area I and polar moment J.

The spar is a thin-walled circular tube (e.g. a carbon-fibre tube), the usual
main spar of a small solar aircraft. Its outer diameter and wall thickness
vary linearly from root to tip, so a constant tube is just equal root and tip
values. Another section type (a box spar, a D-box skin) can be added later as
a new class with the same methods.

Section properties of a circular tube, outer diameter D, inner d = D - 2t:

    A = pi (D^2 - d^2) / 4
    I = pi (D^4 - d^4) / 64      (bending about any diameter)
    J = 2 I                      (torsion of a closed circular tube)

Inputs
------
- Material: Young's modulus E and shear modulus G [Pa], density [kg/m^3],
  allowable direct and shear stresses [Pa].
- Tube outer diameter and wall thickness at the root and tip [m].

Outputs
-------
- Section properties at any fraction of the semispan: diameter [m], wall
  thickness [m], area [m^2], I [m^4], J [m^4], mass per span [kg/m].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Material:
    """A structural material, treated as isotropic for beam theory.

    For a fibre composite, use the effective properties of the laminate
    along the spar axis (e.g. from the tube manufacturer's data sheet).

    Attributes
    ----------
    name : str
        Name of the material, for reports.
    youngs_modulus_Pa : float
        Young's modulus E along the spar [Pa].
    shear_modulus_Pa : float
        Shear modulus G [Pa].
    density_kg_m3 : float
        Density [kg/m^3].
    allowable_stress_Pa : float
        Allowable direct (bending) stress [Pa]: the lower of the tension and
        compression allowables. For composites compression usually governs.
    allowable_shear_stress_Pa : float
        Allowable shear stress [Pa].
    """

    name: str
    youngs_modulus_Pa: float
    shear_modulus_Pa: float
    density_kg_m3: float
    allowable_stress_Pa: float
    allowable_shear_stress_Pa: float

    def __post_init__(self):
        values = [self.youngs_modulus_Pa, self.shear_modulus_Pa, self.density_kg_m3,
                  self.allowable_stress_Pa, self.allowable_shear_stress_Pa]
        if not all(value > 0 for value in values):
            raise ValueError(f"{self.name}: all material properties must be positive")


@dataclass(frozen=True)
class TubeSpar:
    """A thin-walled circular tube spar, tapering linearly from root to tip.

    Positions along the spar are given as a fraction of the semispan,
    span_fraction = s / (b/2), from 0 at the root to 1 at the tip, so the
    spar does not duplicate the wing's span.

    Attributes
    ----------
    material : Material
        What the tube is made of.
    root_outer_diameter_m, tip_outer_diameter_m : float
        Outer diameter at the root and tip [m].
    root_wall_thickness_m, tip_wall_thickness_m : float
        Wall thickness at the root and tip [m]. Must be less than half the
        outer diameter.

    Methods (each takes span_fraction, a float or array of any shape, and
    returns an array of the same shape)
    -------
    outer_diameter_m(span_fraction)    outer diameter D [m]
    wall_thickness_m(span_fraction)    wall thickness t [m]
    area_m2(span_fraction)             cross-sectional area A [m^2]
    second_moment_m4(span_fraction)    second moment of area I [m^4]
    polar_moment_m4(span_fraction)     polar moment J [m^4]
    mass_per_span_kg_m(span_fraction)  rho A [kg/m]
    """

    material: Material
    root_outer_diameter_m: float
    tip_outer_diameter_m: float
    root_wall_thickness_m: float
    tip_wall_thickness_m: float

    def __post_init__(self):
        for end in ("root", "tip"):
            diameter = getattr(self, f"{end}_outer_diameter_m")
            wall = getattr(self, f"{end}_wall_thickness_m")
            if not (diameter > 0 and 0 < wall < diameter / 2):
                raise ValueError(f"Tube {end}: need diameter > 0 and 0 < wall thickness < diameter / 2")

    def outer_diameter_m(self, span_fraction: float | np.ndarray) -> np.ndarray:
        """Outer diameter D [m], linear from root to tip."""
        return _linear(self.root_outer_diameter_m, self.tip_outer_diameter_m, span_fraction)

    def wall_thickness_m(self, span_fraction: float | np.ndarray) -> np.ndarray:
        """Wall thickness t [m], linear from root to tip."""
        return _linear(self.root_wall_thickness_m, self.tip_wall_thickness_m, span_fraction)

    def area_m2(self, span_fraction: float | np.ndarray) -> np.ndarray:
        """Cross-sectional area A = pi (D^2 - d^2) / 4 [m^2]."""
        outer, inner = self._diameters(span_fraction)
        return np.pi * (outer**2 - inner**2) / 4

    def second_moment_m4(self, span_fraction: float | np.ndarray) -> np.ndarray:
        """Second moment of area I = pi (D^4 - d^4) / 64 [m^4]."""
        outer, inner = self._diameters(span_fraction)
        return np.pi * (outer**4 - inner**4) / 64

    def polar_moment_m4(self, span_fraction: float | np.ndarray) -> np.ndarray:
        """Polar moment J = 2 I [m^4] (torsion constant of a circular tube)."""
        return 2 * self.second_moment_m4(span_fraction)

    def mass_per_span_kg_m(self, span_fraction: float | np.ndarray) -> np.ndarray:
        """Spar mass per unit span, rho A [kg/m]."""
        return self.material.density_kg_m3 * self.area_m2(span_fraction)

    def _diameters(self, span_fraction: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Outer and inner diameters [m] at span_fraction (each the same shape as it)."""
        outer = self.outer_diameter_m(span_fraction)
        return outer, outer - 2 * self.wall_thickness_m(span_fraction)


def _linear(root_value: float, tip_value: float, span_fraction: float | np.ndarray) -> np.ndarray:
    """Linear interpolation from root_value (fraction 0) to tip_value (fraction 1)."""
    span_fraction = np.asarray(span_fraction, dtype=float)
    if np.any((span_fraction < -1e-12) | (span_fraction > 1 + 1e-12)):
        raise ValueError("span_fraction must lie between 0 (root) and 1 (tip)")
    return root_value + (tip_value - root_value) * span_fraction
