"""Centre of gravity.

Purpose
-------
Find the aircraft's centre of gravity from its mass breakdown. For now only
the x position is needed (for trim in symmetric flight).

Inputs
------
- Masses [kg] and their body-axis x positions [m] (equal-length sequences).

Outputs
-------
- CG body-axis x [m].
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def centre_of_gravity_x_m(masses_kg: Sequence[float], positions_x_m: Sequence[float]) -> float:
    """Mass-weighted mean x position [m].

    Inputs: masses [kg] and body-axis x of each [m], sequences of the same length N.
    Output: CG x [m] (float).
    """
    masses_kg, positions_x_m = np.asarray(masses_kg, dtype=float), np.asarray(positions_x_m, dtype=float)
    if masses_kg.shape != positions_x_m.shape or masses_kg.size == 0:
        raise ValueError("Need one position per mass, and at least one mass")
    if np.any(masses_kg < 0) or masses_kg.sum() <= 0:
        raise ValueError("Masses must be >= 0 with a positive total")
    return float(np.sum(masses_kg * positions_x_m) / masses_kg.sum())
