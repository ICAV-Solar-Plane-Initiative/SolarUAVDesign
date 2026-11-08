import numpy as np
import pytest

from solaruav import Fuselage, FuselageSection, LiftingSurface
from solaruav.aerodynamics import Airfoil
from solaruav.mass import centre_of_gravity_x_m
from solaruav.stability import trim_symmetric_flight
from solaruav.structures import (
    MassItem,
    PointForce,
    chord_proportional_mass,
    fuselage_loads,
    structure_centroid_x_m,
    wing_mass_centroid_x_m,
)
from solaruav.structures.loads import STANDARD_GRAVITY_M_S2 as G

AIRSPEED, DENSITY = 12.0, 1.225
WING_CHORD, TAIL_CHORD, TAIL_X = 0.3, 0.15, 1.5


def rectangular_pair():
    """Untwisted, unswept rectangular wing and tail: each surface's lift acts on
    its quarter-chord line, at x = c/4 (wing) and TAIL_X + c_t/4 (tail)."""
    wing = LiftingSurface.trapezoidal("wing", span=4.0, root_chord=WING_CHORD)
    tail = LiftingSurface.trapezoidal("htail", span=1.0, root_chord=TAIL_CHORD, origin=(TAIL_X, 0.0, 0.0))
    return wing, tail


def trim(cg_x_m, load_factor=2.0, wing_airfoil=Airfoil(), mass_kg=4.0):
    wing, tail = rectangular_pair()
    return trim_symmetric_flight(wing, wing_airfoil, tail, Airfoil(), mass_kg, cg_x_m,
                                 load_factor, AIRSPEED, DENSITY)


# ---------------------------------------------------------------------- #
# Trim
# ---------------------------------------------------------------------- #
def test_lifts_add_up_and_moment_is_zero():
    result = trim(cg_x_m=0.05, wing_airfoil=Airfoil(alpha_zero_lift_deg=-3.0, cm0=-0.1))
    assert result.wing_lift_N + result.tail_lift_N == pytest.approx(2.0 * 4.0 * G)
    assert result.pitching_moment_N_m == pytest.approx(0.0, abs=1e-9)


def test_cg_at_the_wing_quarter_chord_needs_no_tail_load():
    # Symmetric sections and lift on the quarter-chord line: about that line
    # the wing has no moment, so the tail carries nothing.
    result = trim(cg_x_m=WING_CHORD / 4)
    assert result.tail_lift_N == pytest.approx(0.0, abs=1e-9)
    assert result.wing_lift_N == pytest.approx(result.required_lift_N)


def test_lever_rule_with_cg_ahead_of_the_wing():
    # Lifts act at x_w = c/4 and x_t = TAIL_X + c_t/4. Moments about the CG:
    # L_w (x_w - cg) + L_t (x_t - cg) = 0 with L_w + L_t = n W, so
    # L_t = -n W (x_w - cg) / (x_t - x_w): a download when the CG is ahead.
    cg = 0.0
    result = trim(cg_x_m=cg)
    x_wing, x_tail = WING_CHORD / 4, TAIL_X + TAIL_CHORD / 4
    expected_tail_lift = -result.required_lift_N * (x_wing - cg) / (x_tail - x_wing)
    assert result.tail_lift_N == pytest.approx(expected_tail_lift, rel=1e-3)
    assert result.tail_lift_N < 0


def test_nose_down_section_moment_is_trimmed_by_a_download():
    plain = trim(cg_x_m=WING_CHORD / 4)
    cambered = trim(cg_x_m=WING_CHORD / 4, wing_airfoil=Airfoil(cm0=-0.1))
    assert plain.tail_lift_N == pytest.approx(0.0, abs=1e-9)
    assert cambered.tail_lift_N < 0


# ---------------------------------------------------------------------- #
# Centre of gravity
# ---------------------------------------------------------------------- #
def test_centre_of_gravity():
    assert centre_of_gravity_x_m([1.0, 3.0], [0.0, 2.0]) == pytest.approx(1.5)
    with pytest.raises(ValueError):
        centre_of_gravity_x_m([1.0], [0.0, 1.0])


def test_mass_centroids():
    wing, _ = rectangular_pair()
    assert wing_mass_centroid_x_m(wing, chord_proportional_mass(wing, 1.0), 0.4) == pytest.approx(0.4 * WING_CHORD)
    box = Fuselage("box", [FuselageSection(0.0, 0.1, 0.1), FuselageSection(2.0, 0.1, 0.1)], origin=(-0.5, 0.0, 0.0))
    assert structure_centroid_x_m(box) == pytest.approx(0.5)


# ---------------------------------------------------------------------- #
# Tail force on the boom
# ---------------------------------------------------------------------- #
def test_tail_force_enters_the_fuselage_reaction():
    boom = Fuselage("boom", [FuselageSection(0.0, 0.05, 0.05), FuselageSection(2.0, 0.05, 0.05)])
    items = [MassItem("tail", 0.3, x_m=1.9)]
    without = fuselage_loads(boom, items, 0.0, mount_x_m=0.1, load_factor=3.0)
    tail_lift = -4.0  # a download
    with_tail = fuselage_loads(boom, items, 0.0, mount_x_m=0.1, load_factor=3.0,
                               point_forces=[PointForce("htail", tail_lift, x_m=1.9)])
    assert with_tail.mount_force_N == pytest.approx(without.mount_force_N - tail_lift)
    assert with_tail.mount_moment_N_m == pytest.approx(without.mount_moment_N_m - tail_lift * (1.9 - 0.1))
    assert with_tail.scaled(1.5).point_forces[0].upward_force_N == pytest.approx(1.5 * tail_lift)
    assert np.all(with_tail.bending_moment_N_m >= without.bending_moment_N_m - 1e-12)
