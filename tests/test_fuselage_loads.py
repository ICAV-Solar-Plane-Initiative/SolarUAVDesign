import numpy as np
import pytest

from solaruav import Fuselage, FuselageSection, LiftingSurface
from solaruav.aerodynamics import solve_lifting_line_for_lift
from solaruav.structures import (
    MassItem,
    WingPointLoad,
    chord_proportional_mass,
    fuselage_loads,
    required_fuselage_section_properties,
    wing_loads,
)
from solaruav.aerodynamics import Airfoil
from solaruav.structures.loads import STANDARD_GRAVITY_M_S2 as G

AIRSPEED, DENSITY = 12.0, 1.225


def boom(length=2.0, origin=(0.0, 0.0, 0.0), symmetric=False):
    """A constant-section body from x = 0 to x = length (local), placed at origin."""
    sections = [FuselageSection(0.0, 0.05, 0.05), FuselageSection(length, 0.05, 0.05)]
    return Fuselage("boom", sections, origin=origin, symmetric=symmetric)


# ---------------------------------------------------------------------- #
# Fuselage as a beam supported at its mount
# ---------------------------------------------------------------------- #
def test_tail_mass_cantilever():
    # Massless boom, mount at the nose end, mass m at the tail: M at the mount
    # = n g m L, mount force n g m, mount moment n g m L nose-up.
    loads = fuselage_loads(boom(2.0), [MassItem("tail", 0.5, x_m=2.0)], 0.0, mount_x_m=0.0, load_factor=3.0)
    weight = 3.0 * G * 0.5
    assert loads.mount_force_N == pytest.approx(weight)
    assert loads.mount_moment_N_m == pytest.approx(weight * 2.0)
    assert loads.max_bending_moment_N_m == pytest.approx(weight * 2.0)


def test_uniform_structure_mass_with_central_mount():
    # Uniform mass per length on a constant section, mount at the middle: each
    # half is a cantilever of length L/2 with M_mount = w (L/2)^2 / 2, and the
    # moments balance, so the mount moment is zero.
    length, mass = 2.0, 1.0
    loads = fuselage_loads(boom(length), [], mass, mount_x_m=length / 2, load_factor=1.0)
    load_per_length = G * mass / length
    assert loads.mount_force_N == pytest.approx(G * mass, rel=1e-6)
    assert loads.mount_moment_N_m == pytest.approx(0.0, abs=1e-9)
    assert loads.max_bending_moment_N_m == pytest.approx(load_per_length * (length / 2) ** 2 / 2, rel=1e-4)


def test_mount_reaction_balances_all_loads():
    items = [MassItem("motor", 0.3, -0.4), MassItem("battery", 1.0, 0.1), MassItem("tail", 0.2, 1.4)]
    body = boom(2.0, origin=(-0.5, 0.0, 0.0))
    loads = fuselage_loads(body, items, 0.4, mount_x_m=0.08, load_factor=2.5)
    total_mass = 0.4 + 0.3 + 1.0 + 0.2
    assert loads.total_mass_kg == pytest.approx(total_mass)
    assert loads.mount_force_N == pytest.approx(2.5 * G * total_mass, rel=1e-6)
    # Moment of the point loads about the mount, plus the distributed structure
    # (uniform here, centroid at the body's middle, x = 0.5).
    expected_moment = 2.5 * G * (sum(i.mass_kg * (i.x_m - 0.08) for i in items) + 0.4 * (0.5 - 0.08))
    assert loads.mount_moment_N_m == pytest.approx(expected_moment, rel=1e-5)


def test_mount_station_appears_twice_and_jumps_by_the_mount_moment():
    loads = fuselage_loads(boom(2.0), [MassItem("tail", 0.5, x_m=2.0)], 0.0, mount_x_m=0.5, load_factor=1.0)
    mount_indices = np.flatnonzero(loads.x_m == 0.5)
    assert len(mount_indices) == 2
    nose_side, tail_side = loads.bending_moment_N_m[mount_indices]
    assert tail_side - nose_side == pytest.approx(loads.mount_moment_N_m)


def test_mirrored_fuselage_mounts_at_its_spanwise_station():
    loads = fuselage_loads(boom(origin=(0.0, 0.75, 0.0), symmetric=True), [], 0.2, mount_x_m=0.1, load_factor=1.0)
    point_load = loads.wing_point_load()
    assert point_load.s_m == pytest.approx(0.75)
    assert point_load.downward_force_N == pytest.approx(G * 0.2, rel=1e-6)


def test_ultimate_scaling_and_required_section_modulus():
    limit = fuselage_loads(boom(2.0), [MassItem("tail", 0.5, x_m=2.0)], 0.0, mount_x_m=0.0, load_factor=3.0)
    ultimate = limit.scaled(1.5)
    assert ultimate.load_factor == pytest.approx(4.5)
    required = required_fuselage_section_properties(ultimate, allowable_stress_Pa=100e6)
    assert required.max_required_section_modulus_m3 == pytest.approx(4.5 * G * 0.5 * 2.0 / 100e6)


def test_items_outside_the_body_are_rejected():
    with pytest.raises(ValueError):
        fuselage_loads(boom(2.0), [MassItem("lost", 0.1, x_m=3.0)], 0.0, mount_x_m=0.0, load_factor=1.0)
    with pytest.raises(ValueError):
        fuselage_loads(boom(2.0), [], 0.0, mount_x_m=-1.0, load_factor=1.0)


# ---------------------------------------------------------------------- #
# Mount loads on the wing
# ---------------------------------------------------------------------- #
def wing_and_aero(lift_N=150.0):
    wing = LiftingSurface.trapezoidal("wing", span=4.0, root_chord=0.3)
    aero = solve_lifting_line_for_lift(wing, lift_N, AIRSPEED, DENSITY, n_output=401)
    return wing, aero


def test_point_load_steps_shear_moment_and_torque_inboard():
    wing, aero = wing_and_aero()
    plain = wing_loads(aero, wing, Airfoil(), 1.0, lambda s: np.zeros_like(s))
    mount = WingPointLoad("fuselage", s_m=0.8, downward_force_N=30.0, nose_up_moment_N_m=2.0)
    loaded = wing_loads(aero, wing, Airfoil(), 1.0, lambda s: np.zeros_like(s), point_loads=[mount])

    s = aero.s_m
    inboard, outboard = s < 0.8, s >= 0.8
    np.testing.assert_allclose((loaded.shear_force_N - plain.shear_force_N)[inboard], -30.0)
    np.testing.assert_allclose((loaded.bending_moment_N_m - plain.bending_moment_N_m)[inboard], -30.0 * (0.8 - s[inboard]))
    np.testing.assert_allclose((loaded.torque_N_m - plain.torque_N_m)[inboard], 2.0)
    np.testing.assert_allclose(loaded.bending_moment_N_m[outboard], plain.bending_moment_N_m[outboard])


def test_root_mount_has_no_effect_on_the_wing():
    wing, aero = wing_and_aero()
    plain = wing_loads(aero, wing, Airfoil(), 1.0, lambda s: np.zeros_like(s))
    central = wing_loads(aero, wing, Airfoil(), 1.0, lambda s: np.zeros_like(s),
                         point_loads=[WingPointLoad("fuselage", 0.0, 50.0, 3.0)])
    np.testing.assert_allclose(central.bending_moment_N_m, plain.bending_moment_N_m)


def test_twin_fuselages_carrying_all_non_wing_mass_leave_no_root_shear():
    # Lift = n g (wing mass + 2 x fuselage mass). With every kilogram on the wing
    # or on the two fuselages, nothing hangs at the centreline, so the root shear is zero.
    wing_mass, fuselage_mass, load_factor = 1.0, 1.5, 3.0
    wing, aero = wing_and_aero(lift_N=load_factor * G * (wing_mass + 2 * fuselage_mass))
    mount = WingPointLoad("fuselages", s_m=0.7, downward_force_N=load_factor * G * fuselage_mass)
    loads = wing_loads(aero, wing, Airfoil(), load_factor, chord_proportional_mass(wing, wing_mass),
                       point_loads=[mount])
    assert loads.root_shear_force_N == pytest.approx(0.0, abs=1e-5 * aero.lift_N)  # trapezoid error only
