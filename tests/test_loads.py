import numpy as np
import pytest

from solaruav import LiftingSurface, Station
from solaruav.aerodynamics import Airfoil, solve_lifting_line, solve_lifting_line_for_lift
from solaruav.structures import chord_proportional_mass, design_load_cases, wing_loads
from solaruav.structures.loads import STANDARD_GRAVITY_M_S2

AIRSPEED = 12.0  # m/s
DENSITY = 1.225  # kg/m^3


def no_mass(s):
    return np.zeros_like(s)


def elliptic_wing(span=8.0, root_chord=1.0, n_terms=30):
    """Wing whose stations sit on the solver's collocation points, so the chord
    seen by the solver is exactly elliptic (as in test_lifting_line)."""
    theta = np.arange(1, n_terms + 1) * np.pi / (2 * n_terms)
    s = np.sort(span / 2 * np.cos(theta))
    s[0] = 0.0
    chord = root_chord * np.sqrt(1 - (2 * s / span) ** 2)
    stations = [Station(s=si, x_le=0.0, chord=ci) for si, ci in zip(s, chord)]
    stations.append(Station(s=span / 2, x_le=0.0, chord=1e-6))
    return LiftingSurface("elliptic", stations)


def reference_wing():
    return LiftingSurface.from_panels(
        "wing", root_chord=0.32,
        panels=[{"span": 1.5, "taper": 1.0}, {"span": 1.0, "taper": 0.6, "twist_deg": -2.0}],
    )


# ---------------------------------------------------------------------- #
# Lift-targeted lifting line
# ---------------------------------------------------------------------- #
def test_solve_for_lift_gives_the_required_lift():
    wing = reference_wing()
    result = solve_lifting_line_for_lift(wing, 120.0, AIRSPEED, DENSITY, airfoil=Airfoil(alpha_zero_lift_deg=-3.0))
    assert result.lift_N == pytest.approx(120.0, rel=1e-10)
    direct = solve_lifting_line(wing, result.alpha_deg, AIRSPEED, DENSITY, airfoil=Airfoil(alpha_zero_lift_deg=-3.0))
    assert direct.lift_N == pytest.approx(120.0, rel=1e-10)


# ---------------------------------------------------------------------- #
# Loads against closed-form results
# ---------------------------------------------------------------------- #
def test_loads_vanish_at_the_tip():
    wing = reference_wing()
    aero = solve_lifting_line(wing, 5.0, AIRSPEED, DENSITY)
    loads = wing_loads(aero, wing, Airfoil(cm0=-0.1), 2.0, chord_proportional_mass(wing, 1.0))
    assert loads.shear_force_N[-1] == 0.0
    assert loads.bending_moment_N_m[-1] == 0.0
    assert loads.torque_N_m[-1] == 0.0


def test_root_shear_is_half_the_net_lift():
    wing = reference_wing()
    wing_mass_kg, load_factor = 1.5, 3.0
    aero = solve_lifting_line(wing, 6.0, AIRSPEED, DENSITY, n_output=401)
    loads = wing_loads(aero, wing, Airfoil(), load_factor, chord_proportional_mass(wing, wing_mass_kg))
    expected = (aero.lift_N - load_factor * STANDARD_GRAVITY_M_S2 * wing_mass_kg) / 2
    assert loads.root_shear_force_N == pytest.approx(expected, rel=1e-3)


def test_elliptic_root_bending_moment():
    # Elliptic loading: the lift on one side acts at s = 4 (b/2) / (3 pi).
    wing = elliptic_wing()
    aero = solve_lifting_line(wing, 5.0, AIRSPEED, DENSITY, n_output=801)
    loads = wing_loads(aero, wing, Airfoil(), 1.0, no_mass)
    expected = (aero.lift_N / 2) * 4 * (wing.span / 2) / (3 * np.pi)
    assert loads.root_bending_moment_N_m == pytest.approx(expected, rel=2e-3)


def test_uniform_weight_bending_moment():
    # A uniform weight w per span on a cantilever of length l gives M_root = -w l^2 / 2.
    wing = LiftingSurface.trapezoidal("w", span=4.0, root_chord=0.4)
    aero = solve_lifting_line(wing, 0.0, AIRSPEED, DENSITY, n_output=401)  # no lift at zero alpha
    mass_per_span = 0.5  # kg/m
    loads = wing_loads(aero, wing, Airfoil(), 2.0, lambda s: np.full_like(s, mass_per_span))
    weight_per_span = 2.0 * STANDARD_GRAVITY_M_S2 * mass_per_span
    assert loads.root_bending_moment_N_m == pytest.approx(-weight_per_span * 2.0**2 / 2, rel=1e-4)


def test_torque_about_quarter_chord_is_section_cm0_only():
    # Lift and mass both on the quarter-chord line: only Cm0 q c^2 twists the wing,
    # and its integral over one side is Cm0 q S MAC / 2.
    wing = reference_wing()
    airfoil = Airfoil(cm0=-0.1)
    aero = solve_lifting_line(wing, 5.0, AIRSPEED, DENSITY, airfoil=airfoil, n_output=801)
    loads = wing_loads(aero, wing, airfoil, 1.0, chord_proportional_mass(wing, 1.0),
                       elastic_axis_chord_fraction=0.25, mass_chord_fraction=0.25)
    expected = -0.1 * aero.dynamic_pressure_Pa * wing.area * wing.mac / 2
    assert loads.root_torque_N_m == pytest.approx(expected, rel=1e-3)


def test_lift_ahead_of_the_axis_twists_nose_up():
    # Rectangular wing, axis at mid-chord: lift acts c/4 ahead of it.
    wing = LiftingSurface.trapezoidal("w", span=4.0, root_chord=0.5)
    aero = solve_lifting_line(wing, 5.0, AIRSPEED, DENSITY, n_output=401)
    loads = wing_loads(aero, wing, Airfoil(), 1.0, no_mass, elastic_axis_chord_fraction=0.5)
    assert loads.root_torque_N_m == pytest.approx((aero.lift_N / 2) * 0.25 * 0.5, rel=1e-3)


def test_chord_proportional_mass_integrates_to_total():
    wing = reference_wing()
    s = np.linspace(0, wing.span / 2, 2001)
    m = chord_proportional_mass(wing, 2.0)(s)
    assert 2 * np.sum(0.5 * (m[1:] + m[:-1]) * np.diff(s)) == pytest.approx(2.0, rel=1e-6)


# ---------------------------------------------------------------------- #
# Design load cases
# ---------------------------------------------------------------------- #
def test_design_cases_lift_and_safety_factor():
    wing = reference_wing()
    cases = design_load_cases(
        wing, Airfoil(alpha_zero_lift_deg=-3.0, cm0=-0.08), aircraft_mass_kg=5.0,
        mass_per_span_kg_m=chord_proportional_mass(wing, 1.8),
        airspeed_m_s=14.0, air_density_kg_m3=DENSITY, limit_load_factor=3.0, safety_factor=1.5,
    )
    assert cases.limit_aero.lift_N == pytest.approx(3.0 * 5.0 * STANDARD_GRAVITY_M_S2)
    assert cases.ultimate.load_factor == pytest.approx(4.5)
    np.testing.assert_allclose(cases.ultimate.bending_moment_N_m, 1.5 * cases.limit.bending_moment_N_m)
    np.testing.assert_allclose(cases.ultimate.torque_N_m, 1.5 * cases.limit.torque_N_m)


def test_mismatched_surface_is_rejected():
    wing = reference_wing()
    other = LiftingSurface.trapezoidal("htail", span=1.0, root_chord=0.2)
    aero = solve_lifting_line(other, 2.0, AIRSPEED, DENSITY)
    with pytest.raises(ValueError):
        wing_loads(aero, wing, Airfoil(), 1.0, no_mass)
