import numpy as np
import pytest

from solaruav import LiftingSurface, Station
from solaruav.aerodynamics import Airfoil, effective_lift_slope_per_rad, solve_lifting_line

AIRSPEED = 10.0  # m/s
DENSITY = 1.225  # kg/m^3


def elliptic_wing(span=8.0, root_chord=1.0, n_terms=30):
    """Wing whose stations sit on the solver's collocation points, so the chord
    seen by the solver is exactly elliptic."""
    theta = np.arange(1, n_terms + 1) * np.pi / (2 * n_terms)
    s = np.sort(span / 2 * np.cos(theta))
    s[0] = 0.0  # cos(pi/2) is only zero to rounding error
    chord = root_chord * np.sqrt(1 - (2 * s / span) ** 2)
    stations = [Station(s=si, x_le=0.0, chord=ci) for si, ci in zip(s, chord)]
    stations.append(Station(s=span / 2, x_le=0.0, chord=1e-6))
    return LiftingSurface("elliptic", stations)


# ---------------------------------------------------------------------- #
# Twist geometry
# ---------------------------------------------------------------------- #
def test_twist_is_linear_between_stations():
    w = LiftingSurface("w", [
        Station(s=0.0, x_le=0.0, chord=0.3, twist_deg=2.0),
        Station(s=1.0, x_le=0.0, chord=0.3, twist_deg=0.0),
        Station(s=2.0, x_le=0.0, chord=0.2, twist_deg=-3.0),
    ])
    assert w.twist_deg_at(0.5) == pytest.approx(1.0)
    assert w.twist_deg_at(1.0) == pytest.approx(0.0)
    assert w.twist_deg_at(1.5) == pytest.approx(-1.5)
    assert w.twist_deg_at(-1.5) == pytest.approx(-1.5)  # mirrored side


def test_panel_twist_is_cumulative_from_the_root():
    w = LiftingSurface.from_panels(
        "w", root_chord=0.3, root_twist_deg=2.0,
        panels=[{"span": 1.0, "twist_deg": -1.0}, {"span": 1.0, "twist_deg": -2.0}],
    )
    assert w.twist_deg.tolist() == pytest.approx([2.0, 1.0, -1.0])


def test_trapezoidal_twist():
    w = LiftingSurface.trapezoidal("w", span=4.0, root_chord=0.5, root_twist_deg=3.0, twist_deg=-4.0)
    assert w.twist_deg.tolist() == pytest.approx([3.0, -1.0])


# ---------------------------------------------------------------------- #
# Lifting line against known results
# ---------------------------------------------------------------------- #
def test_elliptic_wing_matches_theory():
    wing = elliptic_wing()
    alpha_deg, a0 = 5.0, 2 * np.pi
    result = solve_lifting_line(wing, alpha_deg, AIRSPEED, DENSITY, n_terms=30)

    aspect_ratio = wing.aspect_ratio
    expected_cl = a0 * np.radians(alpha_deg) / (1 + a0 / (np.pi * aspect_ratio))
    assert result.span_efficiency == pytest.approx(1.0, abs=2e-3)
    assert result.lift_coefficient == pytest.approx(expected_cl, rel=2e-2)
    assert result.induced_drag_coefficient == pytest.approx(
        result.lift_coefficient**2 / (np.pi * aspect_ratio), rel=1e-2
    )


def test_rectangular_wing_efficiency_matches_textbook():
    # Anderson, Fundamentals of Aerodynamics: delta is about 0.05 for a
    # rectangular wing of aspect ratio 6, so e = 1 / (1 + delta) is about 0.95.
    wing = LiftingSurface.trapezoidal("w", span=6.0, root_chord=1.0)
    result = solve_lifting_line(wing, 5.0, AIRSPEED, DENSITY, n_terms=40)
    assert 0.94 < result.span_efficiency < 0.97


def test_solution_converges_with_number_of_terms():
    wing = LiftingSurface.trapezoidal("w", span=6.0, root_chord=1.0, taper=0.4)
    coarse = solve_lifting_line(wing, 5.0, AIRSPEED, DENSITY, n_terms=40)
    fine = solve_lifting_line(wing, 5.0, AIRSPEED, DENSITY, n_terms=80)
    assert coarse.lift_coefficient == pytest.approx(fine.lift_coefficient, rel=1e-3)
    assert coarse.induced_drag_coefficient == pytest.approx(fine.induced_drag_coefficient, rel=1e-3)


def test_zero_lift_at_zero_lift_angle():
    wing = LiftingSurface.trapezoidal("w", span=5.0, root_chord=0.4, taper=0.6)
    result = solve_lifting_line(wing, -3.0, AIRSPEED, DENSITY, airfoil=Airfoil(alpha_zero_lift_deg=-3.0))
    assert result.lift_coefficient == pytest.approx(0.0, abs=1e-12)


def test_uniform_twist_is_the_same_as_extra_angle_of_attack():
    plain = LiftingSurface.trapezoidal("w", span=5.0, root_chord=0.4, taper=0.6)
    incidence = LiftingSurface.trapezoidal("w", span=5.0, root_chord=0.4, taper=0.6, root_twist_deg=2.0)
    a = solve_lifting_line(plain, 4.0, AIRSPEED, DENSITY)
    b = solve_lifting_line(incidence, 2.0, AIRSPEED, DENSITY)
    assert a.lift_coefficient == pytest.approx(b.lift_coefficient)
    assert a.induced_drag_coefficient == pytest.approx(b.induced_drag_coefficient)


def test_washout_unloads_the_tip():
    plain = LiftingSurface.trapezoidal("w", span=5.0, root_chord=0.4)
    washed = LiftingSurface.trapezoidal("w", span=5.0, root_chord=0.4, twist_deg=-4.0)
    a = solve_lifting_line(plain, 5.0, AIRSPEED, DENSITY)
    b = solve_lifting_line(washed, 5.0, AIRSPEED, DENSITY)
    near_tip = (a.s_m > 0.8 * 2.5) & (a.s_m < 2.5)  # the tip itself carries no lift in either case
    assert np.all(b.local_lift_coefficient[near_tip] < a.local_lift_coefficient[near_tip])
    assert b.lift_coefficient < a.lift_coefficient


def test_spanwise_lift_integrates_to_total_lift():
    wing = LiftingSurface.from_panels(
        "w", root_chord=0.32, panels=[{"span": 1.5}, {"span": 1.0, "taper": 0.6, "twist_deg": -2.0}],
    )
    result = solve_lifting_line(wing, 4.0, AIRSPEED, DENSITY, n_output=401)
    lift_per_span, s = result.lift_per_span_N_m, result.s_m
    integrated = 2 * np.sum(0.5 * (lift_per_span[1:] + lift_per_span[:-1]) * np.diff(s))
    assert integrated == pytest.approx(result.lift_N, rel=1e-3)


# ---------------------------------------------------------------------- #
# Pitching moment
# ---------------------------------------------------------------------- #
def test_moment_about_quarter_chord_is_section_cm0():
    # Unswept rectangular wing: every section's lift acts on the quarter-chord
    # line, so about that line only Cm0 remains.
    wing = LiftingSurface.trapezoidal("w", span=4.0, root_chord=0.5, origin=(1.0, 0.0, 0.0))
    result = solve_lifting_line(
        wing, 5.0, AIRSPEED, DENSITY, airfoil=Airfoil(cm0=-0.1), moment_reference_x_m=1.0 + 0.125
    )
    assert result.pitching_moment_coefficient == pytest.approx(-0.1, rel=1e-9)


def test_moment_about_leading_edge():
    # About the leading edge, lift at the quarter chord gives Cm = -CL / 4.
    wing = LiftingSurface.trapezoidal("w", span=4.0, root_chord=0.5)
    result = solve_lifting_line(wing, 5.0, AIRSPEED, DENSITY, n_output=401, moment_reference_x_m=0.0)
    assert result.pitching_moment_coefficient == pytest.approx(-result.lift_coefficient / 4, rel=1e-3)


# ---------------------------------------------------------------------- #
# Input checks
# ---------------------------------------------------------------------- #
def test_unsupported_surfaces_are_rejected():
    fin = LiftingSurface.trapezoidal("fin", span=0.4, root_chord=0.3, symmetric=False, orientation="vertical")
    with pytest.raises(ValueError):
        solve_lifting_line(fin, 0.0, AIRSPEED, DENSITY)
    split = LiftingSurface("split", [Station(s=0.2, x_le=0.0, chord=0.3), Station(s=1.0, x_le=0.0, chord=0.3)])
    with pytest.raises(ValueError):
        solve_lifting_line(split, 0.0, AIRSPEED, DENSITY)


# ---------------------------------------------------------------------- #
# Downwash and induced drag distributions
# ---------------------------------------------------------------------- #
def test_elliptic_wing_has_uniform_downwash():
    wing = elliptic_wing()
    result = solve_lifting_line(wing, 5.0, AIRSPEED, DENSITY)
    expected = AIRSPEED * result.lift_coefficient / (np.pi * wing.aspect_ratio)
    np.testing.assert_allclose(result.downwash_m_s, expected, rtol=1e-2)


def test_induced_drag_per_span_integrates_to_total():
    wing = LiftingSurface.from_panels(
        "w", root_chord=0.32, panels=[{"span": 1.5}, {"span": 1.0, "taper": 0.6, "twist_deg": -2.0}],
    )
    result = solve_lifting_line(wing, 4.0, AIRSPEED, DENSITY, n_output=801)
    drag_per_span, s = result.induced_drag_per_span_N_m, result.s_m
    integrated = 2 * np.sum(0.5 * (drag_per_span[1:] + drag_per_span[:-1]) * np.diff(s))
    assert integrated == pytest.approx(result.induced_drag_N, rel=2e-3)


# ---------------------------------------------------------------------- #
# Sweep correction (Helmbold / DATCOM)
# ---------------------------------------------------------------------- #
def lift_curve_slope(surface, **kwargs):
    """dCL/dalpha [1/rad] from two lifting-line solutions."""
    lift_at_zero = solve_lifting_line(surface, 0.0, AIRSPEED, DENSITY, n_terms=40, **kwargs).lift_coefficient
    lift_at_one = solve_lifting_line(surface, 1.0, AIRSPEED, DENSITY, n_terms=40, **kwargs).lift_coefficient
    return (lift_at_one - lift_at_zero) / np.radians(1.0)


def helmbold_lift_curve_slope(aspect_ratio, midchord_sweep_deg, section_slope=2 * np.pi):
    """DATCOM / Helmbold, incompressible."""
    kappa = section_slope / (2 * np.pi)
    tan_sweep = np.tan(np.radians(midchord_sweep_deg))
    return 2 * np.pi * aspect_ratio / (2 + np.sqrt(aspect_ratio**2 * (1 + tan_sweep**2) / kappa**2 + 4))


def test_unswept_surface_is_unaffected_by_the_correction():
    wing = LiftingSurface.trapezoidal("w", span=6.0, root_chord=0.6, taper=1.0)
    corrected = solve_lifting_line(wing, 4.0, AIRSPEED, DENSITY)
    uncorrected = solve_lifting_line(wing, 4.0, AIRSPEED, DENSITY, sweep_correction=False)
    assert corrected.lift_coefficient == pytest.approx(uncorrected.lift_coefficient, rel=1e-12)


def test_sweep_acts_as_a_reduced_section_slope():
    # A constant-chord wing swept 30 deg (mid-chord sweep 30 deg) behaves like
    # the unswept wing with section slope a0 cos 30.
    swept = LiftingSurface.trapezoidal("w", span=8.0, root_chord=1.0, sweep_le_deg=30.0)
    straight = LiftingSurface.trapezoidal("w", span=8.0, root_chord=1.0)
    reduced = Airfoil(lift_slope_per_rad=2 * np.pi * np.cos(np.radians(30.0)))
    a = solve_lifting_line(swept, 5.0, AIRSPEED, DENSITY)
    b = solve_lifting_line(straight, 5.0, AIRSPEED, DENSITY, airfoil=reduced)
    assert a.lift_coefficient == pytest.approx(b.lift_coefficient, rel=1e-12)


@pytest.mark.parametrize("aspect_ratio", [6.0, 10.0, 17.0])
@pytest.mark.parametrize("sweep_deg", [15.0, 30.0])
def test_sweep_reduces_lift_slope_as_helmbold_predicts(aspect_ratio, sweep_deg):
    swept = LiftingSurface.trapezoidal("w", span=aspect_ratio, root_chord=1.0, sweep_le_deg=sweep_deg)
    straight = LiftingSurface.trapezoidal("w", span=aspect_ratio, root_chord=1.0)
    ratio = lift_curve_slope(swept) / lift_curve_slope(straight)
    expected = helmbold_lift_curve_slope(aspect_ratio, sweep_deg) / helmbold_lift_curve_slope(aspect_ratio, 0.0)
    assert ratio == pytest.approx(expected, rel=1e-2)


def test_correction_can_be_switched_off():
    swept = LiftingSurface.trapezoidal("w", span=8.0, root_chord=1.0, sweep_le_deg=30.0)
    straight = LiftingSurface.trapezoidal("w", span=8.0, root_chord=1.0)
    assert lift_curve_slope(swept, sweep_correction=False) == pytest.approx(lift_curve_slope(straight), rel=1e-12)


def test_multi_panel_sweep_is_area_weighted():
    # Equal-area panels, one unswept and one swept 30 deg at mid-chord:
    # the factor is the mean of cos 0 and cos 30.
    wing = LiftingSurface.from_panels(
        "w", root_chord=0.5, panels=[{"span": 1.0}, {"span": 1.0, "sweep_le_deg": 30.0}],
    )
    expected = 2 * np.pi * (1 + np.cos(np.radians(30.0))) / 2
    assert effective_lift_slope_per_rad(wing, Airfoil()) == pytest.approx(expected)
