import csv

import numpy as np
import pytest

from solaruav.structures import (
    DesignLoadCases,
    WingLoads,
    required_section_properties,
    write_section_properties_csv,
)

SEMISPAN = 2.0  # m
N_POINTS = 401


def make_loads(shear, moment, torque, load_factor=1.0, name="wing"):
    """WingLoads with given V, M, T functions of s, on s from 0 to SEMISPAN."""
    s = np.linspace(0.0, SEMISPAN, N_POINTS)
    zeros = np.zeros_like(s)
    return WingLoads(
        surface_name=name, load_factor=load_factor, elastic_axis_chord_fraction=0.25, s_m=s,
        lift_per_span_N_m=zeros, inertial_load_per_span_N_m=zeros,
        shear_force_N=shear(s), bending_moment_N_m=moment(s), torque_N_m=torque(s),
    )


def make_cases(shear, moment, torque, safety_factor=1.5, name="wing"):
    """Limit loads from the given functions, ultimate = safety factor x limit."""
    limit = make_loads(shear, moment, torque, name=name)
    return DesignLoadCases(limit=limit, ultimate=limit.scaled(safety_factor), limit_aero=None)


def tip_load_cases(tip_load=30.0, torque=0.0):
    """Point load P at the tip (V = P, M = P (L - s)) and a constant torque."""
    return make_cases(
        shear=lambda s: np.full_like(s, tip_load),
        moment=lambda s: tip_load * (SEMISPAN - s),
        torque=lambda s: np.full_like(s, torque),
    )


def integrate_from_root(values, s):
    strips = 0.5 * (values[1:] + values[:-1]) * np.diff(s)
    return np.concatenate([[0.0], np.cumsum(strips)])


# ---------------------------------------------------------------------- #
# Strength
# ---------------------------------------------------------------------- #
def test_required_section_modulus_is_ultimate_moment_over_allowable():
    cases = tip_load_cases(tip_load=30.0)
    properties = required_section_properties(cases, allowable_stress_Pa=200e6)
    expected = 1.5 * 30.0 * (SEMISPAN - properties.s_m) / 200e6
    np.testing.assert_allclose(properties.required_section_modulus_m3, expected)


# ---------------------------------------------------------------------- #
# Bending stiffness
# ---------------------------------------------------------------------- #
def test_constant_curvature_ei_gives_the_allowed_tip_deflection():
    cases = tip_load_cases()
    properties = required_section_properties(cases, 200e6, max_tip_deflection_fraction=0.08)
    s = properties.s_m[:-1]  # M = 0 at the tip, so EI = 0 there; curvature is the limit 0/0
    moment = cases.limit.bending_moment_N_m[:-1]
    curvature = moment / properties.required_flexural_rigidity_N_m2[:-1]
    np.testing.assert_allclose(curvature, properties.design_curvature_per_m)
    tip_deflection = integrate_from_root(integrate_from_root(np.append(curvature, curvature[-1]),
                                                             properties.s_m), properties.s_m)[-1]
    assert tip_deflection == pytest.approx(0.08 * SEMISPAN, rel=1e-6)
    assert len(s) == N_POINTS - 1


def test_uniform_ei_for_a_tip_load():
    # Constant EI under a tip load P: tip deflection P L^3 / (3 EI).
    cases = tip_load_cases(tip_load=30.0)
    properties = required_section_properties(cases, 200e6, max_tip_deflection_fraction=0.1)
    expected = 30.0 * SEMISPAN**3 / (3 * 0.1 * SEMISPAN)
    assert properties.required_uniform_flexural_rigidity_N_m2 == pytest.approx(expected, rel=1e-5)


def test_uniform_ei_for_a_uniform_load():
    # Constant EI under a uniform load w: tip deflection w L^4 / (8 EI).
    load_per_span = 20.0
    cases = make_cases(
        shear=lambda s: load_per_span * (SEMISPAN - s),
        moment=lambda s: load_per_span * (SEMISPAN - s) ** 2 / 2,
        torque=np.zeros_like,
    )
    properties = required_section_properties(cases, 200e6, max_tip_deflection_fraction=0.1)
    expected = load_per_span * SEMISPAN**4 / (8 * 0.1 * SEMISPAN)
    assert properties.required_uniform_flexural_rigidity_N_m2 == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------- #
# Torsional stiffness
# ---------------------------------------------------------------------- #
def test_constant_torque_gives_equal_gj_both_ways():
    # Constant torque T: twist rate T / GJ is already constant, so both answers
    # are T L / phi_tip.
    cases = tip_load_cases(torque=-1.5)
    properties = required_section_properties(cases, 200e6, max_tip_twist_deg=1.0)
    expected = 1.5 * SEMISPAN / np.radians(1.0)
    np.testing.assert_allclose(properties.required_torsional_rigidity_N_m2, expected)
    assert properties.required_uniform_torsional_rigidity_N_m2 == pytest.approx(expected)


# ---------------------------------------------------------------------- #
# Input checks and CSV output
# ---------------------------------------------------------------------- #
def test_non_positive_limits_are_rejected():
    with pytest.raises(ValueError):
        required_section_properties(tip_load_cases(), 200e6, max_tip_deflection_fraction=0.0)
    with pytest.raises(ValueError):
        required_section_properties(tip_load_cases(), -1.0)


def test_csv_is_named_after_the_surface(tmp_path):
    cases = make_cases(lambda s: 5 * (SEMISPAN - s), lambda s: 2.5 * (SEMISPAN - s) ** 2,
                       lambda s: np.full_like(s, 0.5), name="main wing")
    properties = required_section_properties(cases, 200e6)
    path = write_section_properties_csv(properties, tmp_path / "output")

    assert path == tmp_path / "output" / "main_wing.csv"
    with open(path) as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == N_POINTS
    assert float(rows[0]["s_m"]) == 0.0
    assert float(rows[-1]["s_m"]) == pytest.approx(SEMISPAN)
    assert float(rows[0]["required_section_modulus_m3"]) == pytest.approx(
        properties.required_section_modulus_m3[0], rel=1e-7)
    assert float(rows[0]["required_EI_N_m2"]) == pytest.approx(
        properties.required_flexural_rigidity_N_m2[0], rel=1e-7)
