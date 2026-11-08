import csv

import numpy as np
import pytest

from solaruav.structures import (
    Material,
    TubeSpar,
    WingLoads,
    analyse_beam,
    write_beam_csv,
)

SEMISPAN = 2.0  # m
N_POINTS = 401

ALUMINIUM = Material(
    name="aluminium", youngs_modulus_Pa=70e9, shear_modulus_Pa=26e9, density_kg_m3=2700.0,
    allowable_stress_Pa=250e6, allowable_shear_stress_Pa=150e6,
)
CONSTANT_TUBE = TubeSpar(ALUMINIUM, 0.03, 0.03, 0.002, 0.002)


def make_loads(shear, moment, torque, name="wing"):
    """WingLoads with given V, M, T arrays (shape (N_POINTS,)) on s from 0 to SEMISPAN."""
    s = np.linspace(0.0, SEMISPAN, N_POINTS)
    zeros = np.zeros_like(s)
    return WingLoads(
        surface_name=name, load_factor=1.0, elastic_axis_chord_fraction=0.25, s_m=s,
        lift_per_span_N_m=zeros, inertial_load_per_span_N_m=zeros,
        shear_force_N=shear(s), bending_moment_N_m=moment(s), torque_N_m=torque(s),
    )


def tube_second_moment(outer, wall):
    return np.pi * (outer**4 - (outer - 2 * wall) ** 4) / 64


# ---------------------------------------------------------------------- #
# Section properties
# ---------------------------------------------------------------------- #
def test_tube_section_properties():
    outer, wall = 0.03, 0.002
    inner = outer - 2 * wall
    assert CONSTANT_TUBE.area_m2(0.5) == pytest.approx(np.pi * (outer**2 - inner**2) / 4)
    assert CONSTANT_TUBE.second_moment_m4(0.5) == pytest.approx(tube_second_moment(outer, wall))
    assert CONSTANT_TUBE.polar_moment_m4(0.5) == pytest.approx(2 * tube_second_moment(outer, wall))


def test_tube_tapers_linearly():
    tube = TubeSpar(ALUMINIUM, 0.03, 0.01, 0.002, 0.001)
    assert tube.outer_diameter_m(0.5) == pytest.approx(0.02)
    assert tube.wall_thickness_m(0.5) == pytest.approx(0.0015)


def test_impossible_tube_is_rejected():
    with pytest.raises(ValueError):
        TubeSpar(ALUMINIUM, 0.03, 0.03, 0.02, 0.002)  # wall thicker than the radius


# ---------------------------------------------------------------------- #
# Cantilever results
# ---------------------------------------------------------------------- #
def test_uniform_load_tip_deflection():
    # Uniform load w on a cantilever of length L: tip deflection w L^4 / (8 EI).
    load_per_span = 20.0  # N/m
    loads = make_loads(
        shear=lambda s: load_per_span * (SEMISPAN - s),
        moment=lambda s: load_per_span * (SEMISPAN - s) ** 2 / 2,
        torque=np.zeros_like,
    )
    result = analyse_beam(loads, CONSTANT_TUBE, "limit")
    flexural_rigidity = ALUMINIUM.youngs_modulus_Pa * tube_second_moment(0.03, 0.002)
    expected = load_per_span * SEMISPAN**4 / (8 * flexural_rigidity)
    assert result.tip_deflection_m == pytest.approx(expected, rel=1e-4)


def test_tip_point_load_deflection():
    # Point load P at the tip: tip deflection P L^3 / (3 EI).
    tip_load = 30.0  # N
    loads = make_loads(
        shear=lambda s: np.full_like(s, tip_load),
        moment=lambda s: tip_load * (SEMISPAN - s),
        torque=np.zeros_like,
    )
    result = analyse_beam(loads, CONSTANT_TUBE, "limit")
    flexural_rigidity = ALUMINIUM.youngs_modulus_Pa * tube_second_moment(0.03, 0.002)
    assert result.tip_deflection_m == pytest.approx(tip_load * SEMISPAN**3 / (3 * flexural_rigidity), rel=1e-5)


def test_constant_torque_twist():
    # Constant torque T: tip twist T L / GJ.
    torque = 2.0  # N m
    loads = make_loads(shear=np.zeros_like, moment=np.zeros_like, torque=lambda s: np.full_like(s, torque))
    result = analyse_beam(loads, CONSTANT_TUBE, "limit")
    torsional_rigidity = ALUMINIUM.shear_modulus_Pa * 2 * tube_second_moment(0.03, 0.002)
    assert result.tip_twist_deg == pytest.approx(np.degrees(torque * SEMISPAN / torsional_rigidity), rel=1e-9)


def test_root_stresses_and_failure_index():
    tip_load = 30.0
    loads = make_loads(
        shear=lambda s: np.full_like(s, tip_load),
        moment=lambda s: tip_load * (SEMISPAN - s),
        torque=np.zeros_like,
    )
    result = analyse_beam(loads, CONSTANT_TUBE, "ultimate")
    second_moment = tube_second_moment(0.03, 0.002)
    expected_stress = tip_load * SEMISPAN * 0.015 / second_moment
    assert result.max_bending_stress_Pa == pytest.approx(expected_stress)
    assert result.critical_s_m == 0.0
    assert result.max_failure_index == pytest.approx(expected_stress / ALUMINIUM.allowable_stress_Pa)
    assert result.shear_stress_Pa[0] == pytest.approx(2 * tip_load / CONSTANT_TUBE.area_m2(0.0))


def test_spar_mass():
    loads = make_loads(np.zeros_like, np.zeros_like, np.zeros_like)
    result = analyse_beam(loads, CONSTANT_TUBE, "limit")
    expected = ALUMINIUM.density_kg_m3 * CONSTANT_TUBE.area_m2(0.0) * 2 * SEMISPAN
    assert result.spar_mass_kg == pytest.approx(expected)


def test_downward_moment_deflects_down():
    loads = make_loads(np.zeros_like, moment=lambda s: -np.ones_like(s), torque=np.zeros_like)
    assert analyse_beam(loads, CONSTANT_TUBE, "limit").tip_deflection_m < 0


# ---------------------------------------------------------------------- #
# CSV output
# ---------------------------------------------------------------------- #
def test_csv_is_named_after_the_surface(tmp_path):
    loads = make_loads(lambda s: 5 * (SEMISPAN - s), lambda s: 2.5 * (SEMISPAN - s) ** 2, np.zeros_like,
                       name="main wing")
    limit = analyse_beam(loads, CONSTANT_TUBE, "limit")
    ultimate = analyse_beam(loads.scaled(1.5), CONSTANT_TUBE, "ultimate")
    path = write_beam_csv([limit, ultimate], tmp_path / "output")

    assert path == tmp_path / "output" / "main_wing.csv"
    with open(path) as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == N_POINTS
    assert float(rows[0]["s_m"]) == 0.0
    assert float(rows[-1]["s_m"]) == pytest.approx(SEMISPAN)
    assert float(rows[0]["ultimate_bending_moment_N_m"]) == pytest.approx(1.5 * float(rows[0]["limit_bending_moment_N_m"]))
    assert float(rows[-1]["limit_deflection_m"]) == pytest.approx(limit.tip_deflection_m, rel=1e-7)


def test_csv_rejects_mixed_surfaces(tmp_path):
    a = analyse_beam(make_loads(np.zeros_like, np.zeros_like, np.zeros_like, name="wing"), CONSTANT_TUBE, "limit")
    b = analyse_beam(make_loads(np.zeros_like, np.zeros_like, np.zeros_like, name="htail"), CONSTANT_TUBE, "ultimate")
    with pytest.raises(ValueError):
        write_beam_csv([a, b], tmp_path)


def test_bad_load_case_name_is_rejected():
    loads = make_loads(np.zeros_like, np.zeros_like, np.zeros_like)
    with pytest.raises(ValueError):
        analyse_beam(loads, CONSTANT_TUBE, "limit case")
