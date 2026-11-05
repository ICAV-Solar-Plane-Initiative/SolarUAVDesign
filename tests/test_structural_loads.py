import numpy as np
import pytest

from solaruav.structures.loads import calculate_wing_loads


def test_uniform_load_on_cantilever():
    result = calculate_wing_loads(
        span_m=[0.0, 1.0, 2.0],
        lift_per_span_N_m=[10.0, 10.0, 10.0],
        safety_factor=1.5,
    )

    np.testing.assert_allclose(result.shear_force_limit_N, [20.0, 10.0, 0.0])
    np.testing.assert_allclose(result.bending_moment_limit_N_m, [20.0, 5.0, 0.0])
    np.testing.assert_allclose(result.shear_force_ultimate_N, [30.0, 15.0, 0.0])
    np.testing.assert_allclose(result.bending_moment_ultimate_N_m, [30.0, 7.5, 0.0])


def test_triangular_load_and_distributed_weight():
    result = calculate_wing_loads(
        span_m=[0.0, 2.0],
        lift_per_span_N_m=[10.0, 0.0],
        mass_per_span_kg_m=1.0,
        limit_load_factor=2.0,
        safety_factor=1.0,
    )

    expected_root_shear = 2.0 * (10.0 - 2.0 * 9.80665)
    expected_root_moment = 2.0 * (20.0 / 3.0 - 2.0 * 9.80665)
    assert result.shear_force_limit_N[0] == pytest.approx(expected_root_shear)
    assert result.bending_moment_limit_N_m[0] == pytest.approx(expected_root_moment)
    assert result.shear_force_limit_N[-1] == 0.0
    assert result.bending_moment_limit_N_m[-1] == 0.0


@pytest.mark.parametrize(
    "span, lift, mass",
    [
        ([0.0], [1.0], 0.0),
        ([0.0, 1.0], [1.0], 0.0),
        ([0.0, 1.0, 0.5], [1.0, 1.0, 1.0], 0.0),
        ([0.0, 1.0], [1.0, 1.0], [0.0, 1.0, 2.0]),
        ([0.0, 1.0], [1.0, 1.0], -1.0),
    ],
)
def test_invalid_distributions_raise_value_error(span, lift, mass):
    with pytest.raises(ValueError):
        calculate_wing_loads(span, lift, mass)


def test_non_finite_loads_raise_value_error():
    with pytest.raises(ValueError, match="finite"):
        calculate_wing_loads([0.0, 1.0], [1.0, np.nan])
