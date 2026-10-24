import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from solaruav import Aircraft, LiftingSurface, Station


def shoelace(x, y):
    """Polygon area, ignoring NaN separators between sub-polygons."""
    total = 0.0
    breaks = np.where(np.isnan(x))[0]
    for xs, ys in zip(np.split(x, breaks), np.split(y, breaks)):
        xs, ys = xs[~np.isnan(xs)], ys[~np.isnan(ys)]
        total += 0.5 * abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))
    return total


def test_rectangular_wing():
    w = LiftingSurface.trapezoidal("w", span=4.0, root_chord=0.5)
    assert w.area == pytest.approx(2.0)
    assert w.span == pytest.approx(4.0)
    assert w.aspect_ratio == pytest.approx(8.0)
    assert w.mac == pytest.approx(0.5)
    assert w.s_mac == pytest.approx(1.0)
    assert w.x_le_mac == pytest.approx(0.0)
    assert w.taper_ratio == pytest.approx(1.0)


def test_trapezoidal_wing_matches_textbook_formulae():
    b, cr, lam, sweep = 4.0, 1.0, 0.5, 20.0
    w = LiftingSurface.trapezoidal("w", span=b, root_chord=cr, taper=lam, sweep_le_deg=sweep)
    y_mac = b / 6 * (1 + 2 * lam) / (1 + lam)
    assert w.area == pytest.approx(b * cr * (1 + lam) / 2)
    assert w.mac == pytest.approx(2 / 3 * cr * (1 + lam + lam**2) / (1 + lam))
    assert w.s_mac == pytest.approx(y_mac)
    assert w.x_le_mac == pytest.approx(y_mac * np.tan(np.radians(sweep)))
    assert w.sweep_deg(0.0)[0] == pytest.approx(sweep)


def test_subdividing_a_panel_changes_nothing():
    whole = LiftingSurface.trapezoidal("w", span=4.0, root_chord=1.0, taper=0.5, sweep_le_deg=10)
    t = np.tan(np.radians(10))
    split = LiftingSurface(
        "w",
        stations=[Station(0, 0, 1.0), Station(0.5, 0.5 * t, 0.875), Station(2.0, 2.0 * t, 0.5)],
    )
    for prop in ["area", "span", "mac", "s_mac", "x_le_mac", "aspect_ratio"]:
        assert getattr(split, prop) == pytest.approx(getattr(whole, prop)), prop


def test_from_panels_builds_expected_stations():
    w = LiftingSurface.from_panels(
        "w", root_chord=0.3,
        panels=[{"span": 1.0}, {"span": 0.5, "taper": 0.5, "sweep_le_deg": 45}],
    )
    np.testing.assert_allclose(w.s, [0, 1.0, 1.5])
    np.testing.assert_allclose(w.chord, [0.3, 0.3, 0.15])
    np.testing.assert_allclose(w.x_le, [0, 0, 0.5])
    assert w.area == pytest.approx(2 * (0.3 * 1.0 + 0.5 * (0.3 + 0.15) / 2))


def test_fin_is_single_sided():
    fin = LiftingSurface.trapezoidal(
        "fin", span=0.4, root_chord=0.3, taper=0.5, symmetric=False, orientation="vertical",
        origin=(2.0, 0.0, 0.1),
    )
    assert fin.span == pytest.approx(0.4)
    assert fin.area == pytest.approx(0.4 * 0.3 * 1.5 / 2)
    pts = fin.outline_body()
    np.testing.assert_allclose(pts[:, 1], 0.0)  # lies in the symmetry plane
    assert pts[:, 2].min() == pytest.approx(0.1)
    assert pts[:, 2].max() == pytest.approx(0.5)


@pytest.mark.parametrize("s0", [0.0, 0.3])
def test_outline_encloses_planform_area(s0):
    w = LiftingSurface(
        "w",
        stations=[Station(s0, 0.0, 0.4), Station(s0 + 1.0, 0.05, 0.3), Station(s0 + 1.8, 0.2, 0.15)],
    )
    x, s = w.outline()
    assert shoelace(x, s) == pytest.approx(w.area)


def test_chord_interpolation_is_mirrored():
    w = LiftingSurface.trapezoidal("w", span=4.0, root_chord=1.0, taper=0.5)
    assert w.chord_at(1.0) == pytest.approx(0.75)
    assert w.chord_at(-1.0) == pytest.approx(0.75)


def test_invalid_input_rejected():
    with pytest.raises(ValueError):
        Station(0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        LiftingSurface("w", stations=[Station(0, 0, 1)])
    with pytest.raises(ValueError):
        LiftingSurface("w", stations=[Station(1, 0, 1), Station(0.5, 0, 1)])
    with pytest.raises(ValueError):
        LiftingSurface("w", stations=[Station(0, 0, 1), Station(1, 0, 1)], orientation="sideways")


def test_aircraft_reference_and_tail_volumes():
    wing = LiftingSurface.trapezoidal("wing", span=4.0, root_chord=0.5)
    htail = LiftingSurface.trapezoidal("htail", span=1.0, root_chord=0.2, origin=(2.0, 0, 0))
    fin = LiftingSurface.trapezoidal(
        "fin", span=0.3, root_chord=0.2, symmetric=False, orientation="vertical", origin=(2.0, 0, 0)
    )
    ac = Aircraft("test", [wing, htail, fin])
    assert ac.reference_surface == "wing"
    assert ac.s_ref == pytest.approx(2.0)
    assert ac.c_ref == pytest.approx(0.5)
    l_t = (2.0 + 0.05) - 0.125
    assert ac.horizontal_tail_volume("htail") == pytest.approx(0.2 * l_t / (2.0 * 0.5))
    assert ac.vertical_tail_volume("fin") == pytest.approx(0.06 * l_t / (2.0 * 4.0))
    with pytest.raises(ValueError):
        ac.add(LiftingSurface.trapezoidal("wing", span=1.0, root_chord=0.1))


def test_plot_runs():
    wing = LiftingSurface.from_panels("wing", 0.3, [{"span": 1.0}, {"span": 0.5, "taper": 0.6}])
    fin = LiftingSurface.trapezoidal(
        "fin", span=0.3, root_chord=0.2, symmetric=False, orientation="vertical", origin=(1.5, 0, 0)
    )
    fig, axes = Aircraft("test", [wing, fin]).plot()
    assert len(axes) == 2
    fig, axes = Aircraft("test", [wing, fin]).plot(views=["top"])
    assert len(axes) == 1
