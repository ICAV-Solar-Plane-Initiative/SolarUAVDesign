"""Tests for 2D planform geometry, checked against hand calculations.

Textbook formulas for a single straight-tapered wing (span b, root chord c_r,
taper ratio lambda, leading-edge sweep Lambda_LE), as given in Raymer,
*Aircraft Design: A Conceptual Approach*:

    area             S         = b * c_r * (1 + lambda) / 2
    MAC              c_bar     = (2/3) * c_r * (1 + lambda + lambda^2) / (1 + lambda)
    MAC position     y_bar     = (b/6) * (1 + 2*lambda) / (1 + lambda)
    MAC LE position  x_le_bar  = y_bar * tan(Lambda_LE)
"""

import matplotlib

matplotlib.use("Agg")  # draw plots off-screen during tests

import numpy as np
import pytest

from solaruav import Aircraft, LiftingSurface, Station


def polygon_area(x, y):
    """Area enclosed by a closed outline (shoelace formula).

    NaN entries separate independent shapes; their areas are added.
    """
    total = 0.0
    breaks = np.where(np.isnan(x))[0]
    for xs, ys in zip(np.split(x, breaks), np.split(y, breaks)):
        xs, ys = xs[~np.isnan(xs)], ys[~np.isnan(ys)]
        total += 0.5 * abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))
    return total


# ---------------------------------------------------------------------- #
# Single surfaces
# ---------------------------------------------------------------------- #
def test_rectangular_wing():
    # 4 m x 0.5 m rectangle: S = 2 m^2, AR = 4^2 / 2 = 8, MAC = chord,
    # MAC halfway out along each side.
    wing = LiftingSurface.trapezoidal("wing", span=4.0, root_chord=0.5)

    assert wing.area == pytest.approx(2.0)
    assert wing.span == pytest.approx(4.0)
    assert wing.aspect_ratio == pytest.approx(8.0)
    assert wing.taper_ratio == pytest.approx(1.0)
    assert wing.mac == pytest.approx(0.5)
    assert wing.mac_span_position == pytest.approx(1.0)
    assert wing.mac_x_leading_edge == pytest.approx(0.0)


def test_tapered_swept_wing_matches_textbook_formulas():
    span, root_chord, taper, sweep_le_deg = 4.0, 1.0, 0.5, 20.0
    wing = LiftingSurface.trapezoidal(
        "wing", span=span, root_chord=root_chord, taper=taper, sweep_le_deg=sweep_le_deg
    )

    expected_area = span * root_chord * (1 + taper) / 2
    expected_mac = 2 / 3 * root_chord * (1 + taper + taper**2) / (1 + taper)
    expected_mac_span_position = span / 6 * (1 + 2 * taper) / (1 + taper)
    expected_mac_x_le = expected_mac_span_position * np.tan(np.radians(sweep_le_deg))

    assert wing.area == pytest.approx(expected_area)
    assert wing.mac == pytest.approx(expected_mac)
    assert wing.mac_span_position == pytest.approx(expected_mac_span_position)
    assert wing.mac_x_leading_edge == pytest.approx(expected_mac_x_le)
    assert wing.sweep_deg(chord_fraction=0.0)[0] == pytest.approx(sweep_le_deg)


def test_splitting_a_panel_changes_nothing():
    # Adding an extra station on the straight line between root and tip
    # describes the same planform, so every property must be unchanged.
    whole = LiftingSurface.trapezoidal(
        "wing", span=4.0, root_chord=1.0, taper=0.5, sweep_le_deg=10.0
    )
    tan_sweep = np.tan(np.radians(10.0))
    split = LiftingSurface(
        "wing",
        stations=[
            Station(span_position=0.0, x_leading_edge=0.0, chord=1.0),
            Station(span_position=0.5, x_leading_edge=0.5 * tan_sweep, chord=0.875),
            Station(span_position=2.0, x_leading_edge=2.0 * tan_sweep, chord=0.5),
        ],
    )

    for name in ["area", "span", "aspect_ratio", "mac", "mac_span_position", "mac_x_leading_edge"]:
        assert getattr(split, name) == pytest.approx(getattr(whole, name)), name


def test_from_panels_builds_expected_stations():
    # Panel 1: 1.0 m long, rectangular. Panel 2: 0.5 m long, taper 0.5, 45 deg sweep,
    # so its leading edge moves 0.5 m aft.
    wing = LiftingSurface.from_panels(
        "wing",
        root_chord=0.3,
        panels=[
            {"span": 1.0},
            {"span": 0.5, "taper": 0.5, "sweep_le_deg": 45.0},
        ],
    )

    np.testing.assert_allclose(wing.span_positions, [0.0, 1.0, 1.5])
    np.testing.assert_allclose(wing.chords, [0.3, 0.3, 0.15])
    np.testing.assert_allclose(wing.x_leading_edges, [0.0, 0.0, 0.5])

    one_side_area = 0.3 * 1.0 + 0.5 * (0.3 + 0.15) / 2
    assert wing.area == pytest.approx(2 * one_side_area)


def test_fin_is_single_sided_and_vertical():
    fin = LiftingSurface.trapezoidal(
        "fin",
        span=0.4,
        root_chord=0.3,
        taper=0.5,
        symmetric=False,
        orientation="vertical",
        origin=(2.0, 0.0, 0.1),
    )

    assert fin.span == pytest.approx(0.4)  # height, not doubled
    assert fin.area == pytest.approx(0.4 * 0.3 * (1 + 0.5) / 2)

    outline = fin.outline_body()
    np.testing.assert_allclose(outline[:, 1], 0.0)  # lies in the symmetry plane
    assert outline[:, 2].min() == pytest.approx(0.1)  # root at origin height
    assert outline[:, 2].max() == pytest.approx(0.5)  # tip 0.4 m higher


@pytest.mark.parametrize("first_span_position", [0.0, 0.3])
def test_outline_encloses_the_planform_area(first_span_position):
    # 0.0: halves meet on the centreline. 0.3: two separate halves.
    s0 = first_span_position
    wing = LiftingSurface(
        "wing",
        stations=[
            Station(span_position=s0, x_leading_edge=0.0, chord=0.4),
            Station(span_position=s0 + 1.0, x_leading_edge=0.05, chord=0.3),
            Station(span_position=s0 + 1.8, x_leading_edge=0.2, chord=0.15),
        ],
    )

    x, span_position = wing.outline()
    assert polygon_area(x, span_position) == pytest.approx(wing.area)


def test_chord_interpolation_is_mirrored_on_symmetric_surfaces():
    # Chord falls linearly from 1.0 at the root to 0.5 at the tip (2 m out),
    # so halfway out it is 0.75 on either side.
    wing = LiftingSurface.trapezoidal("wing", span=4.0, root_chord=1.0, taper=0.5)

    assert wing.chord_at(1.0) == pytest.approx(0.75)
    assert wing.chord_at(-1.0) == pytest.approx(0.75)


# ---------------------------------------------------------------------- #
# Input checking
# ---------------------------------------------------------------------- #
def test_station_rejects_bad_chord():
    with pytest.raises(ValueError, match="chord must be positive"):
        Station(span_position=0.0, x_leading_edge=0.0, chord=0.0)
    with pytest.raises(ValueError, match="finite"):
        Station(span_position=np.nan, x_leading_edge=0.0, chord=1.0)


def test_surface_rejects_bad_stations_and_orientation():
    root = Station(span_position=0.0, x_leading_edge=0.0, chord=1.0)
    with pytest.raises(ValueError, match="at least two stations"):
        LiftingSurface("wing", stations=[root])
    with pytest.raises(ValueError, match="strictly increase"):
        LiftingSurface("wing", stations=[Station(1.0, 0.0, 1.0), Station(0.5, 0.0, 1.0)])
    with pytest.raises(ValueError, match="orientation"):
        LiftingSurface("wing", stations=[root, Station(1.0, 0.0, 1.0)], orientation="sideways")


def test_from_panels_rejects_unknown_or_missing_keys():
    with pytest.raises(ValueError, match="unknown key"):
        LiftingSurface.from_panels("wing", root_chord=0.3, panels=[{"span": 1.0, "twist_deg": 5.0}])
    with pytest.raises(ValueError, match="'span' is required"):
        LiftingSurface.from_panels("wing", root_chord=0.3, panels=[{"taper": 0.5}])


# ---------------------------------------------------------------------- #
# Aircraft
# ---------------------------------------------------------------------- #
def test_aircraft_reference_values_and_tail_volumes():
    wing = LiftingSurface.trapezoidal("wing", span=4.0, root_chord=0.5)
    htail = LiftingSurface.trapezoidal("htail", span=1.0, root_chord=0.2, origin=(2.0, 0.0, 0.0))
    fin = LiftingSurface.trapezoidal(
        "fin", span=0.3, root_chord=0.2, symmetric=False, orientation="vertical",
        origin=(2.0, 0.0, 0.0),
    )
    aircraft = Aircraft("test", [wing, htail, fin])

    assert aircraft.reference_surface == "wing"  # defaults to the first surface
    assert aircraft.reference_area == pytest.approx(2.0)
    assert aircraft.reference_span == pytest.approx(4.0)
    assert aircraft.reference_chord == pytest.approx(0.5)

    # Aerodynamic centres: wing at 0.25 * 0.5 = 0.125 m, tails at 2.0 + 0.25 * 0.2 = 2.05 m.
    tail_arm = 2.05 - 0.125
    # V_H = S_t * l_t / (S_ref * c_ref);  V_V = S_v * l_v / (S_ref * b_ref)
    assert aircraft.horizontal_tail_volume("htail") == pytest.approx(0.2 * tail_arm / (2.0 * 0.5))
    assert aircraft.vertical_tail_volume("fin") == pytest.approx(0.06 * tail_arm / (2.0 * 4.0))


def test_aircraft_rejects_duplicate_and_unknown_names():
    wing = LiftingSurface.trapezoidal("wing", span=4.0, root_chord=0.5)
    aircraft = Aircraft("test", [wing])

    with pytest.raises(ValueError, match="already exists"):
        aircraft.add(LiftingSurface.trapezoidal("wing", span=1.0, root_chord=0.1))
    with pytest.raises(KeyError, match="No surface named"):
        aircraft["canard"]
    with pytest.raises(ValueError, match="not one of the surfaces"):
        Aircraft("test", [wing], reference_surface="canard")


# ---------------------------------------------------------------------- #
# Plotting
# ---------------------------------------------------------------------- #
def test_plotting_runs_for_each_view_choice():
    wing = LiftingSurface.from_panels("wing", 0.3, [{"span": 1.0}, {"span": 0.5, "taper": 0.6}])
    fin = LiftingSurface.trapezoidal(
        "fin", span=0.3, root_chord=0.2, symmetric=False, orientation="vertical",
        origin=(1.5, 0.0, 0.0),
    )
    aircraft = Aircraft("test", [wing, fin])

    _, axes = aircraft.plot()
    assert len(axes) == 2
    _, axes = aircraft.plot(views=["top"])
    assert len(axes) == 1
    with pytest.raises(ValueError, match="view must be one of"):
        aircraft.plot(views=["front"])
