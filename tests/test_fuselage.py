import numpy as np
import pytest

from solaruav import Aircraft, Fuselage, FuselageSection, LiftingSurface


def box(length=2.0, width=0.2, height=0.1, **kwargs):
    return Fuselage("box", [FuselageSection(0.0, width, height), FuselageSection(length, width, height)], **kwargs)


def test_box_volume_and_wetted_area():
    body = box(length=2.0, width=0.2, height=0.1)
    assert body.length == pytest.approx(2.0)
    assert body.volume_m3 == pytest.approx(2.0 * 0.2 * 0.1)
    assert body.wetted_area_m2 == pytest.approx(2.0 * 2 * (0.2 + 0.1))  # perimeter x length, no end faces
    assert body.frontal_area_m2 == pytest.approx(0.02)


def test_pyramid_volume_is_exact():
    # A rectangular pyramid (base w x h, closing to a point) has volume w h L / 3.
    pyramid = Fuselage("pyramid", [FuselageSection(0.0, 0.3, 0.2), FuselageSection(1.5, 0.0, 0.0)])
    assert pyramid.volume_m3 == pytest.approx(0.3 * 0.2 * 1.5 / 3)


def test_wedge_wetted_area_uses_slant_lengths():
    # Width constant, height closing to zero symmetrically: top and bottom faces
    # are rectangles of slant length sqrt(L^2 + (h/2)^2); sides are triangles.
    length, width, height = 1.0, 0.4, 0.6
    wedge = Fuselage("wedge", [FuselageSection(0.0, width, height), FuselageSection(length, width, 0.0)])
    slant = np.hypot(length, height / 2)
    expected = 2 * width * slant + 2 * 0.5 * height * length
    assert wedge.wetted_area_m2 == pytest.approx(expected)


def test_from_segments_builds_sections_from_the_nose():
    body = Fuselage.from_segments(
        "pod", nose_width=0.0, nose_height=0.0,
        segments=[{"length": 0.2, "width": 0.1, "height": 0.12}, {"length": 0.8, "width": 0.1, "height": 0.12, "z_centre": 0.05}],
    )
    assert body.x.tolist() == pytest.approx([0.0, 0.2, 1.0])
    assert body.width.tolist() == pytest.approx([0.0, 0.1, 0.1])
    assert body.z_centre.tolist() == pytest.approx([0.0, 0.0, 0.05])
    assert body.width_at(0.1) == pytest.approx(0.05)


def test_mirrored_fuselage_is_two_bodies():
    pair = box(origin=(0.0, 0.8, 0.0), symmetric=True)
    assert pair.n_bodies == 2
    top_starboard, top_port = pair.outline_top_body(), pair.outline_top_body(mirrored=True)
    def centre_y(outline):
        return 0.5 * (outline[:, 1].max() + outline[:, 1].min())

    assert centre_y(top_starboard) == pytest.approx(0.8)
    assert centre_y(top_port) == pytest.approx(-0.8)


def test_invalid_fuselages_are_rejected():
    with pytest.raises(ValueError):
        box(symmetric=True)  # mirrored about the centreline it sits on
    with pytest.raises(ValueError):
        Fuselage("backwards", [FuselageSection(1.0, 0.1, 0.1), FuselageSection(0.0, 0.1, 0.1)])
    with pytest.raises(ValueError):
        FuselageSection(0.0, -0.1, 0.1)


def test_aircraft_holds_fuselages():
    wing = LiftingSurface.trapezoidal("wing", span=4.0, root_chord=0.3)
    aircraft = Aircraft("test", surfaces=[wing], fuselages=[box()])
    assert aircraft["box"].name == "box"
    assert aircraft.fuselage_summary()[0]["volume_m3"] == pytest.approx(0.04)
    with pytest.raises(ValueError):
        aircraft.add(box())  # duplicate name
    with pytest.raises(ValueError):
        Aircraft("clash", surfaces=[wing], fuselages=[Fuselage("wing", box().sections)])
