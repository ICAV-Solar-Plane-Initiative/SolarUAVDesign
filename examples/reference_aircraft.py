"""Build an example solar UAV planform, print its properties and plot it.

Run from the repository root:

    python examples/reference_aircraft.py

The numbers here are a placeholder concept (about 5 m span, conventional
tail), not a design.
"""

import matplotlib.pyplot as plt

from solaruav import Aircraft, LiftingSurface


def build_reference_aircraft() -> Aircraft:
    # Main wing: rectangular centre section with tapered outer panels.
    wing = LiftingSurface.from_panels(
        "wing",
        root_chord=0.32,
        panels=[
            {"span": 1.5, "taper": 1.0, "twist_deg": 10.0, "sweep_le_deg": 0.0},
            {"span": 1.0, "taper": 0.6, "sweep_le_deg": 3.0},
        ],
        symmetric=True,
        origin=(0.0, 0.0, 0.0),
    )

    # Horizontal tailplane, 1.6 m behind the wing leading edge.
    htail = LiftingSurface.trapezoidal(
        "htail",
        span=1.0,
        root_chord=0.20,
        taper=0.8,
        sweep_le_deg=5.0,
        symmetric=True,
        origin=(1.60, 0.0, 0.0),
    )

    # Single fin on the centreline.
    fin = LiftingSurface.trapezoidal(
        "fin",
        span=0.40,
        root_chord=0.28,
        taper=0.6,
        sweep_le_deg=25.0,
        symmetric=False,
        orientation="vertical",
        origin=(1.58, 0.0, 0.0),
    )

    return Aircraft("Reference concept", surfaces=[wing, htail, fin], reference_surface="wing")


if __name__ == "__main__":
    ac = build_reference_aircraft()

    for row in ac.summary():
        print(
            f"{row['name']:>6}:  S = {row['area_m2']:.3f} m^2   b = {row['span_m']:.2f} m   "
            f"AR = {row['aspect_ratio']:.1f}   MAC = {row['mac_m']:.3f} m   "
            f"x_ac = {row['x_ac_body_m']:.3f} m"
        )
    print(f"Horizontal tail volume: {ac.horizontal_tail_volume('htail'):.3f}")
    print(f"Vertical tail volume:   {ac.vertical_tail_volume('fin'):.4f}")

    fig, _ = ac.plot()
    fig.savefig("reference_aircraft.png", dpi=150)
    plt.show()
