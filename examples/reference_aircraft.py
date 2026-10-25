"""Build an example solar UAV planform, print its properties and plot it.

Run from the repository root:

    python examples/reference_aircraft.py

The numbers are a placeholder concept (5 m span, conventional tail), not a
design. The plot is also saved as reference_aircraft.png in the folder you run
from; that file is ignored by git.
"""

import matplotlib.pyplot as plt

from solaruav import Aircraft, LiftingSurface


def build_reference_aircraft() -> Aircraft:
    """Return the placeholder reference aircraft: wing, tailplane and fin."""

    # Main wing: rectangular centre section with tapered outer panels.
    wing = LiftingSurface.from_panels(
        "wing",
        root_chord=0.32,
        panels=[
            {"span": 1.5, "taper": 1.0, "sweep_le_deg": 0.0},
            {"span": 1.0, "taper": 0.6, "sweep_le_deg": 3.0},
        ],
        symmetric=True,
        origin=(0.0, 0.0, 0.0),
    )

    # Horizontal tailplane, leading edge 1.6 m behind the wing leading edge.
    horizontal_tail = LiftingSurface.trapezoidal(
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

    return Aircraft(
        "Reference concept",
        surfaces=[wing, horizontal_tail, fin],
        reference_surface="wing",
    )


def print_summary(aircraft: Aircraft) -> None:
    """Print the main planform properties of each surface and the tail volumes."""
    for row in aircraft.summary():
        print(
            f"{row['name']:>6}:  "
            f"area = {row['area_m2']:.3f} m^2   "
            f"span = {row['span_m']:.2f} m   "
            f"AR = {row['aspect_ratio']:.1f}   "
            f"MAC = {row['mac_m']:.3f} m   "
            f"x_ac = {row['x_aerodynamic_centre_body_m']:.3f} m"
        )
    print(f"Horizontal tail volume: {aircraft.horizontal_tail_volume('htail'):.3f}")
    print(f"Vertical tail volume:   {aircraft.vertical_tail_volume('fin'):.4f}")


if __name__ == "__main__":
    aircraft = build_reference_aircraft()
    print_summary(aircraft)

    fig, _ = aircraft.plot()
    fig.savefig("reference_aircraft.png", dpi=150)
    plt.show()
