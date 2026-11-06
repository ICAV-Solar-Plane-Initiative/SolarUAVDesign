"""Build an example solar UAV, analyse its wing and derive the wing loads.

Run from the repository root:

    python examples/reference_aircraft.py

The script
1. builds the planform and prints its geometry,
2. runs a lifting-line analysis of the wing at the cruise condition (L = W),
3. derives the spanwise shear force, bending moment and torque at the limit
   and ultimate load factors, and
4. saves a three-view and a wing-loads plot.

The numbers here are a placeholder concept (about 5 m span, conventional
tail), not a design. The airfoil, masses and load factors below are
representative guesses, to be replaced with real data.
"""

import matplotlib.pyplot as plt

from solaruav import Aircraft, LiftingSurface
from solaruav.aerodynamics import Airfoil, solve_lifting_line_for_lift
from solaruav.structures import chord_proportional_mass, design_load_cases
from solaruav.structures.loads import STANDARD_GRAVITY_M_S2

# ---------------------------------------------------------------------- #
# Placeholder inputs
# ---------------------------------------------------------------------- #
# A cambered low-Reynolds-number section, roughly like those used on small
# solar aircraft. Replace with XFOIL data for the chosen airfoil.
WING_AIRFOIL = Airfoil(lift_slope_per_rad=5.7, alpha_zero_lift_deg=-3.0, cm0=-0.08)

AIRCRAFT_MASS_KG = 5.0       # total mass
WING_MASS_KG = 1.8           # structure + solar panels carried in the wing, both sides
CRUISE_AIRSPEED_M_S = 10.0
AIR_DENSITY_KG_M3 = 1.225    # sea level, ISA

# The limit pull-up is flown at a manoeuvre speed, not at cruise: a 3 g pull-up
# at 10 m/s would need section Cl far beyond stall. Set this from the stall
# speed once Cl_max is known: V_A = V_stall sqrt(n_limit).
MANOEUVRE_AIRSPEED_M_S = 14.0
LIMIT_LOAD_FACTOR = 3.0
SAFETY_FACTOR = 1.5
SPAR_CHORD_FRACTION = 0.25   # elastic axis, fraction of chord from the leading edge
WING_MASS_CHORD_FRACTION = 0.40


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


def print_geometry(ac: Aircraft) -> None:
    for row in ac.summary():
        print(
            f"{row['name']:>6}:  S = {row['area_m2']:.3f} m^2   b = {row['span_m']:.2f} m   "
            f"AR = {row['aspect_ratio']:.1f}   MAC = {row['mac_m']:.3f} m   "
            f"x_ac = {row['x_ac_body_m']:.3f} m"
        )
    print(f"Horizontal tail volume: {ac.horizontal_tail_volume('htail'):.3f}")
    print(f"Vertical tail volume:   {ac.vertical_tail_volume('fin'):.4f}")


def print_cruise(cruise) -> None:
    print(f"\nWing at cruise (L = W = {AIRCRAFT_MASS_KG * STANDARD_GRAVITY_M_S2:.1f} N, "
          f"V = {CRUISE_AIRSPEED_M_S:.1f} m/s):")
    print(f"  alpha = {cruise.alpha_deg:.2f} deg   CL = {cruise.lift_coefficient:.3f}   "
          f"CDi = {cruise.induced_drag_coefficient:.4f}   e = {cruise.span_efficiency:.3f}")
    print(f"  induced drag = {cruise.induced_drag_N:.3f} N   "
          f"peak section Cl = {cruise.local_lift_coefficient.max():.3f}")


def print_loads(cases) -> None:
    aero = cases.limit_aero
    print(f"\nLimit pull-up at V = {MANOEUVRE_AIRSPEED_M_S:.1f} m/s:  alpha = {aero.alpha_deg:.2f} deg   "
          f"CL = {aero.lift_coefficient:.3f}   peak section Cl = {aero.local_lift_coefficient.max():.2f}")
    print("  (The linear lifting line has no stall: check peak Cl against the airfoil's Cl_max.)")
    print("Wing root loads, one side:")
    for name, loads in [("limit", cases.limit), ("ultimate", cases.ultimate)]:
        print(
            f"  {name:>8} (n = {loads.load_factor:.1f}):  "
            f"V = {loads.root_shear_force_N:6.1f} N   M = {loads.root_bending_moment_N_m:6.1f} N m   "
            f"T = {loads.root_torque_N_m:6.2f} N m"
        )


def plot_wing_loads(cruise, cases):
    """Four stacked plots against span: lift per span, shear, bending moment, torque."""
    fig, axes = plt.subplots(4, 1, figsize=(7, 10), sharex=True)
    axes[0].plot(cruise.s_m, cruise.lift_per_span_N_m, "k--", label="cruise, n = 1")
    for loads in (cases.limit, cases.ultimate):
        label = f"n = {loads.load_factor:.1f}"
        axes[0].plot(loads.s_m, loads.net_load_per_span_N_m, label=f"{label} (net of wing weight)")
        axes[1].plot(loads.s_m, loads.shear_force_N, label=label)
        axes[2].plot(loads.s_m, loads.bending_moment_N_m, label=label)
        axes[3].plot(loads.s_m, loads.torque_N_m, label=label)

    ylabels = ["Running load [N/m]", "Shear force [N]", "Bending moment [N m]",
               f"Torque about {SPAR_CHORD_FRACTION:.0%} chord [N m]"]
    for ax, ylabel in zip(axes, ylabels):
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Spanwise position s [m]")
    axes[0].set_title("Reference wing loads (one side)")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    ac = build_reference_aircraft()
    wing = ac["wing"]
    print_geometry(ac)

    cruise = solve_lifting_line_for_lift(
        wing, AIRCRAFT_MASS_KG * STANDARD_GRAVITY_M_S2, CRUISE_AIRSPEED_M_S, AIR_DENSITY_KG_M3,
        airfoil=WING_AIRFOIL, n_output=201,
    )
    print_cruise(cruise)

    cases = design_load_cases(
        wing, WING_AIRFOIL, AIRCRAFT_MASS_KG,
        mass_per_span_kg_m=chord_proportional_mass(wing, WING_MASS_KG),
        airspeed_m_s=MANOEUVRE_AIRSPEED_M_S, air_density_kg_m3=AIR_DENSITY_KG_M3,
        limit_load_factor=LIMIT_LOAD_FACTOR, safety_factor=SAFETY_FACTOR,
        elastic_axis_chord_fraction=SPAR_CHORD_FRACTION,
        mass_chord_fraction=WING_MASS_CHORD_FRACTION,
    )
    print_loads(cases)

    fig, _ = ac.plot()
    fig.savefig("reference_aircraft.png", dpi=150)
    plot_wing_loads(cruise, cases).savefig("reference_wing_loads.png", dpi=150)
    plt.show()
