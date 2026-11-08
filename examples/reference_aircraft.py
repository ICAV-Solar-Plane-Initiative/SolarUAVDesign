"""Build an example solar UAV, analyse its wing, derive the wing loads and size the spar.

Run from the repository root:

    python examples/reference_aircraft.py

The script
1. builds the planform and prints its geometry,
2. runs a lifting-line analysis of the wing at the cruise condition (L = W),
3. derives the spanwise shear force, bending moment and torque at the limit
   and ultimate load factors,
4. analyses a carbon-tube spar as a cantilever beam under those loads and
   writes the results to output/wing.csv, and
5. saves a three-view, a lifting-line plot (lift distribution, section Cl,
   angles, downwash, induced drag), a wing-loads plot and a beam plot to output/.

The numbers here are a placeholder concept (about 5 m span, conventional
tail), not a design. The airfoil, masses, load factors, spar size and
material below are representative guesses, to be replaced with real data.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from solaruav import Aircraft, LiftingSurface
from solaruav.aerodynamics import Airfoil, solve_lifting_line_for_lift
from solaruav.plotting import plot_beam, plot_lifting_line, plot_wing_loads
from solaruav.structures import (
    Material,
    TubeSpar,
    analyse_design_cases,
    chord_proportional_mass,
    design_load_cases,
    write_beam_csv,
)
from solaruav.structures.loads import STANDARD_GRAVITY_M_S2

OUTPUT_DIR = Path("output")  # CSV files and figures go here (ignored by git)

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

# Main spar: a carbon-fibre tube tapering from root to tip. Typical values for
# a roll-wrapped tube; replace with the supplier's data sheet.
CARBON_TUBE = Material(
    name="Carbon-fibre tube (placeholder)",
    youngs_modulus_Pa=90e9,
    shear_modulus_Pa=15e9,
    density_kg_m3=1550.0,
    allowable_stress_Pa=400e6,       # compression governs
    allowable_shear_stress_Pa=60e6,
)
WING_SPAR = TubeSpar(
    material=CARBON_TUBE,
    root_outer_diameter_m=0.030, tip_outer_diameter_m=0.016,
    root_wall_thickness_m=0.0010, tip_wall_thickness_m=0.0005,
)
AIRFOIL_THICKNESS_RATIO = 0.12  # max thickness / chord, to check the spar fits


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


def print_beam(beam_results, wing, csv_path) -> None:
    limit, ultimate = beam_results
    print(f"\nSpar beam ({WING_SPAR.material.name}):")
    print(f"  spar mass = {limit.spar_mass_kg:.3f} kg (both sides)")
    print(f"  ultimate: max failure index = {ultimate.max_failure_index:.2f} at s = {ultimate.critical_s_m:.2f} m   "
          f"(reserve factor {1 / ultimate.max_failure_index:.2f})")
    print(f"  limit:    tip deflection = {limit.tip_deflection_m:.3f} m "
          f"({limit.tip_deflection_fraction:.1%} of semispan)   tip twist = {limit.tip_twist_deg:.2f} deg")
    if limit.tip_deflection_fraction > 0.1:
        print("  Warning: tip deflection exceeds 10% of the semispan; small-deflection theory is unreliable.")
    for name, span_fraction in [("root", 0.0), ("tip", 1.0)]:
        depth_m = AIRFOIL_THICKNESS_RATIO * float(wing.chord_at(span_fraction * wing.span / 2))
        diameter_m = float(WING_SPAR.outer_diameter_m(span_fraction))
        fits = "fits" if diameter_m < depth_m else "DOES NOT FIT"
        print(f"  {name}: tube {diameter_m * 1e3:.0f} mm in a {depth_m * 1e3:.0f} mm deep section: {fits}")
    print(f"  Results written to {csv_path}")


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

    beam_results = analyse_design_cases(cases, WING_SPAR)
    csv_path = write_beam_csv(beam_results, OUTPUT_DIR)
    print_beam(beam_results, wing, csv_path)

    OUTPUT_DIR.mkdir(exist_ok=True)
    figures = {
        "reference_aircraft.png": ac.plot()[0],
        "reference_lifting_line.png": plot_lifting_line(cruise, wing)[0],
        "reference_wing_loads.png": plot_wing_loads(cases, cruise)[0],
        "reference_wing_beam.png": plot_beam(beam_results)[0],
    }
    for file_name, figure in figures.items():
        figure.savefig(OUTPUT_DIR / file_name, dpi=150)
    print(f"Figures saved to {OUTPUT_DIR}/")
    plt.show()
