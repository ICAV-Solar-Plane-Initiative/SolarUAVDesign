"""A twin-fuselage version of the reference aircraft, compared with the original.

Run from the repository root:

    python examples/twin_fuselage.py

The wing, total mass and flight condition are the same as in
``reference_aircraft.py``; only the layout changes. The single central
fuselage becomes two identical fuselages at y = +-0.75 m, each carrying half
of the original's masses, joined at the back by a tailplane between the booms
with a fin on each. The tailplane is held up by the booms at its tips, and
each boom carries half of the tail lift.

A central fuselage hangs its weight on the wing root. Twin fuselages hang it
part-way out along the span, where it offsets the lift, so the wing's
bending moment inboard of the mounts falls. The script runs the same four
steps on both layouts and prints the difference. Results go to
output/twin/ (one CSV per component, named after it).
"""

from __future__ import annotations

from pathlib import Path

from reference_aircraft import (
    FUSELAGE_MASSES,
    OUTPUT_DIR,
    FuselageMasses,
    build_fuselage_pod,
    build_reference_aircraft,
    build_wing,
    print_geometry,
    print_results,
    run_structural_analysis,
    save_figures,
)

from solaruav import Aircraft, LiftingSurface
from solaruav.structures import MassItem

FUSELAGE_Y_M = 0.75  # spanwise position of each fuselage
TWIN_OUTPUT_DIR = OUTPUT_DIR / "twin"


def build_twin_fuselage_aircraft() -> Aircraft:
    htail = LiftingSurface.trapezoidal(
        "htail", span=2 * FUSELAGE_Y_M, root_chord=0.18, symmetric=True, origin=(1.60, 0.0, 0.0),
    )
    fins = [
        LiftingSurface.trapezoidal(
            f"fin_{side}", span=0.30, root_chord=0.24, taper=0.6, sweep_le_deg=25.0,
            symmetric=False, orientation="vertical", origin=(1.58, sign * FUSELAGE_Y_M, 0.0),
        )
        for side, sign in [("port", -1), ("starboard", 1)]
    ]
    return Aircraft(
        "Twin-fuselage concept",
        surfaces=[build_wing(), htail, *fins],
        reference_surface="wing",
        fuselages=[build_fuselage_pod("fuselages", origin_y=FUSELAGE_Y_M, symmetric=True)],
    )


def half_of(masses: FuselageMasses) -> FuselageMasses:
    """The masses of one body when a single fuselage's load is split between two."""
    return FuselageMasses(
        structure_mass_kg=masses.structure_mass_kg / 2,
        items=tuple(MassItem(item.name, item.mass_kg / 2, item.x_m) for item in masses.items),
    )


def print_comparison(single, twin) -> None:
    rows = [
        ("Aircraft mass [kg]", single.aircraft_mass_kg, twin.aircraft_mass_kg),
        ("Wing root shear, limit [N]", single.wing_cases.limit.root_shear_force_N,
         twin.wing_cases.limit.root_shear_force_N),
        ("Wing root bending, ultimate [N m]", single.wing_cases.ultimate.root_bending_moment_N_m,
         twin.wing_cases.ultimate.root_bending_moment_N_m),
        ("Wing root torque, limit [N m]", single.wing_cases.limit.root_torque_N_m,
         twin.wing_cases.limit.root_torque_N_m),
        ("Wing root Z required [mm^3]", single.wing_required.required_section_modulus_m3[0] * 1e9,
         twin.wing_required.required_section_modulus_m3[0] * 1e9),
        ("Wing uniform EI required [N m^2]", single.wing_required.required_uniform_flexural_rigidity_N_m2,
         twin.wing_required.required_uniform_flexural_rigidity_N_m2),
        ("Tail lift, limit [N]", single.limit.tail_lift_N, twin.limit.tail_lift_N),
        ("Tailplane max |M|, ultimate [N m]", abs(single.tail_cases.ultimate.bending_moment_N_m).max(),
         abs(twin.tail_cases.ultimate.bending_moment_N_m).max()),
    ]
    print(f"\n{'':36}{'single':>10}{'twin':>10}{'change in size':>16}")
    for label, single_value, twin_value in rows:
        change = (abs(twin_value) - abs(single_value)) / abs(single_value) if single_value else float("nan")
        print(f"{label:36}{single_value + 0.0:10.1f}{twin_value + 0.0:10.1f}{change:16.0%}")


if __name__ == "__main__":
    print("--- Single fuselage ---")
    single = run_structural_analysis(build_reference_aircraft(), FUSELAGE_MASSES)

    print("--- Twin fuselage ---")
    twin_aircraft = build_twin_fuselage_aircraft()
    print_geometry(twin_aircraft)
    twin = run_structural_analysis(twin_aircraft, {"fuselages": half_of(FUSELAGE_MASSES["fuselage"])},
                                   tail_fuselage="fuselages", output_dir=TWIN_OUTPUT_DIR)
    print_results(twin)
    save_figures(twin, "twin", TWIN_OUTPUT_DIR)

    print_comparison(single, twin)
