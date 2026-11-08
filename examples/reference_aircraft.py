"""The reference solar UAV, from geometry to required structural section properties.

Run from the repository root:

    python examples/reference_aircraft.py

The analysis follows four steps (see ``run_structural_analysis``):

1. Geometry: wing, tail surfaces and fuselage.
2. Aerodynamics: lifting-line analysis of the wing at cruise (L = W) and at
   the limit pull-up (L = n W).
3. Loads: each fuselage as a beam carrying its structure and mass items,
   supported at its wing mount; then the wing loads, including the force and
   pitching moment each fuselage puts into the wing at its mount.
4. Required section properties at the ultimate load, for every component:
   section modulus for strength (wing and fuselage), EI and GJ for the wing's
   stiffness limits. Each component's results go to output/<name>.csv.

Figures are saved to output/ as well. ``examples/twin_fuselage.py`` runs the
same steps on a twin-fuselage version of this aircraft.

The numbers here are a placeholder concept (about 5 m span, single fuselage,
conventional tail), not a design. The airfoil, masses, load factors,
allowable stress and stiffness limits are representative guesses, to be
replaced with real data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from solaruav import Aircraft, Fuselage, LiftingSurface
from solaruav.aerodynamics import Airfoil, LiftingLineResult, solve_lifting_line_for_lift
from solaruav.plotting import plot_fuselage_loads, plot_lifting_line, plot_section_properties, plot_wing_loads
from solaruav.structures import (
    DesignLoadCases,
    FuselageLoads,
    MassItem,
    RequiredFuselageSectionProperties,
    RequiredSectionProperties,
    chord_proportional_mass,
    design_load_cases,
    elastic_axis_x_m,
    fuselage_loads,
    required_fuselage_section_properties,
    required_section_properties,
    write_section_properties_csv,
)
from solaruav.structures.loads import STANDARD_GRAVITY_M_S2

OUTPUT_DIR = Path("output")  # CSV files and figures go here (ignored by git)

# ---------------------------------------------------------------------- #
# Placeholder inputs
# ---------------------------------------------------------------------- #
# A cambered low-Reynolds-number section, roughly like those used on small
# solar aircraft. Replace with XFOIL data for the chosen airfoil.
WING_AIRFOIL = Airfoil(lift_slope_per_rad=5.7, alpha_zero_lift_deg=-3.0, cm0=-0.08)

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

# Structural requirements. The allowable stress is for the intended material
# (here a placeholder for a carbon-fibre laminate, where compression governs).
ALLOWABLE_STRESS_PA = 400e6
MAX_TIP_DEFLECTION_FRACTION = 0.10  # of the semispan, at limit load
MAX_TIP_TWIST_DEG = 2.0             # elastic twist at limit load


@dataclass(frozen=True)
class FuselageMasses:
    """What one fuselage body carries: its structure [kg] and its mass items."""

    structure_mass_kg: float
    items: tuple[MassItem, ...]

    @property
    def total_kg(self) -> float:
        return self.structure_mass_kg + sum(item.mass_kg for item in self.items)


# Masses carried by the single fuselage (body-axis x of each item; the nose is ahead of the wing).
FUSELAGE_MASSES = {
    "fuselage": FuselageMasses(
        structure_mass_kg=0.5,
        items=(
            MassItem("motor", 0.3, x_m=-0.55),
            MassItem("avionics + payload", 0.6, x_m=-0.30),
            MassItem("battery", 1.5, x_m=0.05),
            MassItem("tail surfaces", 0.3, x_m=1.68),
        ),
    ),
}


# ---------------------------------------------------------------------- #
# Step 1: geometry
# ---------------------------------------------------------------------- #
def build_wing() -> LiftingSurface:
    """Main wing: rectangular centre section with tapered outer panels."""
    return LiftingSurface.from_panels(
        "wing",
        root_chord=0.32,
        panels=[
            {"span": 1.5, "taper": 1.0, "twist_deg": 10.0, "sweep_le_deg": 0.0},
            {"span": 1.0, "taper": 0.6, "sweep_le_deg": 3.0},
        ],
        symmetric=True,
        origin=(0.0, 0.0, 0.0),
    )


def build_fuselage_pod(name: str, origin_y: float = 0.0, symmetric: bool = False) -> Fuselage:
    """A pod with a pointed nose ahead of the wing and a thin boom to the tail."""
    return Fuselage.from_segments(
        name,
        nose_width=0.0,
        nose_height=0.0,
        segments=[
            {"length": 0.15, "width": 0.12, "height": 0.14},  # nose cone
            {"length": 0.60, "width": 0.12, "height": 0.14},  # payload and battery bay
            {"length": 0.25, "width": 0.03, "height": 0.03},  # taper into the boom
            {"length": 1.35, "width": 0.02, "height": 0.02},  # tail boom
        ],
        origin=(-0.6, origin_y, -0.03),
        symmetric=symmetric,
    )


def build_reference_aircraft() -> Aircraft:
    htail = LiftingSurface.trapezoidal(
        "htail", span=1.0, root_chord=0.20, taper=0.8, sweep_le_deg=5.0,
        symmetric=True, origin=(1.60, 0.0, 0.0),
    )
    fin = LiftingSurface.trapezoidal(
        "fin", span=0.40, root_chord=0.28, taper=0.6, sweep_le_deg=25.0,
        symmetric=False, orientation="vertical", origin=(1.58, 0.0, 0.0),
    )
    return Aircraft(
        "Reference concept",
        surfaces=[build_wing(), htail, fin],
        reference_surface="wing",
        fuselages=[build_fuselage_pod("fuselage")],
    )


# ---------------------------------------------------------------------- #
# Steps 2-4: aerodynamics, loads, required section properties
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class StructuralAnalysis:
    """Everything the four steps produce, for printing and plotting."""

    aircraft: Aircraft
    aircraft_mass_kg: float
    cruise: LiftingLineResult
    fuselage_limit_loads: dict[str, FuselageLoads]
    wing_cases: DesignLoadCases
    wing_required: RequiredSectionProperties
    fuselage_required: dict[str, RequiredFuselageSectionProperties]
    csv_paths: list[Path]


def aircraft_mass_kg(aircraft: Aircraft, fuselage_masses: dict[str, FuselageMasses]) -> float:
    """Wing mass plus every fuselage body's structure and items [kg]."""
    fuselage_total = sum(fuselage_masses[f.name].total_kg * f.n_bodies for f in aircraft.fuselages)
    return WING_MASS_KG + fuselage_total


def run_structural_analysis(
    aircraft: Aircraft, fuselage_masses: dict[str, FuselageMasses], output_dir: Path = OUTPUT_DIR
) -> StructuralAnalysis:
    """Steps 2-4 for an aircraft whose geometry (step 1) is built.

    Inputs: the aircraft (must have a surface named "wing"); the masses each
    fuselage body carries, keyed by fuselage name; the output folder.
    Output: a StructuralAnalysis. Writes one CSV per component to output_dir.
    """
    wing = aircraft["wing"]
    total_mass_kg = aircraft_mass_kg(aircraft, fuselage_masses)

    # Step 2: aerodynamics. Cruise here; the limit pull-up is solved inside
    # design_load_cases at L = n W.
    cruise = solve_lifting_line_for_lift(
        wing, total_mass_kg * STANDARD_GRAVITY_M_S2, CRUISE_AIRSPEED_M_S, AIR_DENSITY_KG_M3,
        airfoil=WING_AIRFOIL, n_output=201,
    )

    # Step 3a: each fuselage body as a beam supported at its wing mount, which
    # sits on the wing's elastic axis at the fuselage's spanwise station.
    fuselage_limit = {}
    for fuselage in aircraft.fuselages:
        masses = fuselage_masses[fuselage.name]
        mount_s_m = abs(fuselage.origin[1])
        fuselage_limit[fuselage.name] = fuselage_loads(
            fuselage, masses.items, masses.structure_mass_kg,
            mount_x_m=elastic_axis_x_m(wing, mount_s_m, SPAR_CHORD_FRACTION),
            load_factor=LIMIT_LOAD_FACTOR,
        )

    # Step 3b: wing loads, with each fuselage's mount reaction applied where it attaches.
    wing_cases = design_load_cases(
        wing, WING_AIRFOIL, total_mass_kg,
        mass_per_span_kg_m=chord_proportional_mass(wing, WING_MASS_KG),
        airspeed_m_s=MANOEUVRE_AIRSPEED_M_S, air_density_kg_m3=AIR_DENSITY_KG_M3,
        limit_load_factor=LIMIT_LOAD_FACTOR, safety_factor=SAFETY_FACTOR,
        elastic_axis_chord_fraction=SPAR_CHORD_FRACTION,
        mass_chord_fraction=WING_MASS_CHORD_FRACTION,
        point_loads=[loads.wing_point_load() for loads in fuselage_limit.values()],
    )

    # Step 4: required section properties at the ultimate load, every component.
    wing_required = required_section_properties(
        wing_cases, ALLOWABLE_STRESS_PA,
        max_tip_deflection_fraction=MAX_TIP_DEFLECTION_FRACTION, max_tip_twist_deg=MAX_TIP_TWIST_DEG,
    )
    fuselage_required = {
        name: required_fuselage_section_properties(loads.scaled(SAFETY_FACTOR), ALLOWABLE_STRESS_PA)
        for name, loads in fuselage_limit.items()
    }
    csv_paths = [write_section_properties_csv(p, output_dir) for p in [wing_required, *fuselage_required.values()]]

    return StructuralAnalysis(aircraft, total_mass_kg, cruise, fuselage_limit, wing_cases,
                              wing_required, fuselage_required, csv_paths)


# ---------------------------------------------------------------------- #
# Reporting
# ---------------------------------------------------------------------- #
def print_geometry(ac: Aircraft) -> None:
    for row in ac.summary():
        print(
            f"{row['name']:>14}:  S = {row['area_m2']:.3f} m^2   b = {row['span_m']:.2f} m   "
            f"AR = {row['aspect_ratio']:.1f}   MAC = {row['mac_m']:.3f} m   x_ac = {row['x_ac_body_m']:.3f} m"
        )
    for row in ac.fuselage_summary():
        print(
            f"{row['name']:>14}:  {row['n_bodies']} x (length {row['length_m']:.2f} m, "
            f"volume {row['volume_m3'] * 1e3:.1f} L, wetted area {row['wetted_area_m2']:.3f} m^2, "
            f"frontal area {row['frontal_area_m2'] * 1e4:.0f} cm^2)"
        )


def print_results(analysis: StructuralAnalysis) -> None:
    cruise, cases, wing_required = analysis.cruise, analysis.wing_cases, analysis.wing_required
    print(f"\nAircraft mass {analysis.aircraft_mass_kg:.2f} kg")
    print(f"Cruise (V = {CRUISE_AIRSPEED_M_S:.1f} m/s): alpha = {cruise.alpha_deg:.2f} deg   "
          f"CL = {cruise.lift_coefficient:.3f}   CDi = {cruise.induced_drag_coefficient:.4f}   "
          f"e = {cruise.span_efficiency:.3f}")
    aero = cases.limit_aero
    print(f"Limit pull-up (V = {MANOEUVRE_AIRSPEED_M_S:.1f} m/s, n = {LIMIT_LOAD_FACTOR:.1f}): "
          f"alpha = {aero.alpha_deg:.2f} deg   peak section Cl = {aero.local_lift_coefficient.max():.2f} "
          "(check against Cl_max)")

    for name, loads in analysis.fuselage_limit_loads.items():
        required = analysis.fuselage_required[name]
        print(f"Fuselage '{name}' (one body), mount at s = {loads.mount_s_m:.2f} m: limit reaction "
              f"{loads.mount_force_N:.1f} N down, {loads.mount_moment_N_m:+.2f} N m nose-up; "
              f"ultimate max |M| = {required.ultimate_loads.max_bending_moment_N_m:.1f} N m, "
              f"Z = {required.max_required_section_modulus_m3 * 1e9:.0f} mm^3")

    limit, ultimate = cases.limit, cases.ultimate
    print(f"Wing root, one side: limit V = {limit.root_shear_force_N:.1f} N, "
          f"M = {limit.root_bending_moment_N_m:.1f} N m, T = {limit.root_torque_N_m:.2f} N m;  "
          f"ultimate M = {ultimate.root_bending_moment_N_m:.1f} N m")
    print(f"Wing required: root Z = {wing_required.required_section_modulus_m3[0] * 1e9:.0f} mm^3   "
          f"uniform EI = {wing_required.required_uniform_flexural_rigidity_N_m2:.0f} N m^2   "
          f"uniform GJ = {wing_required.required_uniform_torsional_rigidity_N_m2:.1f} N m^2")
    print("Results written to " + ", ".join(str(path) for path in analysis.csv_paths))


def save_figures(analysis: StructuralAnalysis, prefix: str, output_dir: Path = OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    wing = analysis.aircraft["wing"]
    figures = {
        f"{prefix}_aircraft.png": analysis.aircraft.plot()[0],
        f"{prefix}_lifting_line.png": plot_lifting_line(analysis.cruise, wing)[0],
        f"{prefix}_wing_loads.png": plot_wing_loads(analysis.wing_cases, analysis.cruise)[0],
        f"{prefix}_wing_section_properties.png": plot_section_properties(analysis.wing_required)[0],
    }
    for name, loads in analysis.fuselage_limit_loads.items():
        figures[f"{prefix}_{name}_loads.png"] = plot_fuselage_loads(loads, analysis.fuselage_required[name])[0]
    for file_name, figure in figures.items():
        figure.savefig(output_dir / file_name, dpi=150)
    print(f"Figures saved to {output_dir}/")


if __name__ == "__main__":
    reference = build_reference_aircraft()  # step 1
    print_geometry(reference)
    analysis = run_structural_analysis(reference, FUSELAGE_MASSES)  # steps 2-4
    print_results(analysis)
    save_figures(analysis, "reference")
    plt.show()
