"""The reference solar UAV, from geometry to required structural section properties.

Run from the repository root:

    python examples/reference_aircraft.py

The analysis follows four steps (see ``run_structural_analysis``):

1. Geometry: wing, tail surfaces and fuselage.
2. Aerodynamics: trim at cruise (n = 1) and in the limit pull-up (n = n_limit).
   The centre of gravity comes from the mass breakdown; the wing and tail
   lifts are split so that together they carry n W with zero pitching moment
   about the CG, and each surface is solved by lifting line at its trimmed lift.
3. Loads:
   a. each fuselage as a beam carrying its structure, its mass items and its
      share of the tail lift, supported at its wing mount;
   b. the wing, with each fuselage's mount reaction applied where it attaches;
   c. the tailplane, held up where it meets the boom(s).
4. Required section properties at the ultimate load, for every component:
   section modulus for strength (wing, tailplane, fuselage), EI and GJ for
   the wing's and tailplane's stiffness limits. Each component's results go
   to output/<name>.csv.

Figures are saved to output/ as well. ``examples/twin_fuselage.py`` runs the
same steps on a twin-fuselage version of this aircraft.

The numbers here are a placeholder concept (about 5 m span, single fuselage,
conventional tail), not a design. The airfoils, masses, load factors,
allowable stress and stiffness limits are representative guesses, to be
replaced with real data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from solaruav import Aircraft, Fuselage, LiftingSurface
from solaruav.aerodynamics import Airfoil
from solaruav.mass import centre_of_gravity_x_m
from solaruav.plotting import plot_fuselage_loads, plot_lifting_line, plot_section_properties, plot_wing_loads
from solaruav.stability import TrimmedFlight, trim_symmetric_flight
from solaruav.structures import (
    DesignLoadCases,
    FuselageLoads,
    MassItem,
    PointForce,
    RequiredFuselageSectionProperties,
    RequiredSectionProperties,
    chord_proportional_mass,
    design_load_cases,
    elastic_axis_x_m,
    fuselage_loads,
    required_fuselage_section_properties,
    required_section_properties,
    structure_centroid_x_m,
    wing_mass_centroid_x_m,
    write_section_properties_csv,
)

OUTPUT_DIR = Path("output")  # CSV files and figures go here (ignored by git)

# ---------------------------------------------------------------------- #
# Placeholder inputs
# ---------------------------------------------------------------------- #
# A cambered low-Reynolds-number section, roughly like those used on small
# solar aircraft, and a symmetric section for the tail. Replace with XFOIL data.
WING_AIRFOIL = Airfoil(lift_slope_per_rad=5.7, alpha_zero_lift_deg=-3.0, cm0=-0.08)
TAIL_AIRFOIL = Airfoil(lift_slope_per_rad=5.7)

WING_MASS_KG = 1.8           # structure + solar panels carried in the wing, both sides
TAILPLANE_MASS_KG = 0.2      # part of the "tail surfaces" item below; used for the tailplane's own beam
CRUISE_AIRSPEED_M_S = 10.0
AIR_DENSITY_KG_M3 = 1.225    # sea level, ISA

# The limit pull-up is flown at a manoeuvre speed, not at cruise: a 3 g pull-up
# at 10 m/s would need section Cl far beyond stall. Set this from the stall
# speed once Cl_max is known: V_A = V_stall sqrt(n_limit).
MANOEUVRE_AIRSPEED_M_S = 14.0
LIMIT_LOAD_FACTOR = 3.0
SAFETY_FACTOR = 1.5
SPAR_CHORD_FRACTION = 0.25   # elastic axis, fraction of chord from the leading edge
MASS_CHORD_FRACTION = 0.40   # centre of the wing and tailplane mass, fraction of chord

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
    cg_x_m: float
    cruise: TrimmedFlight
    limit: TrimmedFlight
    fuselage_limit_loads: dict[str, FuselageLoads]
    wing_cases: DesignLoadCases
    tail_cases: DesignLoadCases
    wing_required: RequiredSectionProperties
    tail_required: RequiredSectionProperties
    fuselage_required: dict[str, RequiredFuselageSectionProperties]
    csv_paths: list[Path]


def wing_mass_distribution(wing: LiftingSurface):
    return chord_proportional_mass(wing, WING_MASS_KG)


def mass_breakdown(aircraft: Aircraft, fuselage_masses: dict[str, FuselageMasses]) -> tuple[list, list]:
    """Every mass on the aircraft and its body-axis x: (masses [kg], x [m]), two lists.

    The wing's distributed mass and each fuselage body's structure and items.
    A mirrored fuselage counts twice (both bodies share the same x).
    """
    wing = aircraft["wing"]
    masses = [WING_MASS_KG]
    positions = [wing_mass_centroid_x_m(wing, wing_mass_distribution(wing), MASS_CHORD_FRACTION)]
    for fuselage in aircraft.fuselages:
        carried = fuselage_masses[fuselage.name]
        masses += [fuselage.n_bodies * carried.structure_mass_kg]
        positions += [structure_centroid_x_m(fuselage)]
        masses += [fuselage.n_bodies * item.mass_kg for item in carried.items]
        positions += [item.x_m for item in carried.items]
    return masses, positions


def run_structural_analysis(
    aircraft: Aircraft,
    fuselage_masses: dict[str, FuselageMasses],
    tail_fuselage: str = "fuselage",
    output_dir: Path = OUTPUT_DIR,
) -> StructuralAnalysis:
    """Steps 2-4 for an aircraft whose geometry (step 1) is built.

    Inputs: the aircraft (needs surfaces named "wing" and "htail"); the masses
    each fuselage body carries, keyed by fuselage name; the name of the
    fuselage whose boom(s) carry the tailplane; the output folder.
    Output: a StructuralAnalysis. Writes one CSV per component to output_dir.
    """
    wing, htail = aircraft["wing"], aircraft["htail"]
    masses, positions = mass_breakdown(aircraft, fuselage_masses)
    total_mass_kg, cg_x_m = sum(masses), centre_of_gravity_x_m(masses, positions)

    # Step 2: trim. Split n W between wing and tail with zero moment about the CG,
    # and solve each surface's lifting line at its trimmed lift.
    def trim(load_factor, airspeed_m_s):
        return trim_symmetric_flight(wing, WING_AIRFOIL, htail, TAIL_AIRFOIL, total_mass_kg, cg_x_m,
                                     load_factor, airspeed_m_s, AIR_DENSITY_KG_M3)

    cruise = trim(1.0, CRUISE_AIRSPEED_M_S)
    limit = trim(LIMIT_LOAD_FACTOR, MANOEUVRE_AIRSPEED_M_S)

    # Step 3a: each fuselage body as a beam supported at its wing mount (on the
    # wing's elastic axis at its spanwise station). The tail-carrying bodies
    # share the tail lift, applied at the tail's aerodynamic centre.
    tail_carrier = aircraft[tail_fuselage]
    tail_force = PointForce("htail", limit.tail_lift_N / tail_carrier.n_bodies, htail.x_ac_body)
    fuselage_limit = {}
    for fuselage in aircraft.fuselages:
        carried = fuselage_masses[fuselage.name]
        fuselage_limit[fuselage.name] = fuselage_loads(
            fuselage, carried.items, carried.structure_mass_kg,
            mount_x_m=elastic_axis_x_m(wing, abs(fuselage.origin[1]), SPAR_CHORD_FRACTION),
            load_factor=LIMIT_LOAD_FACTOR,
            point_forces=[tail_force] if fuselage.name == tail_fuselage else [],
        )

    # Step 3b: wing loads, with each fuselage's mount reaction where it attaches.
    common = dict(safety_factor=SAFETY_FACTOR, elastic_axis_chord_fraction=SPAR_CHORD_FRACTION,
                  mass_chord_fraction=MASS_CHORD_FRACTION)
    wing_cases = design_load_cases(
        limit.wing, wing, WING_AIRFOIL, LIMIT_LOAD_FACTOR, wing_mass_distribution(wing),
        point_loads=[loads.wing_point_load() for loads in fuselage_limit.values()], **common,
    )

    # Step 3c: tailplane loads, held up where it meets the boom(s): at the
    # centreline for a single boom, at y = +-y_boom for twin booms.
    tail_cases = design_load_cases(
        limit.tail, htail, TAIL_AIRFOIL, LIMIT_LOAD_FACTOR, chord_proportional_mass(htail, TAILPLANE_MASS_KG),
        support_s_m=abs(tail_carrier.origin[1]), **common,
    )

    # Step 4: required section properties at the ultimate load, every component.
    stiffness_limits = dict(max_tip_deflection_fraction=MAX_TIP_DEFLECTION_FRACTION,
                            max_tip_twist_deg=MAX_TIP_TWIST_DEG)
    wing_required = required_section_properties(wing_cases, ALLOWABLE_STRESS_PA, **stiffness_limits)
    tail_required = required_section_properties(tail_cases, ALLOWABLE_STRESS_PA, **stiffness_limits)
    fuselage_required = {
        name: required_fuselage_section_properties(loads.scaled(SAFETY_FACTOR), ALLOWABLE_STRESS_PA)
        for name, loads in fuselage_limit.items()
    }
    components = [wing_required, tail_required, *fuselage_required.values()]
    csv_paths = [write_section_properties_csv(properties, output_dir) for properties in components]

    return StructuralAnalysis(aircraft, total_mass_kg, cg_x_m, cruise, limit, fuselage_limit, wing_cases,
                              tail_cases, wing_required, tail_required, fuselage_required, csv_paths)


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


def print_trim(name: str, trim: TrimmedFlight, airspeed_m_s: float) -> None:
    print(f"{name} (n = {trim.load_factor:.1f}, V = {airspeed_m_s:.1f} m/s): "
          f"wing lift {trim.wing_lift_N:.1f} N at alpha = {trim.wing.alpha_deg:.2f} deg "
          f"(CL = {trim.wing.lift_coefficient:.3f}, peak section Cl = {trim.wing.local_lift_coefficient.max():.2f}); "
          f"tail lift {trim.tail_lift_N:+.2f} N (CL_t = {trim.tail.lift_coefficient:+.3f}); "
          f"residual moment {trim.pitching_moment_N_m:.1e} N m")


def print_results(analysis: StructuralAnalysis) -> None:
    mac = analysis.aircraft.reference
    cg_fraction = (analysis.cg_x_m - mac.origin[0] - mac.x_le_mac) / mac.mac
    print(f"\nAircraft mass {analysis.aircraft_mass_kg:.2f} kg, CG at x = {analysis.cg_x_m:.3f} m "
          f"({cg_fraction:.0%} of the wing MAC)")
    print_trim("Cruise", analysis.cruise, CRUISE_AIRSPEED_M_S)
    print_trim("Limit pull-up", analysis.limit, MANOEUVRE_AIRSPEED_M_S)
    print(f"  cruise CDi = {analysis.cruise.wing.induced_drag_coefficient:.4f}, "
          f"e = {analysis.cruise.wing.span_efficiency:.3f}; check peak section Cl against Cl_max")

    for name, loads in analysis.fuselage_limit_loads.items():
        required = analysis.fuselage_required[name]
        print(f"Fuselage '{name}' (one body), mount at s = {loads.mount_s_m:.2f} m: limit reaction "
              f"{loads.mount_force_N:.1f} N down, {loads.mount_moment_N_m:+.2f} N m nose-up; "
              f"ultimate max |M| = {required.ultimate_loads.max_bending_moment_N_m:.1f} N m, "
              f"Z = {required.max_required_section_modulus_m3 * 1e9:.0f} mm^3")

    for label, cases, required in [("Wing", analysis.wing_cases, analysis.wing_required),
                                   ("Tailplane", analysis.tail_cases, analysis.tail_required)]:
        ultimate = cases.ultimate
        print(f"{label}, one side: ultimate root V = {ultimate.root_shear_force_N:.1f} N, "
              f"root M = {ultimate.root_bending_moment_N_m:.2f} N m, "
              f"max |M| = {abs(ultimate.bending_moment_N_m).max():.2f} N m;  required max Z = "
              f"{required.required_section_modulus_m3.max() * 1e9:.1f} mm^3, "
              f"uniform EI = {required.required_uniform_flexural_rigidity_N_m2:.1f} N m^2, "
              f"uniform GJ = {required.required_uniform_torsional_rigidity_N_m2:.2f} N m^2")
    print("Results written to " + ", ".join(str(path) for path in analysis.csv_paths))


def save_figures(analysis: StructuralAnalysis, prefix: str, output_dir: Path = OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    wing, htail = analysis.aircraft["wing"], analysis.aircraft["htail"]
    figures = {
        f"{prefix}_aircraft.png": analysis.aircraft.plot()[0],
        f"{prefix}_wing_lifting_line.png": plot_lifting_line(analysis.cruise.wing, wing)[0],
        f"{prefix}_htail_lifting_line.png": plot_lifting_line(analysis.limit.tail, htail)[0],
        f"{prefix}_wing_loads.png": plot_wing_loads(analysis.wing_cases, analysis.cruise.wing)[0],
        f"{prefix}_htail_loads.png": plot_wing_loads(analysis.tail_cases, analysis.cruise.tail)[0],
        f"{prefix}_wing_section_properties.png": plot_section_properties(analysis.wing_required)[0],
        f"{prefix}_htail_section_properties.png": plot_section_properties(analysis.tail_required)[0],
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
    plt.show()  # open every figure in its own window
