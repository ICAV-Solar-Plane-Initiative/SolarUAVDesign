# Roadmap

## Goal

Size a solar-powered UAV so that its 24-hour energy balance closes with margin at
a chosen design latitude and date.

---

## 1. The optimisation problem

Anyone can run an optimiser. The work is in posing a problem that is well posed
(it has a meaningful, unique answer) and tractable (it converges reliably and quickly).

### 1.1 Objective

**Maximise the charge margin:** the battery energy left over at sunrise, after one
full day–night cycle.

- Start the 24-hour simulation at sunrise with the battery at its reserve state of charge.
- Fly the day: solar income charges the battery, clipped at full charge.
- Fly the night: cruise power drains the battery.
- At the next sunrise, the charge margin is

      t_margin [h] = (E_battery,sunrise − E_reserve) / P_cruise

  This is how long the aircraft could keep flying after sunrise with zero solar income.
  A positive margin means the energy cycle closes, so the aircraft can fly indefinitely.

Alternative formulation, worth supporting later: minimise mass or span subject to
t_margin ≥ t_margin,min.

### 1.2 Design variables

| Group         | Variables                                                     | Notes                                     |
|---------------|---------------------------------------------------------------|-------------------------------------------|
| Planform      | Chord, segment span and twist at N stations per surface        | Start with N = 2–3 per surface            |
| Configuration | x, z position of each surface from the nose datum              | Main wing and tail                        |
| Cruise        | Cruise velocity                                                | Altitude held as a fixed parameter at first |
| Battery       | Battery mass, battery x-position                               | Mass sets stored energy via energy density |
| Solar         | Fraction of the wing upper surface covered by panels           | Smooth stand-in for the packing problem   |

**Fixed parameters** (inputs, not optimised): airfoil sections, payload and avionics
mass, design latitude and date, cruise altitude, material properties, load and
safety factors.

### 1.3 Solved inside the analysis, not by the optimiser

Handing these to the optimiser as equality constraints makes convergence slow and
fragile. Each analysis call solves them directly.

- **Trim:** find the angle of attack and tail incidence that give L = W and zero
  pitching moment about the CG at the cruise condition.
- **Mass closure:** total mass = structure + battery + panels + avionics + payload.
  Iterate if structural mass depends on loads, which depend on mass.

### 1.4 Constraints

All are inequalities, written in the form g(x) ≤ 0.

| Constraint                  | Form                                             |
|-----------------------------|--------------------------------------------------|
| Trim is achievable          | Trim angle of attack and tail incidence within limits |
| Stall margin                | V_cruise ≥ 1.2 V_stall                            |
| Static stability            | SM_min ≤ static margin ≤ SM_max                   |
| Strength                    | Stress ≤ allowable at the ultimate load factor    |
| Stiffness                   | Wing tip deflection ≤ limit at the limit load     |
| Battery placement           | Battery lies within the fuselage bounds          |
| Panel area                  | Panel area ≤ usable wing upper-surface area       |
| Size and mass limits        | span ≤ b_max, total mass ≤ m_max (from rules, transport or hangar) |

### 1.5 Keeping it tractable

- Scale all design variables and constraints to order 1.
- Keep every model smooth in the design variables. No discrete choices inside the loop:
  - use a packing efficiency factor for the panels, and solve the real panel layout as a
    post-processing step;
  - use a smooth approximation where the battery clips at full charge, since a hard clip
    puts a kink in the objective.
- Start with 3–4 variables (span, chord, cruise speed, battery mass). Add more only once
  the small problem converges reliably.
- Use finite-difference gradients at first, and check them before trusting the optimiser.
- Every discipline must run and be tested on its own before it joins the loop.

---

## 2. Disciplines

### 2.1 Geometry (exists)
- Surfaces defined by stations, assembled into an aircraft.
- Derived quantities (area, span, MAC, aspect ratio) computed on demand.

### 2.2 Aerodynamics
- Lifting-line theory using airfoil lift curve slope and Cm0.
- Viscous drag: section profile drag plus fuselage and tail parasite drag.
- Power required: P = D·V / (η_prop · η_motor), from the computed drag or from a
  user-specified polar.

### 2.3 Mass
- Component mass breakdown: structure, battery, panels, avionics, payload.
- Empirical scaling for wing and fuselage structural mass to start; replaced by the
  structural model later.
- Centre of gravity from component masses and positions.

### 2.4 Structures
- Wing as a beam: bending and torsion under the ultimate load factor.
- Flexural rigidity (EI) and torsional rigidity (GJ) from beam theory.
- Feeds structural mass back into the mass model.

### 2.5 Stability and trim
- Forces and moments about the CG.
- Neutral point and static margin.
- Trim solve (see 1.3).

### 2.6 Solar income
- Irradiance as a function of latitude, date, time of day and altitude.
- Incidence on the panels, including the aircraft's attitude.
- Panel area, cell efficiency and packing efficiency give electrical power in.
- Box packing of the real panel layout, as a post-processing step only.

### 2.7 Battery and energy
- State-of-charge time-stepping over 24 hours.
- MPPT and charge-controller efficiency, charge and discharge efficiency.
- Battery mass to capacity via energy density; reserve state of charge.
- Output: the charge margin (1.1).

### 2.8 Optimisation
- Problem definition: variables, bounds, scaling, objective and constraints.
- Wrapper around `scipy.optimize` (SLSQP to start).

### 2.9 Plotting
- 2D three-view.
- Constraint diagram (power loading vs wing loading).
- Drag and power polars.
- 24-hour energy plot: solar power in, power out, state of charge.

---

## 3. Package structure

    solaruav/
        geometry/       aircraft.py, surface.py              (exists)
        aerodynamics/   lifting_line.py, drag.py, power.py
        mass/           components.py, scaling.py, cg.py
        structures/     loads.py, beam.py
        stability/      trim.py, static_margin.py
        solar/          irradiance.py, panels.py
        energy/         battery.py, mission.py   (24-hour simulation, charge margin)
        optimisation/   problem.py, run.py
        plotting/       three_view.py, constraint_diagram.py, polars.py, energy.py

`plotting.py` stays a single file until it outgrows one job, then becomes the package above.

---

## 4. Milestones

Each milestone ends with tests and a check against an independent result,
run on the reference aircraft in `examples/`.

| #  | Milestone               | Deliverable                                   | Validate against                                    |
|----|-------------------------|-----------------------------------------------|-----------------------------------------------------|
| M0 | Geometry                | Stations, derived properties, three-view (done) | Hand calculation                                  |
| M1 | Aerodynamics            | Lift distribution, drag, power-required curve | Elliptic-wing result CDi = CL²/(πAR); AVL or XFLR5 |
| M2 | Mass and CG             | Mass breakdown and CG position                | Weighed reference or prototype aircraft             |
| M3 | Trim and stability      | Trim state, static margin                     | AVL                                                 |
| M4 | Solar and energy        | 24-hour SoC trace and charge margin for a fixed design | PVGIS irradiance data                     |
| M5 | Structures              | Bending stress, tip deflection, structural mass | Hand beam calculation; FE if available            |
| M6 | Optimisation            | Optimised design from the 3–4 variable problem, then the full set | Sensible trends; result agrees from several starting points |

M4 and M5 can run in parallel. M4 is the first end-to-end analysis: once it works,
the team can evaluate the charge margin of any hand-drawn design before the optimiser exists.
