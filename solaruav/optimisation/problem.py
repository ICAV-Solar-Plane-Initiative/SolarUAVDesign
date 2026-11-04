"""Definition of the optimisation problem.

Purpose
-------
Set out the design variables, their bounds and scaling, the objective and the
constraints, and map a design-variable vector onto an aircraft and run the
full analysis.

Inputs
------
- Baseline aircraft and fixed parameters (airfoils, payload, design latitude
  and date, altitude, materials, load factors).
- Which design variables are active, with bounds and scale factors.
- Constraint limits (stall margin, static margin range, allowable stress,
  deflection limit, span and mass limits).

Outputs
-------
- Objective value: negative charge margin, scaled (float), since optimisers minimise.
- Constraint values g(x) <= 0, scaled, as an array.
- The analysed aircraft and its results for a given design vector.
"""
