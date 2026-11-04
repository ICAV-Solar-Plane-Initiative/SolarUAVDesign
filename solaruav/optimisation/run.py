"""Run the optimiser.

Purpose
-------
Wrap ``scipy.optimize`` (SLSQP to start) around a defined problem, run it from
one or more starting points, and record the history.

Inputs
------
- A problem from ``problem``.
- Starting design vector(s), as arrays of length n_variables.
- Optimiser settings: tolerances, maximum iterations, finite-difference step.

Outputs
-------
- Optimised design vector and the corresponding aircraft.
- Final objective and constraint values, and whether the optimiser converged.
- Iteration history, for convergence plots.
"""
