%% Code development


% Posing the optimisation problem

Anyone can run an optimizer, how do we make the problem well poised, and tractable

Main Optimisation Loop 
    Objective function:
        Maximise the charge margin that the vehicle has (i.e. excess battery charge duration under the asumption of no power income)
    Design variables:
        Section control: specify a number key segments per surface
            Chord
            Span
            Twist 
        Config control:  specify distances between surfaces and set datum
            X Y Z pos from nose datum
        Cruise condition:
            Cruise velocity

    Constraints:
        Ineq Constr.
            Minimum time margin (robustness)
            cruise speed > stall speed
            Battery geometric position constraints

        Eq Constr.
            Zero moment, L = W at cruise condtion
            Structure withstands ultimate loads 



Sub processes:

- Solar income model
    - local solar energy (time and place dependent)
    - panel arrangement (box packing problem)
Battery model (SoC)
    - MPPT /  Balance charging
    - Battery mass sizing/scaling
    - Battery locations
- Aerodynamic Model
    - Lifting line theory (with airfoil Cm0 and lift curve slope)
    - Viscous drag model
    - Power consumption estimation (uses result of drag or with specified polar)
- Structural Model
    - Loose empirical scaling model (wing and fuselage specific)
    - Calculate structural qualities (flexural and torsional rigidity, from beam theory)
    - Calculate CG
- Stability and trim
    - Evaluate mo
- Plotting utils




% Package Structure

    solaruav
        plotting # the module has all of plotting utilities
            2D 3 view plot
            Constraint diagram
            Forcing Polars
        geometry # define the aircraft
            aircraft
            surface
        aerodynamic
            discretize
            
        
            

% Aircraft Geometry and Attributes


