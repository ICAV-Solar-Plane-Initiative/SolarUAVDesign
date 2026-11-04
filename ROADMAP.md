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

        Eq Constr.
            Zero moment, L = W at cruise condtion 


Sub processes:

- solar income model
- charging model (SoC)
- Aerodynamic Model
    - Lifting line theory (with airfoil Cm0 and lift curve slope)
    - Viscous drag model
    - Power consumption estimation (uses result of drag or with specified polar)




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


