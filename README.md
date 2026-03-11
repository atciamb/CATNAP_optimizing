# CATNAP
Coupled Adiabatic Tank and Nozzle Analysis Program - A multiphase flow and regenerative cooling solver for dual-acting vapor pressurized propulsion systems.

CATNAP couples an iterative control volume solver with multiphase flow functions, regenerative cooling functions, and NASA CEA combustion analysis to form a high fidelity estimate of propulsion system characteristics over the course of a burn. Generally, it is meant to be used
for Nitrous Oxide (N2O) and a generic fuel (Ethanol, IPA, Jet-A). 

Required inputs:
-Geometry: engine, cooling channels, and injector
-Propellants (CEA_obj class input used in combustion and transport analysis)
-Initial Conditions: saturation temperature, vapor mass fraction, total mass, target mass flow rates, target chamber pressure
-Material Properties: hotwall thermal conductivity, channel surface roughness, [maximum temperature, maximum stress] - set by user

CATNAP is unfinished! There may be bugs I have not caught.
