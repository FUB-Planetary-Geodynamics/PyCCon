# PyCCon - Python Code for Convection

A full documentation and tutorial will soon be uploaded.

#############################################################

This code can be used for simple convection simulations in a 2D box or 2D sphere,
using both thermal and chemical convection (the latter can be solved either
with a field approach or with particles for better resolution), and can use
temperature- or pressure-dependent viscosity (using the FK or Arrhenius approximation).
Convection is incompresisble with the (Extended) Boussinesq approximation or compressible with TALA approximation.

- main.ipynb: This is the main routine, please adapt the simulation parameters in the parameters cell below, assigning here the non-default parameter settings (default values will be overwritten)
- solver.py: Solves the energy, mass, momentum and chemical conservation equations
- supp_functions: Different initialization and support function, particle treatment, output measurements, and visualization (adapt as needed)
- outgassing.py: Includes outgassing calculations to build an atmosphere
- examples.py: Includes example parameter sets as outlined in tutorial, include via SimCase = NAME (e.g. NAME = "Blankenbach")
- input_default.py: Loads the default settings and parameters, do not change

#############################################################

## Python Code for Convection (PyCCon) v.2.3
@author: Lena Noack

## v2.3 Updates May 2025
- include restart option
- allow for variable aspect ratios
- include simple subduction zone set-up
- include partial melting and outgassing

## v2.2 Updates April 2025
- Added Arrhenius viscosity and plasticity
- Corrected periodic bnd conditions and solver
- Tested chemical field approach
- Added simple compressibility (EBA and TALA)

## v.2 Update in September - October 2024
- v.2: Spherical Annulus
- v.2.1: Periodicity, FDM vs FVM

## v.1 Created in Dec 2022 - Jan 2023
- 2D Cartesian Box
- Particles for chemical convection

## Future ToDo's:
- final test all benchmarks (periodic boundaries, variable aspect ratios, particles)
- animations: if output 10,20,..,90,100,... -> sometimes error in order of figures -> write figures as 00010,...
