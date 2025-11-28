# PyFracX-Examples
Verification examples for the simulation of fluid-driven frictional ruptures with PyFracX

## Description 
PyFracX simulate the re-activation and propagation of fluid-induced frictional/shear ruptures allowing also full opening of the fracture. The X hints that the code can simulate multiple intersecting fractures. In PyFracX, the fractures are pre-existing and meshed accordingly.
This repository documents a comprehensive series of verification tests of this solver against existing analytical/semi-analytical solutions for this class of hydromechanical fracture problems.
This tests are for fluid-induced tensile fracture (hydraulic fracture re-opening of a pre-existing discontinuity) and frictional shear ruptures. 
The solver can simulate a number of problems, and notably cases where fluid flow can be "uncoupled" from the mechanical problem (e.g.  frictional ruptures for which the evolution of pore-pressure is not affected by mechanical deformation) or "coupled" hydromechanical problems (e.g. hydraulic fractures, shear-induced dilatant frictional ruptures). The structure of the sub-folders follows this convention.

## How to run these scripts 

To run this script you need first to compile [BigWham](https://github.com/GeoEnergyLab-EPFL/BigWham) and install `bigwham4py`. Once done you can install [PyfracX](https://github.com/GeoEnergyLab-EPFL/PyFracX). All instructions are given in the associated repositories. We recommend setting up a new python environment for it (using `venv`then installing the `bigwham4py` and `pyfracx` using pip).

Once done you should be set to execute all the examples of this repository. Each example is composed of a simulation script that will run the simulation and save the results and a post-processing script that can read the results and make the associated plots.

You can run all examples by executing :
```bash
run_all_examples.sh # run all simulations (takes up to a few hours)
run_all_postprocessing.sh # generate all associated figures
clean_figures.sh # to clean all the generated figures
```

These examples have been developed and tested only on Unix OS, some incompatibilities with Windows regarding paths management are to be expected.

## Organization
This repository is organized with the following sub-folders

- ReferenceSolutions/ contains python modules containing existing reference solutions with sub-folders HF for hydraulic fracture (tensile mode I case) and FDFR for fluid-driven frictional ruptures
-  2D-uncoupled/ contains sub-folders with scripts for 2D plane elasticity configurations for frictional ruptures for which the evolution of pore-pressure is not affected by mechanical deformation such that only the mechanical part of the solver is used
- 2D-coupled/ contains sub-folders with scripts for 2D plane elasticity configurations using the coupled hydromechanical solver (thus allowing to model coupled as well uncoupled case)
-  3D-uncoupled/ contains sub-folders with scripts for 3D configurations for frictional ruptures for which the evolution of pore-pressure is not affected by mechanical deformation such that only the mechanical part of the solver is used
- 3D-coupled/ contains sub-folders with scripts for 3D configurations using the coupled hydromechanical solver (thus allowing to model coupled as well uncoupled case)
-  Axisymmetry-uncoupled/ contains sub-folders with scripts for 2D Axisymmetric configurations (circular rupture) for frictional ruptures for which the evolution of pore-pressure is not affected by mechanical deformation such that only the mechanical part of the solver is used
- Axisymmetry-coupled/ contains sub-folders with scripts for 2D Axisymmetric configurations (circular rupture) using the coupled hydromechanical solver (thus allowing to model coupled as well uncoupled case)

## TODO 
- `Axisymmetry-coupled/Constant-Friction-Constant-Permeability/ConstantRate-SlipWeakeningFriction.py` takes too long (only 2d example that can't be run in under 1h).
- 2d HF reopening is broken
- axisym slip weakening with coupled model is broken
- stats plots for Basel are weird, should we get rid of them ?
- this is only compatible with the `cuda_dev` branch of pyfracx which is itself only compatible with the `cuda_dev` branch of bigwham



## Contributors

- Brice Lecampion
- Antareep Sarma
- Ankit Gupta
- Regina Fakhretdinova
- Sylvain Brisson

