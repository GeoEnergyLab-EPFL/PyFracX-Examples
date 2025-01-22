# PyFracX-Examples
Verification examples for the simulation of fluid-driven frictional ruptures with PyFracX

## Description 
PyFracX simulate the re-activation and propagation of fluid-induced frictional/shear ruptures allowing also full opening of the fracture. The X hints that the code can simulate multiple intersecting fractures. In PyFracX, the fractures are pre-existing and meshed accordingly.
This repository documents a comprehensive series of verification tests of this solver against existing analytical/semi-analytical solutions for this class of hydromechanical fracture problems.
This tests are for fluid-induced tensile fracture (hydraulic fracture re-opening of a pre-existing discontinuity) and frictional shear ruptures. 
The solver can simulate a number of problems, and notably cases where fluid flow can be "uncoupled" from the mechanical problem (e.g.  frictional ruptures for which the evolution of pore-pressure is not affected by mechanical deformation) or "coupled" hydromechanical problems (e.g. hydraulic fractures, shear-induced dilatant frictional ruptures). The structure of the sub-folders follows this convention.

### Important notes

You must have in your python path BigWham, PyFracX as well as the base folder of this repo (to allow proper import of reference solutions).
In each sub-folder, when running the script - if the option save_results =True at the head of the script, a sub-folder will be created where the results of the simulation will be dump. NEVER commit these files.

## Organization
This repository is organized with the following sub-folders

- ReferenceSolutions/ contains python modules containing existing reference solutions with sub-folders HF for hydraulic fracture (tensile mode I case) and FDFR for fluid-driven frictional ruptures
-  2D-uncoupled/ contains sub-folders with scripts for 2D plane elasticity configurations for frictional ruptures for which the evolution of pore-pressure is not affected by mechanical deformation such that only the mechanical part of the solver is used
- 2D-coupled/ contains sub-folders with scripts for 2D plane elasticity configurations using the coupled hydromechanical solver (thus allowing to model coupled as well uncoupled case)
-  3D-uncoupled/ contains sub-folders with scripts for 3D configurations for frictional ruptures for which the evolution of pore-pressure is not affected by mechanical deformation such that only the mechanical part of the solver is used
- 3D-coupled/ contains sub-folders with scripts for 3D configurations using the coupled hydromechanical solver (thus allowing to model coupled as well uncoupled case)
-  Axisymmetry-uncoupled/ contains sub-folders with scripts for 2D Axisymmetric configurations (circular rupture) for frictional ruptures for which the evolution of pore-pressure is not affected by mechanical deformation such that only the mechanical part of the solver is used
- Axisymmetry-coupled/ contains sub-folders with scripts for 2D Axisymmetric configurations (circular rupture) using the coupled hydromechanical solver (thus allowing to model coupled as well uncoupled case)




## Contributors

- Brice Lecampion
- Antareep Sarma
- Ankit Gupta
- Regina Fakhretdinova

