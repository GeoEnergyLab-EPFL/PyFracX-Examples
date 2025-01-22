# 2D Plane-strain fluid-induced shear frictional rupture on a planar fault (Coupled case)

This example demonstrates the capability of the PyFracX code to reproduce the semi-analytical solution for the problem of planar shear rupture embedded in homogeneous elastic media and driven by fluid injection. The hydraulic properties of the fault are constant (and the medium is impermeable). The dilatancy is assumed to be zero and friction coefficient is constant. Fluid injection is taken as a line source with constant injection overpressure. 

In this verification test, the problem itself remains uncoupled, but we solve it using coupled step with constant permeability, constant friction and zero dilatancy.

For this problem formulation there exists a sole dimensionless parameter T = (f_p * sigma'_o - tau_o) / (f_p * sigma'_o) bounded between 0 and 1 that governs the rupture propagation. The numerator of T represents a distance to failure under initial loading conditions, whereas the denominator is in charge of the overpressure. Parameter T determines two limiting regimes, the so-called “marginally pressurized” and “critically stressed” regimes. Each regime corresponds to different physical mechanisms governing the fault propagation. The marginally pressurized limit corresponds to the case when the fluid pressure is “just sufficient” to activate the fault by reducing the shear strength. Crack propagating in this regime will always lag the fluid front. The critically stressed limit corresponds to the case when the fault is initially close to failure. For this limit, the crack front will always outpace fluid diffusion.

The reference solution for this case can be found in the work of 
Viesca R. 2021. Self-similar fault slip in response to fluid injection. J. Fluid Mech. Vol. 928, A29, doi:10.1017/jfm.2021.825

### Overview

It can be shown that the rupture front propagation is self-similar and is related to the diffusion length scale through the fault growth factor lambda. 
In this example the simplest way to explore different propagation regimes is to vary parameter T (between 0 and 1). Slip proffiles will be automatically plotted with respect to analytical solution corresponding to the regime defined by T.

In the limit when T->1, the mesh must be refined in order to compare with analytical results. The easiest way to refine the mesh is to compute the diffusion length scale sqrt(4*alpha_hyd*dt). For a given timestep dt, one needs to make sure that the diffusion front will surpass at least one mesh element.

In the limit when T->0, make sure that for a given set of parameters (effective stresses, dp_star), the fracture will not open with time. 

### Comparisons reference / PyfracX

A mesh with 2000 elements is used. The default adaptive time-stepping is used. 

We compare rupture front and slip profile with semi-analytical solution. 
Semi-analytical solution is coded in the file Plane_TwoD_frictional_ruptures.py. 

Note here figures showing the slip distribution with respect to the dimensionless coordinate along the fault