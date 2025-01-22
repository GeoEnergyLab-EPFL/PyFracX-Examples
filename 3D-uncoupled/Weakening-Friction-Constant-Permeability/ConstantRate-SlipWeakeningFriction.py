#
# This file is part of PyFracX-Examples
# Fluid injection at constant rate into a slip-weakening frictional fault in 3D 
# Reference results from Sáez & Lecampion (2023) (Poisson's ratio = 0)


# # Set the dimensionless parameters of the simulation defined in Section 4 in Sáez & Lecampion (2023)
#%%
S_factor = 0.6 #tau0/(f_p*(sigma0-p0)), #Pre-stress ratio or stress criticiality
overpressure = 0.05 #dp_star/sigma0, Overpressure ratio
f = 0.7  #Residual to Peak friction ratio
value_critical = 0.1*(1-S_factor)  #Critical overpressure ratio

print("S factor = ", S_factor)
print("peakfriction/residualfriction", f)
print("overpressure = ", overpressure)
print("Critical overpressure value", value_critical)

if overpressure > value_critical:
    print("Slip activates")
    if S_factor < f:
        print("Ultimately stable regime, Tr > 0")
    else:
        print("Unstable regime, Tr < 0")
else:
    print("No slip activation")

# %%+
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path
import scipy

import time
import meshio
from scipy import linalg
from scipy.special import expi
import matplotlib
import matplotlib.pyplot as plt
import os
import sys

#%% mesh a circle with gmsh with a refinement in the center.
from mesh.usmesh import usmesh
import pygmsh
center_res = 0.01 #mesh size at the center
out_res = 10 #mesh size at the boundary
Lx_ext = 142 #domain size (radius)

#%%
#HERE 'wrong' ORIENTATION -> left hand rule .... 
with pygmsh.geo.Geometry() as geom:
    center_point = geom.add_point([0, 0,0], center_res)
    top_point = geom.add_point([0, Lx_ext,0], out_res )
    right_point = geom.add_point([Lx_ext, 0,0], out_res )
    bottom_point = geom.add_point([0.,-Lx_ext,0.],out_res)
    left_point =  geom.add_point([-Lx_ext,0.,0.],out_res)

    l1 = geom.add_line(center_point, top_point)
    arc1 = geom.add_circle_arc(top_point, center_point, right_point)
    l2 = geom.add_line(right_point, center_point)
    loop1 = geom.add_curve_loop([l1,arc1,l2])
    
    l3 = geom.add_line(center_point,right_point)
    arc2 = geom.add_circle_arc(right_point, center_point,bottom_point)
    l4 = geom.add_line(bottom_point,center_point)
    loop2 = geom.add_curve_loop([l3,arc2,l4])
    
    l5 = geom.add_line(center_point,bottom_point)
    arc3 = geom.add_circle_arc(bottom_point,center_point,left_point) 
    l6 = geom.add_line(left_point,center_point)   
    loop3 = geom.add_curve_loop([l5,arc3,l6])
    
    l7=geom.add_line(center_point,left_point)
    arc4 = geom.add_circle_arc(left_point,center_point,top_point)
    l8 = geom.add_line(top_point,center_point)
    loop4 = geom.add_curve_loop([l7,arc4,l8])
    
    geom.add_plane_surface(loop1)
    geom.add_plane_surface(loop2)
    geom.add_plane_surface(loop3)
    geom.add_plane_surface(loop4)
    
    geom.synchronize()
    g_mesh = geom.generate_mesh(order=1,algorithm=2)
    
mesh=usmesh.fromMeshio(g_mesh,1)
#%%
# swap connectivity because 
swap_c = mesh.conn.copy()
swap_c[:,0]=mesh.conn[:,2]
swap_c[:,2]=mesh.conn[:,0]
mesh.conn=swap_c
Nelts = mesh.nelts
print("Number of elements", Nelts)

## plotting the unstructured mesh
triang=matplotlib.tri.Triangulation(mesh.coor[:,0], mesh.coor[:,1], triangles=mesh.conn, mask=None)
fig1, ax1 = plt.subplots()
ax1.set_aspect('equal')
ax1.triplot(triang, 'b-', lw=1)
ax1.plot(0.,0.,'ko')
plt.xlabel('x')
plt.ylabel('y')
#plt.title('Triangular mesh')
plt.show()

Nnodes=mesh.nnodes
coor=np.asarray(mesh.coor)
conn=np.asarray(mesh.conn)
colPts= [(coor[conn[i][0]]+coor[conn[i][1]]+coor[conn[i][2]])/3. for i in range(Nelts)] # put it in usmesh
r=np.array([scipy.linalg.norm(coor[i]) for i in range(Nnodes)])
r_col=np.array([scipy.linalg.norm(colPts[i]) for i in range(Nelts)])

#%%
# Material Parameters of the simulation

alpha = 0.01  #  [m^2/s], rock hyd diffusivity
mu = 8.9e-4  # [Pa s], viscosity
wh = (12 * 3e-12) ** (1/3)  # [m], hydraulic aperture
hyd_cond = wh**2 / (12 * mu)  # (wh^2 / (12 mu)) [m^2/Pa s], intrinsic perm over viscosity
storage = hyd_cond / alpha  # [1/Pa], storage coefficient

# elastic properties
G = 30e9
nu = 0.0
YoungM = 2 * G * (1 + nu) 
beta_spring = 100 # kn, ks are beta_spring elastic stiffness

# Frictional properties
f_p = 0.6 #peak friction coefficient
f_r = f*f_p #residual friction coefficient
f_dil = 0 # dilatancy coefficient

#Fault Parameters
cohesion = 0.0 #cohesion
T_expected = 0.01 
sigma0 = 120e6  #effective normal stress
p0 = 0 # background pore pressure
dp_star = overpressure * (sigma0 - p0)  #Characteristic pressure drop [Pa]
tau0 = S_factor * f_p * (sigma0 - p0)  # [Pa], shear stress
tau_yz = 0
tau_xz = np.sqrt(tau0**2 - tau_yz**2)
Qinj = dp_star* (4 * hyd_cond * np.pi * wh) #wellbore flow rate: m^3/s

T_p = (f_p * (sigma0 - p0) - tau0) / (f_p * dp_star)
T_r = (f_r * (sigma0 - p0) - tau0) / (f_r * dp_star)
print("tau_xz = ", tau_xz)
print("tau_yz = ", tau_yz)
print("tau0 = ", tau0)
print("T_p = ", T_p)
print("T_r = ", T_r)

#%%
#Calculating the critical slip distance

# Fixing the rupture length scale from the mesh size
Rw = 50 * center_res

# Computing the critical slip distance from the rupture length scale defined in Section 4.1 in Sáez & Lecampion (2023)
d_c = (Rw*(f_p-f_r)*(sigma0-p0))/G

alpha = 0.01  #  [m^2/s], rock hyd diffusivity

# Set the time of simulation to reproduce plots in Figure 5 of Sáez & Lecampion (2023)
# Get x_ord (sqrt(4 * alpha * t)/ Rw) from the plot you want to reproduce in Figure 5 of Sáez & Lecampion (2023)
x_ord = 100
tend= (x_ord*Rw)**2/(4* alpha)
dt_ini= tend/100 #(10 * center_res)**2 / alpha
maxSteps=10000000

print("Maximum rupture length", Lx_ext)
print("Element size in center", center_res)
print("Number of elements", Nelts)
print("Rupture Length Scale", Rw)
print("total time", tend)
print("initial time step", dt_ini)
print("Critical slip distance", d_c)

#%%

# analytical solution for pressure at collocation points
pressure = lambda r, t: (p0 -Qinj/(4*hyd_cond*np.pi*wh) * expi(-r**2/(4.*alpha*t)) )

p_col=pressure(r_col, 5000.)
fig, ax = plt.subplots()
ax.plot(r_col,p_col,'.r')
plt.show()

# Elasticity matrix
from mechanics.H_Elasticity import Elasticity

kernel="3DT0-H"
elas_properties=np.array([YoungM, nu])
elastic_m=Elasticity(kernel, elas_properties,max_leaf_size=32,eta=4.0,eps_aca=1.e-5,n_openMP_threads=16)
# hmat creation
h1=elastic_m.constructHmatrix(mesh)


# in_situ_tractions =np.full((Nelts,3),[-tau_xz, -tau_yz, -sigma0]) # positive stress in traction !
# e1n=[]; e2n=[]; e3n=[]; C=[]; local_in_situ_tractions=np.array([])

# for i in range(Nelts):
#     e1n.append( (coor[conn[i][1]]-coor[conn[i][0]])/np.linalg.norm((coor[conn[i][1]]-coor[conn[i][0]])) )
#     e3n.append( np.cross(e1n[i], coor[conn[i,2]]-coor[conn[i,0]]) /np.linalg.norm(np.cross(e1n[i], coor[conn[i,2]]-coor[conn[i,0]])) )
#     e2n.append( np.cross(e3n[i], e1n[i])/np.linalg.norm(np.cross(e3n[i], e1n[i])) )
#     C.append(np.array([e1n[i],e2n[i],e3n[i]]))
#     local_in_situ_tractions=np.append(local_in_situ_tractions, C[i] @ in_situ_tractions[i])

#%%
# insitu tractions
insitu_tractions_global = np.full((mesh.nelts, 3), [-tau_xz, -tau_yz, -sigma0])  # positive stress in traction ! tension positive convention
# flattened array
insitu_tractions_local = h1.convert_to_local(insitu_tractions_global.flatten())
#%%
from MaterialProperties import PropertyMap
from mechanics.friction3D import *
from mechanics.evolutionLaws import *
from mechanics.mech_utils import *

# Preparing  properties for simulation
friction_p=PropertyMap(np.zeros(mesh.nelts, dtype=int),np.array([f_p]))           # peak friction coefficient
friction_r=PropertyMap(np.zeros(mesh.nelts, dtype=int),np.array([f_r]))
dilatant_c=PropertyMap(np.zeros(mesh.nelts, dtype=int),np.array([f_dil]))# dilatancy coefficient
cohesion_p=PropertyMap(np.zeros(Nelts,dtype=int),np.array([cohesion]))
slip_dc = PropertyMap(np.zeros(Nelts,dtype=int),np.array([d_c]))

k_sn=PropertyMap(
    np.zeros(mesh.nelts, dtype=int),
    np.array([[beta_spring*G,beta_spring*YoungM]])) # springs shear, normal

mat_properties={
    "Peak friction": friction_p,
    "Residual friction": friction_r,
    "Peak dilatancy coefficient": dilatant_c,
    "Critical slip distance":slip_dc,
    "Peak cohesion": cohesion_p,
    "Spring Cij":k_sn,
    "Elastic parameters":{"Young":YoungM,"Poisson":nu}}# instantiating the constant friction model

# instantiating the slip-weakening friction model

##### MANUAL INPUT OF THE EVOLUTION LAWS #####
#### "linearEvolution" : Linear slip-weakening evolution law ######
#### "expEvolution" : Exponential slip-weakening evolution law ######

frictionModel = FrictionVar3D(mat_properties, Nelts, linearEvolution)

# quasi-dynamics term (put here to a small value)
c_s = np.sqrt(G/1.)
c_p = c_s * np.sqrt(2*(1-nu)/(1.-2*nu))
eta_s = 1.e-4*0.5 * G / c_s
eta_p = 1.e-4*0.5 * G / c_p
qd=QuasiDynamicsOperator(Nelts,3,eta_s,eta_p)

# creating the mechanical model: hmat, preconditioner, number of collocation points, constitutive model
mech_model=MechanicalModel(h1, mesh.nelts, frictionModel, precType="Jacobi", QDoperator=qd)

#%%
from hm.HMFsolver import HMFSolution
# Initial solution
sol0=HMFSolution(time=0., 
                 effective_tractions=insitu_tractions_local.flatten(), 
                 pressure=np.zeros(mesh.nelts), 
                 DDs=0. * insitu_tractions_local.flatten(), 
                 DDs_plastic=0. * insitu_tractions_local.flatten(),
                 pressure_rate=np.zeros(mesh.nnodes), 
                 DDs_rate=0. * insitu_tractions_local.flatten(),
                 Internal_variables=np.zeros(2*mesh.nelts),
                 res_mech=0.*insitu_tractions_local.flatten())

# # simulation stepper options
# stepper_options={"Time integration":{"Method":"implicit","LTE goal":0.01,"Acceptance res tolerance":1.e5,"Number of attempts":4,"Step decrease factor":2},
#     "Newton":{"Norm order":2,"Max iterations":20,"Residuals rtol":1e-3,"Residuals tol":np.inf,"Dx rtol":1e-3,"Dx tol":np.inf,"Line search":True,"Initial guess factor":1.},
#                  "Tangent solver":{"GMRES":{"Max iterations":200,"Tolerance":1.e-7,"Restart":150}}}
#
#%%
# stepper options
from utils.options_utils import TimeIntegration_options,NonLinear_step_options,NonLinearSolve_options,IterativeLinearSolve_options

res_atol = 1.e-3*max(np.linalg.norm(insitu_tractions_local.flatten()),1e3)
print("res_atol: %g" %(res_atol))

# newton solve options
newton_solver_options=NonLinearSolve_options(max_iterations=50,
                                             residuals_atol=res_atol,
                                             residuals_rtol=np.inf,
                                             dx_atol=np.inf,
                                             dx_rtol=1e-2,
                                             line_search=True,
                                             line_search_type="cheap",verbose=True)

# options for the jacobian solver
jac_solve_options=IterativeLinearSolve_options(max_iterations=300,
    restart_iterations=250,
    absolute_tolerance=0.,
    relative_tolerance=1.e-5,
    preconditioner_side="Left",
    schur_ilu_fill_factor=1,
    schur_ilu_drop_tol=1e-2,
    mech_rtol=1e-4,
    mech_atol=0.,
    mech_max_iterations=int(mesh.nelts/2))

# combining the 2 as option for the non-linear time-step
step_solve_options=NonLinear_step_options(jacobian_solver_type="GMRES",
                                          jacobian_solver_opts=jac_solve_options,
                                          non_linear_start_factor= 0.0,
                                          non_linear_solver_opts=newton_solver_options)

#%%
################################
## function wrapping the one-way H-M solver for this case
from hm.HMFsolver import hmf_one_way_flow_numerical, HMFSolution,  hmf_one_way_step

Dtinf=np.zeros(3 * Nelts)

def onewayStepWrapper(solN,dt):
    time_s = solN.time + dt
    # compute pressure at collocation point
    dpcol = pressure(r_col, time_s) - solN.pressure
    solTnew = hmf_one_way_step(solN, dt, mech_model, dpcol, Dtinf, step_solve_options)
    solTnew.pressure = solN.pressure + dpcol  # must be change to pressure at nodes !!!
    return solTnew
#%%
# Create the simulation object
from datetime import datetime
# dd/mm/YY H:M:S
Simul_description="3D - ct injection rate - LW friction - one way simulation"
now=datetime.now()
dt_string = now.strftime("%d-%m-%Y-%H:%M:%S")
basename="3D-lwfric-oneway-S-0.6-P-0.05"
basefolder="./" + basename + "-" + dt_string+"/"
# Ensure the parent directory exists before creating the subdirectory
os.makedirs(os.path.dirname(basefolder), exist_ok=True)
model_config={ # in this dict, we store object etc. (but not the hmat that is not storable for now)
    "Mesh":mesh,
    "Elasticity":elastic_m,
    "Friction model": frictionModel,
    "Material properties": mat_properties,
}

model_parameters={
    "Elasticity":{"Young":YoungM,"Nu":nu},
    "Injection":{"Injection rate": Qinj},
    "Flow": {"Hydraulic diffusivity": alpha, "Hydraulic conductivity":hyd_cond},
    "Hydraulic aperture": wh,
    "Initial stress":[sigma0, tau_xz,tau_yz,],
    "Peak Friction":f_p,"Dilatancy peak":f_dil,"Dc":d_c,
    "T peak": T_p, "T residual": T_r, "Pre-stress ratio": S_factor, "Overpressure ratio": overpressure,
    "Peak to residual friction": f, "Rupture length scale": Rw,
}
#%%
from utils.App import TimeIntegrationApp
from utils.options_utils import TimeIntegration_options


# options of the time inegration !  note that we also pass the step_solve_options
ts_options=TimeIntegration_options(max_attempts=10,
                                   dt_reduction_factor=1.5,
                                   max_dt_increase_factor=1.03,
                                   lte_goal=0.001,
                                   acceptance_a_tol= res_atol,
                                   minimum_dt= 0.1,
                                   maximum_dt = 2500,
                                   stepper_opts=step_solve_options)

my_simul=TimeIntegrationApp(basename, model_config,
                            model_parameters, 
                            ts_options,onewayStepWrapper,
                            description=Simul_description, 
                            basefolder=basefolder)

my_simul.setAdditionalStoppingCriteria(lambda sol : sol.Nyielded == mesh.nelts)

my_simul.setupSimulation(sol0,tend,dt=dt_ini,maxSteps=maxSteps,saveEveryNSteps = 5,log_level="INFO")

my_simul.saveParametersToJSon()
my_simul.saveConfigToBinary()
mesh.saveToJson(basefolder+"Mesh.json")
# import dill
# with open(my_simul.basefolder+"Configuration","br") as f:
#      aux=dill.load(f)
#%%
# now we are ready to run the simulation
zt= time.process_time()
res,status_ts=my_simul.run()
elapsed=time.process_time()-zt
print("End of simulation in ",elapsed)
#%%
#-----------------------
# post-processing
tts=np.array([res[i].time for i in range(len(res))])
nyi=np.array([res[i].Nyielded for i in range(len(res))])

rmax=0.*tts

for i in range(len(res)):
    aux_k = np.where(res[i].yieldedElts)[0]
    
    if len(aux_k) > 0:
        rmax[i] = r_col[aux_k].max()
    else:
        rmax[i] = 0.0  # or some other appropriate value

#%%
solN = res[-1]
fig, ax = plt.subplots()
ax.plot(r_col, solN.pressure,'.')
plt.xlabel('r')
plt.ylabel('Fluid Pressure (Pa)')
plt.show()
# %%# %% slip 

global_dds = h1.convert_to_global(solN.DDs)
 
fig1, ax1 = plt.subplots()
tri=ax1.tripcolor(triang, global_dds[0::3],cmap = plt.cm.rainbow, 
                    alpha = 0.5)
ax1.axis('equal')
plt.colorbar(tri) 
plt.title('Slip along x')
plt.show()

fig1, ax1 = plt.subplots()
tri=ax1.tripcolor(triang, global_dds[1::3],cmap = plt.cm.rainbow, 
                    alpha = 0.5)
ax1.axis('equal')
plt.colorbar(tri) 
plt.title('Slip along y')
plt.show()

fig1, ax1 = plt.subplots()
tri=ax1.tripcolor(triang, global_dds[2::3],cmap = plt.cm.rainbow, 
                    alpha = 0.5)
ax1.axis('equal')
plt.colorbar(tri) 
plt.title('Opening')
plt.show()


# %%
plastic_dds = h1.convert_to_global(res[-1].DDs_plastic)

fig, ax = plt.subplots()
ax.plot(r_col,-plastic_dds[0::3] ,'.r', label='plastic')
ax.plot(r_col,-global_dds[0::3]+plastic_dds[0::3] ,'.b', label='elastic')
plt.xlabel('r')
plt.ylabel('Slip')
plt.legend()
#plt.xlim(0, 15)
plt.show()


fig, ax = plt.subplots()
ax.plot(r_col,plastic_dds[2::3] ,'.r', label='plastic')
ax.plot(r_col,global_dds[2::3]-plastic_dds[2::3] ,'.b', label='elastic')
plt.xlabel('r')
plt.ylabel('opening')
plt.legend()
plt.show()

# %% traction

T_eff_n = h1.convert_to_global(solN.effective_tractions)

fig1, ax1 = plt.subplots()
tri=ax1.tripcolor(triang, T_eff_n[2::3],cmap = plt.cm.rainbow, 
                    alpha = 0.5)
ax1.axis('equal')
plt.colorbar(tri) 
plt.title('Normal effective traction')
plt.show()

fig1, ax1 = plt.subplots()
tri=ax1.tripcolor(triang, T_eff_n[0::3],cmap = plt.cm.rainbow, 
                    alpha = 0.5)
ax1.axis('equal')
plt.colorbar(tri) 
plt.title('Shear effective traction')
plt.show()


# %%
eff_end=np.reshape(h1.convert_to_global(res[-1].effective_tractions),(-1,3))

myF = eff_end[:,0]+f_p*eff_end[:,2]

fig1, ax1 = plt.subplots()
tri=ax1.tripcolor(triang, myF,cmap = plt.cm.rainbow, 
                    alpha = 0.5)
ax1.axis('equal')
plt.colorbar(tri) 
plt.title('Yield function')
plt.show()
