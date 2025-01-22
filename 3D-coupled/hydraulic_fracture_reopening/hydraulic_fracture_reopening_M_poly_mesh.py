#%% This file is part of PyFracX.
#
# Created by Brice Lecampion on 08.01.25.
# Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2025.  All rights reserved.
# See the LICENSE.TXT file for more details.
#
#
# ct friction case

# %% General Imports 
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

import gmsh

# analytical solution for pressure 
import scipy.special as sc
def pres(r,t,c=1.): # divided by dp_c
    return (-1./(4*np.pi))*sc.expi(-r*r/(4*c*t))

#  Imports from PyFracX
from mesh.usmesh import usmesh
from mesh.mesh_utils import *
from MaterialProperties import PropertyMap
#from fe.assemble import assemble,assembleLoadFun
from flow.FlowConstitutiveLaws import *
from flow.flow_utils import FlowModelFractureSurfaces
from loads.Injection import *
from mechanics.mech_utils import *
from mechanics.H_Elasticity import *
from mechanics.friction3D import FrictionCt3D
from hm.HMFsolver import HMFSolution,hmf_coupled_step
from ts.TimeStepper import *
from utils.App import TimeIntegrationApp


#%%   Mesh a polygon centered on 0,0
center_res = 0.005
out_res = 0.05
Lx_ext = 1.

p=6
x=[([Lx_ext*np.cos(2*np.pi*i/p),Lx_ext*np.sin(2*np.pi*i/p),0. ]) for i in range(p)]

gmsh.initialize()
for i in range(p):
    gmsh.model.geo.addPoint(x[i][0], x[i][1], x[i][2], out_res, i+1)

for i in range(p-1):
    gmsh.model.geo.addLine(i+1, i+2, i+1)
    
gmsh.model.geo.addLine(p,1,p)    

ll = [ i+1 for i in range(p)]
gmsh.model.geo.addCurveLoop(ll,1)
gmsh.model.geo.addPlaneSurface([1], 1)

# We define a new point for the origin to enforce the mesh
gmsh.model.geo.addPoint(0.0, 0.0, 0., center_res, p+1)

gmsh.model.geo.synchronize()
gmsh.model.mesh.embed(0, [p+1], 2, 1)

gmsh.model.mesh.generate(5)

## post-processing the gmsh generated triangulation
dim=-1
tag=-1
nodeTags,coords,parametricCoord=gmsh.model.mesh.getNodes(dim,tag)
coords=coords.reshape((-1,3))

defined_elt_type=gmsh.model.mesh.getElementTypes()

eletype = 2 # triangle
tag=-1
eleTags,nodeTags = gmsh.model.mesh.getElementsByType(eletype,tag)
nodeTags=nodeTags.reshape((-1,3))-1
#gmsh.fltk.run()
mesh=usmesh(coor=coords,conn=nodeTags,dimension=3)
gmsh.finalize()    

Nelts = mesh.nelts
#
## plotting the unstructured mesh
triang=matplotlib.tri.Triangulation(mesh.coor[:,0], mesh.coor[:,1], triangles=mesh.conn, mask=None)
fig1, ax1 = plt.subplots()
ax1.set_aspect('equal')
ax1.triplot(triang, 'b-', lw=1)
ax1.plot(0.,0.,'ro')
plt.show()

#%%   Parameters definition

fluid_visc =1e-3 # fluid viscosity
cf = 4.5e-10  # [Pa^-1] compressibility of the fluid
beta_skempton = 0.95 # Skempton coefficient
pcbysigop = 200.   
taubyfsigop = 0.0 # 0.0 means only HF with leaky tip

# elastic properties
G = 20e9
nu = 0.
YoungM = 2*G*(1+nu) 

# ct friction
f_p = 0.6
f_dil = 0# zero dilatancy

# flow
k =1.5e-18  # permeability [m^2]
H = 1  # interface thickness assumed equal to 1. so that k*H is transmissibility
T_h = k*H
wh_o = (12*T_h)**(1./3) # initial hydraulic width

alpha_h = ((wh_o**2)/12)/(fluid_visc*cf) # this is just to compute the initial time-step

#---- in-situ conditions
sigmap_o = 10e6  # effective normal stress
tau_xz = taubyfsigop*sigmap_o
tau_yz=0.
p0 = 1.e6 # background pore pressure

# injection rate
Qinj = pcbysigop * (T_h* sigmap_o) / (fluid_visc)
dp_c = Qinj/((T_h/fluid_visc))
# springs
ks = 1e3 * G  # shear spring [Pa/m]
kn =  (1 - beta_skempton) / (beta_skempton * cf  * wh_o) # normal spring , 5e2 * YoungM  #

### 
Dbar = (1-beta_skempton)*(1./(cf*sigmap_o))*(pcbysigop**(-2./3)) 

t_op = (YoungM**2)*(12*fluid_visc)/(sigmap_o**3)
print(["Dbar ", Dbar])
print(["t_op ", t_op])

#%%  Mechanical model & initial tractions
# Elasticity model
from mechanics import H_Elasticity

kernel="3DT0-H"
elas_properties=np.array([YoungM, nu])
elastic_m=Elasticity(kernel, elas_properties,max_leaf_size=32,eta=3,eps_aca=1.e-5,n_openMP_threads=8)
# hmat creation
h1=elastic_m.constructHmatrix(mesh)

# Preparing  properties for simulation
friction_c=PropertyMap(np.zeros(mesh.nelts, dtype=int),np.array([f_p]))           # friction coefficient
dilatant_c=PropertyMap(np.zeros(mesh.nelts, dtype=int),np.array([f_dil]))           # dilatancy coefficient
k_sn=PropertyMap(np.zeros(mesh.nelts, dtype=int),np.array([[ks,kn]])) # springs shear, normal

mech_properties={"Friction coefficient":friction_c,
                "Dilatancy coefficient":dilatant_c,"Spring Cij":k_sn,"Elastic parameters":{"Young":YoungM,"Poisson":nu}}
# instantiating the constant friction model
frictionModel=FrictionCt3D(mech_properties, mesh.nelts)

# creating the mechanical model: hmat, preconditioner, number of collocation points, constitutive model
mech_model=MechanicalModel(h1, mesh.nelts, frictionModel, precType="Jacobi")

# In-situ traction and pore pressure field
# insitu tractions
insitu_tractions_global = np.full((mesh.nelts, 3), [-tau_xz, 0.0, -sigmap_o])  # positive stress in traction ! tension positive convention
# flattened array
insitu_tractions_local = h1.convert_to_local(insitu_tractions_global.flatten())
# initial pore pressure array at nodes
pp_0  =np.zeros(mesh.nnodes,dtype=float)+p0

## Flow model

stor_c=PropertyMap(np.zeros(mesh.nelts,dtype=int),np.array([cf]))
aperture=PropertyMap(np.zeros(mesh.nelts,dtype=int),np.array([wh_o]))  #aperture
flow_properties={"Initial hydraulic width":aperture,
                 "Compressibility":stor_c,"Fluid viscosity":fluid_visc}
cubicModel=CubicLawNewtonian(flow_properties,Nelts)   # instantiate the permeability model
#mechanics

the_inj=Injection(np.array([0.,0.,0.]),np.array([[0.,Qinj]]),"Rate")

flow_model=FlowModelFractureSurfaces(mesh, cubicModel, the_inj,scalingR=1,scalingX=1)


# %% initiial solution

sol0=HMFSolution(time=0.,effective_tractions= insitu_tractions_local.flatten(), 
                pressure=0.*np.zeros(mesh.nnodes, dtype=float),
                DDs=0.*insitu_tractions_local.flatten(),
                DDs_plastic=0.*insitu_tractions_local.flatten(),
                Internal_variables=np.zeros(2*mesh.nelts),
                pressure_rate=0.*np.zeros(mesh.nnodes, dtype=float),
                DDs_rate=0.*insitu_tractions_local.flatten(),
                res_mech=0.*insitu_tractions_local.flatten(),
                res_flow=0.*np.zeros(mesh.nnodes, dtype=float))


#%% stepper options
from utils.options_utils import TimeIntegration_options,NonLinear_step_options,NonLinearSolve_options,IterativeLinearSolve_options
# newton solve options
res_atol = 1.e-3*max(np.linalg.norm(insitu_tractions_local.flatten()),1e3)
print("res_atol: %g" %(res_atol))

newton_solver_options=NonLinearSolve_options(max_iterations=20,
                                             residuals_atol=res_atol,
                                             residuals_rtol=np.inf,
                                             dx_atol=np.inf,
                                             dx_rtol=1e-1,
                                             line_search=False,
                                             line_search_type="none",
                                             verbose=True)

# options for the jacobian solver
jac_solve_options = IterativeLinearSolve_options(
    max_iterations=300,
    restart_iterations=150,
    absolute_tolerance=0.,
    relative_tolerance=1e-6,
    preconditioner_side="Left",
    schur_ilu_fill_factor=10,
    schur_ilu_drop_tol=1e-5,
    mech_rtol=1e-6,
    mech_atol=0.,
    mech_max_iterations=mesh.nelts
)

# combining the 2 as option for the non-linear time-step
step_solve_options=NonLinear_step_options(jacobian_solver_type="BICGSTAB",
                                        jacobian_solver_opts=jac_solve_options,
                                        non_linear_start_factor=0.0,
                                        non_linear_solver_opts=newton_solver_options)

#  Preparing the time-stepper
def StepWrapper(solF,dt):
    solFNew = hmf_coupled_step(solF, dt, mech_model, flow_model, step_solve_options)
    return solFNew


# - initial time step from pure diffusion 
h_x = center_res # resolution... 

dt_ini=(1.*h_x)**2 / alpha_h  # setting the initial time-step to have something "moving"
tend=10
maxSteps= 100

# options of the time inegration !  note that we also pass the step_solve_options for consistency
ts_options=TimeIntegration_options(max_attempts=6,dt_reduction_factor=1.5,max_dt_increase_factor=1.1,lte_goal=0.01,
                                   acceptance_a_tol=res_atol,minimum_dt=dt_ini/100.0,maximum_dt=dt_ini*100,
                                   stepper_opts=step_solve_options)

#%%
from datetime import datetime
# dd/mm/YY H:M:S
import os,sys 
home = os.environ["HOME"]

Simul_description="3D - ct injection rate - HF M-vertex re-opening - cubic law model - coupled simulation"
now=datetime.now()
dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
basename="3D-HF-ReOpening-benchmark"
basefolder="./res_data/"+basename+"-"+dt_string+"/"

# Ensure the parent directory exists before creating the subdirectory
os.makedirs(os.path.dirname(basefolder), exist_ok=True)

model_config={ # in this dict, we store object etc. (but not the hmat that is not storable for now)
    "Mesh":mesh,
    "Elasticity":elastic_m,
    "Friction model": frictionModel,
    "Material properties": mech_properties | flow_properties,
}
model_parameters={
    "Elasticity":{"Young":YoungM,"Nu":nu},
    "Injection":{"Injection rate": Qinj},
    "Flow": {"fracture diffusivity": alpha_h, "Hydraulic conductivity":T_h/fluid_visc
             ,"Initial aperture":wh_o,"Fluid viscosity":fluid_visc,"Storage ":cf},
    "Initial stress":[tau_xz,tau_yz,sigmap_o],
    "Friction coefficient":f_p, "Dilatancy coefficient":f_dil,
    "Dp_c over sigmap_o":pcbysigop
}
    
my_simul=TimeIntegrationApp(basename, 
                            model_config, 
                            model_parameters, 
                            ts_options,
                            StepWrapper,
                            description=Simul_description, 
                            basefolder=basefolder)
my_simul.setupSimulation(sol0,
                         tend,
                         dt=dt_ini,
                         maxSteps=maxSteps,
                         saveEveryNSteps=5,
                         log_level='INFO',
                         start_step_number=0) #

my_simul.setAdditionalStoppingCriteria(lambda sol : sol.Nyielded==mesh.nelts)

restart=False
if not(restart):
    my_simul.saveParametersToJSon()
    mesh.saveToJson(basefolder+"Mesh.json")

# %% now we are ready to run the simulation
import time
zt= time.process_time()
res,status_ts=my_simul.run()
elapsed=time.process_time()-zt
print("elapsed time",elapsed)


#%% time evolution of inlet pressure
inj_pressure = np.zeros(len(res))
timestamp =  np.zeros(len(res))
i_inj = the_inj.locate_in_mesh(mesh)
for k in range(len(res)):
    inj_pressure[k]=res[k].pressure[i_inj]
    timestamp[k]=res[k].time

fig1, ax1 = plt.subplots()
plt.plot(timestamp,inj_pressure)
plt.xlabel(" time (s)")
plt.ylabel(" Injection pressure (Pa)")
plt.show()

# %%
solN = res[-1]

rcoor= (mesh.coor[:,0]**2+mesh.coor[:,1]**2)**(0.5)

fig1, ax1 = plt.subplots()
tri=ax1.tricontourf(triang, solN.pressure,cmap = plt.cm.rainbow, 
                    alpha = 0.5)
ax1.axis('equal')
plt.colorbar(tri)
plt.title('Fluid pressure')
plt.show()

fig, ax = plt.subplots()
ax.plot(rcoor, solN.pressure,'.')
plt.xlabel(" r (m)")
plt.ylabel(" Pressure (Pa)")
plt.show()


# %% slip 

global_dds = h1.convert_to_global(solN.DDs)
 
fig1, ax1 = plt.subplots()
tri=ax1.tripcolor(triang, global_dds[0::3],cmap = plt.cm.rainbow, 
                    alpha = 0.5)
ax1.axis('equal')
plt.colorbar(tri) 
plt.title('Slip along x (m)')
plt.show()

fig1, ax1 = plt.subplots()
tri=ax1.tripcolor(triang, global_dds[1::3],cmap = plt.cm.rainbow, 
                    alpha = 0.5)
ax1.axis('equal')
plt.colorbar(tri) 
plt.title('Slip along y (m)')
plt.show()

fig1, ax1 = plt.subplots()
tri=ax1.tripcolor(triang, global_dds[2::3],cmap = plt.cm.rainbow, 
                    alpha = 0.5)
ax1.axis('equal')
plt.colorbar(tri) 
plt.title('Opening (m)')
plt.show()

col_pts = np.asarray([np.mean(mesh.coor[mesh.conn[e,:],:],axis=0) for e in range(mesh.nelts)])  #h1.getCollocationPoints()
rcoor_mid = np.sqrt(col_pts[:,0]**2+col_pts[:,1]**2)
fig, ax = plt.subplots()
ax.plot(rcoor_mid,  global_dds[2::3],'.')
plt.xlabel(" r (m)")
plt.ylabel(" Opening (m)")
plt.show()

fig, ax = plt.subplots()
ax.plot(rcoor_mid, global_dds[0::3],'.')
plt.xlabel(" r (m)")
plt.ylabel(" slip (m)")
plt.show()


fig, ax = plt.subplots()
ax.plot(rcoor_mid, (global_dds[2::3]/wh_o)**3.,'.')
plt.xlabel(" r (m)")
plt.ylabel(" Trans increase (-)")
plt.show()

# %% yield function 

fig1, ax1 = plt.subplots()
tri=ax1.tripcolor(triang, solN.yieldF,cmap = plt.cm.rainbow, 
                    alpha = 0.5)
ax1.axis('equal')
plt.colorbar(tri) 
plt.title('Yield function')
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
solN.yieldedElts


aux_k=np.where(solN.yieldedElts)[0]












# %%
