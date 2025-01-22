#%% This file is part of PyFracX.
#
# Created by Brice Lecampion on 28.04.21.
# Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2021.  All rights reserved.
# See the LICENSE.TXT file for more details.
#
#
# Unstructured surface mesh - ct rate injection
# ct friction case

# %%+
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import splu
from scipy.sparse import csr_matrix
import matplotlib
import matplotlib.pyplot as plt

#%%
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


# analytical solution for pressure 
import scipy.special as sc
def pres(r,t,c=1.): # divided by dp_c
    return (-1./(4*np.pi))*sc.expi(-r*r/(4*c*t))


#%% mesh a circle with gmsh with a refinement in the center.
import pygmsh

center_res = 0.05
out_res = 0.1
Lx_ext = 1.

#   HERE 'wrong' ORIENTATION -> left hand rule .... 
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

# swap connectivity because 
swap_c = mesh.conn.copy()
swap_c[:,0]=mesh.conn[:,2]
swap_c[:,2]=mesh.conn[:,0]
mesh.conn=swap_c
#
Nelts = mesh.nelts
#
## plotting the unstructured mesh
triang=matplotlib.tri.Triangulation(mesh.coor[:,0], mesh.coor[:,1], triangles=mesh.conn, mask=None)
fig1, ax1 = plt.subplots()
ax1.set_aspect('equal')
ax1.triplot(triang, 'b-', lw=1)
ax1.plot(0.,0.,'ko')
plt.show()


#%% 
# injection rate [m3/s]
Qinj = 1.8/60
# hydraulic properties - cubic law

fluid_visc =8.9e-4 # fluid viscosity
wh_o = 3.3e-4 # initial hydraulic width
alpha_h = 0.01
hyd_cond = (wh_o**2 /(12.*fluid_visc))
S_e = hyd_cond / alpha_h   # storage [1/Pa] 
dp_star = Qinj/((4.*np.pi)*(hyd_cond*wh_o))

# elastic properties
G = 30e9
nu = 0
YoungM = 2*G*(1+nu) 

# spring factor 
beta_spring = 100 # kn, ks are beta_spring elastic stiffness

# ct friction
f_p = 0.6
f_dil = 0# zero dilatancy

#---- in-situ conditions
#Fault Parameters
T_expected = 0.01 
sigmap_o = 80e6  # effective normal stress
p0 = 0 # background pore pressure
tau_xz = f_p * (sigmap_o) - f_p * T_expected * dp_star
tau_yz=0
T = (f_p*(sigmap_o)-tau_xz)/ (f_p*dp_star) 
print("T = ", T)
print("tau_xz = ", tau_xz)
print("tau_yz = ", tau_yz)

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
k_sn=PropertyMap(np.zeros(mesh.nelts, dtype=int),np.array([[beta_spring*YoungM/(2*(1+nu)),beta_spring*YoungM]])) # springs shear, normal

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

#%% Flow model

stor_c=PropertyMap(np.zeros(mesh.nelts,dtype=int),np.array([S_e]))
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
                                             dx_rtol=1e-3,
                                             line_search=True,
                                             line_search_type="cheap",
                                             verbose=True)

# options for the jacobian solver
jac_solve_options = IterativeLinearSolve_options(
    max_iterations=300,
    restart_iterations=150,
    absolute_tolerance=0.,
    relative_tolerance=1e-6,
    preconditioner_side="Left",
    schur_ilu_fill_factor=1,
    schur_ilu_drop_tol=1e-4,
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
tend=0.17*10
maxSteps= 1000

# options of the time inegration !  note that we also pass the step_solve_options for consistency
ts_options=TimeIntegration_options(max_attempts=6,dt_reduction_factor=1.5,max_dt_increase_factor=1.3,lte_goal=0.001,
                                   acceptance_a_tol=res_atol,minimum_dt=dt_ini/100.0,maximum_dt=dt_ini*100,
                                   stepper_opts=step_solve_options)

#%%
from datetime import datetime
# dd/mm/YY H:M:S
import os,sys 
home = os.environ["HOME"]

Simul_description="3D - ct injection rate - ct friction - cubic law model - coupled simulation"
now=datetime.now()
dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
basename="3D-ctFriction-CubicLaw-benchmark"
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
    "Flow": {"fracture diffusivity": alpha_h, "Hydraulic conductivity":hyd_cond
             ,"Initial aperture":wh_o,"Fluid viscosity":fluid_visc,"Storage ":S_e},
    "Initial stress":[tau_xz,tau_yz,sigmap_o],
    "Friction coefficient":f_p, "Dilatancy coefficient":f_dil,
    "T parameter":T
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














#%%
alpha = alpha_h
from scipy.special import exp1
tts = np.array([res[i].time for i in range(len(res))])
# analytical solution for pressure at collocation points, Eq. 4 in Alexi JMPS
pressure = lambda r, t: (p0 + dp_star * exp1((r**2) / (4.0 * alpha * t)))
p_col = pressure(rcoor, tts[-1]) 
plt.figure()
plt.plot(rcoor, solN.pressure,'.')
plt.plot(rcoor, p_col,'r.')
plt.xlabel("r (m)")
plt.ylabel("p(r) / po")
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

#%%
colPt = h1.getCollocationPoints()

rcol = (colPt[:,0]**2+colPt[:,1]**2)**(0.5)

plastic_dds = h1.convert_to_global(res[-1].DDs_plastic)

fig, ax = plt.subplots()
ax.plot(rcol,-plastic_dds[0::3] ,'.r')
ax.plot(rcol,-global_dds[0::3]+plastic_dds[0::3] ,'.b')
plt.xlabel('r')
plt.ylabel('Slip')
plt.show()


fig, ax = plt.subplots()
ax.plot(rcol,plastic_dds[2::3] ,'.r')
ax.plot(rcol,global_dds[2::3]-plastic_dds[2::3] ,'.b')
plt.xlabel('r')
plt.ylabel('opening')
plt.show()


# %%Verification for slip profile
import mpmath

lambda_crticallstress = lambda T: 1.0 / np.sqrt(2.0 * T)
# Marginally pressurized case, Eq. 23
lambda_marginallypressurized = lambda T: 0.5 * np.exp(
    (2.0 - float(mpmath.euler) - T) * 0.5
)

def lambda_approx(T):
    lam = 0
    if T > 2:
        lam = lambda_marginallypressurized(T)
    elif T < 0.3:
        lam = lambda_crticallstress(T)
    else:
        lam = (lambda_crticallstress(T) + lambda_marginallypressurized(T)) * 0.5

    return lam

def lambda_analytical(T):
    # Eq 21
    eq = (
        lambda lam2: 2
        - mpmath.euler
        + (2 / 3) * lam2 * mpmath.hyp2f2(1, 1, 2, 2.5, -lam2)
        - mpmath.log(4 * lam2)
        - T
    )
    return float(mpmath.sqrt(mpmath.findroot(eq, lambda_approx(T) ** 2).real))

tts = np.array([res[i].time for i in range(len(res))])
nyi = np.array([res[i].Nyielded for i in range(len(res))])

# find position of last yielded element (slip becomes zero)
rmax = 0.0 * tts
for i in range(len(res)):
    aux_k = np.where(res[i].yieldedElts)[0]
    rmax[i] = rcol[aux_k].max()
    
# %% Check for rupture radius
lam = lambda_analytical(T)
print("lambda = ", lam)
print("Predicted Rmax = ", np.sqrt(4.0 * alpha * tend) * lam)
t_ = np.linspace(tts[0], tts.max())
R = lam * np.sqrt(4.0 * alpha * t_)
plt.figure()
plt.plot(t_, R, "r")
plt.plot(tts, rmax, ".")
#plt.semilogx()
#plt.semilogy()
plt.xlabel("Time (s)")
plt.ylabel("Rupture radius (m)")
plt.legend(["Analytical solution", "Numerics"])
plt.show()

# %%
plt.figure()
plt.plot(tts, lam * np.ones(len(tts)), "-r")
plt.plot(tts, rmax / np.sqrt(4 * alpha * tts), "ko")
plt.semilogx()
plt.xlabel("Time (s)")
plt.ylabel(r"$\lambda$ = Rupture radius / $\sqrt{4 \alpha t}$")
plt.gca().legend(["Analytical solution", "Numerics"])
plt.title(r"T = %.3f, $\lambda = %.2f$" % (T, lam))
# plt.ylim([2, 7.5])
print(
    "Relative error from analytical prediction in Percentage : ",
    (np.abs(lam - rmax[-1] / np.sqrt(4 * alpha * tts[-1])) / lam) * 100.0,)
# %% Analytical solution for slip profile Eq.~25, 26
# slip profile
fac = f_p * dp_star / G
# Eq 25, marginally pressurized regime
slip_marg = (
    lambda r: fac
    * (8 / np.pi)
    * (np.sqrt(1 - r**2) - np.abs(r) * np.arccos(np.abs(r)))
)
# Eq 26, critically stressed regime
slip_crit = (
    lambda r: fac
    * ((2 * np.sqrt(2 * T)) / np.pi)
    * (np.arccos(np.abs(r)) / np.abs(r) - np.sqrt(1 - r**2))
)
print(fac)
# %% Fig 3
# CS outer asymptotic solution, doesnt match for T = 0.1
lt = np.sqrt(4 * alpha * tts[-1])
rt = rmax[-1]
fac_lt = fac * lt
fac_rt = fac * rt
plt.figure()
if T <= 0.8:
    plt.plot(
        rcol / rt,
        slip_crit(rcol / rt) * lt / fac_lt,
        ".r",
        label="Reference solution",
    )
elif T >= 2.0:
    plt.plot(
        rcol / rt,
        slip_marg(rcol / rt) * rt / fac_lt,
        ".r",
        label="Analytical solution: MP",
    )
plt.plot(
    rcol / rt,
    -global_dds[0::3] / fac_lt,
    "*k",
    ms=1.2,
    label="Numerics",
)
plt.xlabel("r/Rmax")
plt.ylabel("normalized slip")
plt.legend()
#plt.xlim([0.0, 1.5])
plt.ylim([0.0, 1.5 * np.max(np.abs(res[-1].DDs.reshape(-1, 2)[:, 0]) / fac_lt)])

# %% Check for slip profile Eq.~25, 26
lam = lambda_analytical(T)
Lt = np.sqrt(4*alpha*tts[-1])
Rt = lam*Lt
# slip profile
fac = f_p * dp_star / G
# Eq 25, marginally pressurized regime
slip_marg = (
    lambda r: fac
    * (8 / np.pi)
    * (np.sqrt(1 - r**2) - np.abs(r) * np.arccos(np.abs(r)))
)
# Eq 26, critically stressed regime
slip_crit = (
    lambda r: fac
    * ((2 * np.sqrt(2 * T)) / np.pi)
    * (np.arccos(np.abs(r)) / np.abs(r) - np.sqrt(1 - r**2))
)

plt.figure()
if T <= 0.8:
    plt.plot(
        rcol,
        slip_crit(rcol)*Rt,
        ".r",
        label="Reference solution",
    )
elif T >= 2.0:
    plt.plot(
        rcol,
        slip_marg(rcol) * Lt,
        ".r",
        label="Analytical solution: MP",
    )
    
plt.figure(0)
plt.plot(
    rcol,global_dds[2::3],
    "*k",
    ms=1.2,
    label="Numerics",
)
plt.xlabel("r")
plt.ylabel("Slip")
plt.legend()

# %%
