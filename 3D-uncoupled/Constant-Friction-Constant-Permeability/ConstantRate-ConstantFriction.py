#%% This file is part of PyFracX-Examples.
#
# Fluid injection at constant rate into a frictional fault in 3D 
# Constant friction coefficient  
# %%+
#Importing all the libraries
from datetime import datetime
from pathlib import Path
import os,sys 
import numpy as np
import scipy
from scipy import linalg
from scipy.special import expi
from scipy.sparse.linalg import splu
from scipy.sparse import csr_matrix
import matplotlib
import matplotlib.pyplot as plt

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
from hm.HMFsolver import HMFSolution,hmf_coupled_step, hmf_one_way_step
from ts.TimeStepper import *
from utils.App import TimeIntegrationApp

# analytical solution for pressure 
import scipy.special as sc
def pres(r,t,c=1.): # divided by dp_c
    return (-1./(4*np.pi))*sc.expi(-r*r/(4*c*t))


#%% mesh the circular fracture with gmsh with a refinement in the center.
# the mesh is then converted to a usmesh object
import pygmsh

center_res = 0.005
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
coor = mesh.coor
conn = mesh.conn
Nelts = mesh.nelts
Nnodes = mesh.nnodes
colPts= [(coor[conn[i][0]]+coor[conn[i][1]]+coor[conn[i][2]])/3. for i in range(Nelts)] # put it in usmesh
r=np.array([scipy.linalg.norm(coor[i]) for i in range(Nnodes)])
r_col=np.array([scipy.linalg.norm(colPts[i]) for i in range(Nelts)])
#
## plotting the unstructured mesh
triang=matplotlib.tri.Triangulation(mesh.coor[:,0], mesh.coor[:,1], triangles=mesh.conn, mask=None)
fig1, ax1 = plt.subplots()
ax1.set_aspect('equal')
ax1.triplot(triang, 'b-', lw=1)
ax1.plot(0.,0.,'ko')
plt.show()


# cmesh = Mesh(mesh.coor.flatten(), mesh.conn.flatten(), "3DT0")
# elem_number = 0
# normal = cmesh.get_element_normal(elem_number)


#%% 
# Defining fluid injection parameters
# injection rate [m3/s]
Qinj = 1.8/60

# hydraulic properties 
fluid_visc =8.9e-4 # fluid viscosity
wh_o = 3.3e-4 # initial hydraulic width
alpha_h = 0.01
hyd_cond = (wh_o**2 /(12.*fluid_visc))
S_e = hyd_cond / alpha_h   # storage [1/Pa] 
dp_star = Qinj/((4.*np.pi)*(hyd_cond*wh_o))

# elastic properties
G = 30e9
nu = 0.45
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

#%%
# analytical solution for pressure at collocation points
pressure = lambda r, t: (p0 -Qinj/(4*hyd_cond*np.pi*wh_o) * expi(-r**2/(4.*alpha_h*t)) )

t_i = 5000.
p_col=pressure(r_col, 5000.)
fig, ax = plt.subplots()
ax.plot(r_col,p_col,'.r')
ax.set_title(f"Pressure at t={t_i}s")
plt.show()
#%%  Mechanical model & initial tractions
# Elasticity model
from mechanics import H_Elasticity

kernel="3DT0-H"
elas_properties=np.array([YoungM, nu])
elastic_m=Elasticity(kernel, elas_properties,max_leaf_size=32,eta=4.0,eps_aca=1.e-5,n_openMP_threads=16)
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
pp_0_col = np.zeros(mesh.nelts,dtype=float)+p0
# %% initiial solution

sol0=HMFSolution(effective_tractions=insitu_tractions_local.flatten(),
                 pressure=0.*pp_0_col,DDs=0.*insitu_tractions_local.flatten(),
                 DDs_plastic=0.*insitu_tractions_local.flatten(),
                 pressure_rate=0.*pp_0,DDs_rate=0.*insitu_tractions_local.flatten(),
                 Internal_variables=np.zeros(0),
                 res_mech=0.*insitu_tractions_local.flatten(),res_flow=0.*pp_0)


#%% stepper options
from utils.options_utils import TimeIntegration_options,NonLinear_step_options,NonLinearSolve_options,IterativeLinearSolve_options
# newton solve options
res_atol = 1.e-3*max(np.linalg.norm(insitu_tractions_local.flatten()),1e3)
print("res_atol: %g" %(res_atol))

newton_solver_options=NonLinearSolve_options(max_iterations=20,residuals_atol=res_atol,
                                             residuals_rtol=np.inf,
                                             dx_atol=np.inf,
                                             dx_rtol=1e-2,
                                             line_search=True,line_search_type="cheap",verbose=True)

# options for the jacobian solver
jac_solve_options = IterativeLinearSolve_options(
    max_iterations=250,
    restart_iterations=250,
    absolute_tolerance=0.0,
    relative_tolerance=1e-5,
    preconditioner_side="Left",
    schur_ilu_fill_factor=1,
    schur_ilu_drop_tol=1e-2,
    mech_rtol=1e-4,
    mech_atol=0.0,
    mech_max_iterations=int(mesh.nelts/2)
)
# combining the 2 as options for the non-linear time-step
step_solve_options=NonLinear_step_options(jacobian_solver_type="GMRES",
                                          jacobian_solver_opts=jac_solve_options,
                                          non_linear_start_factor=0.0,
                                          non_linear_solver_opts=newton_solver_options)

################################
## function wrapping the one-way H-M solver for this case
Dtinf=np.zeros(3 * Nelts)

def onewayStepWrapper(solN,dt):
    time_s = solN.time + dt
    # compute pressure at collocation point
    dpcol = pressure(r_col, time_s) - solN.pressure
    solTnew = hmf_one_way_step(solN, dt, mech_model, dpcol, Dtinf, step_solve_options)
    solTnew.pressure = solN.pressure + dpcol  # must be change to pressure at nodes !!!
    return solTnew

# - initial time step from pure diffusion 
h_x = center_res # resolution... 

dt_ini=(2*h_x)**2 / alpha_h  # setting the initial time-step to have something "moving"
tend=0.17
maxSteps= 200

# options of the time inegration !  note that we also pass the step_solve_options for consistency
ts_options=TimeIntegration_options(max_attempts=6,dt_reduction_factor=1.5,max_dt_increase_factor=1.3,lte_goal=0.001,
                                   acceptance_a_tol=res_atol,minimum_dt=dt_ini/100.0,maximum_dt=dt_ini*100,
                                   stepper_opts=step_solve_options)

#%%
from datetime import datetime
# dd/mm/YY H:M:S
import os,sys 
home = os.environ["HOME"]

Simul_description="3D - ct injection rate - ct friction - oneway simulation"
now=datetime.now()
dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
basename="3D-ctFriction-ctperm-benchmark"
basefolder="./res_data/"+basename+"-"+dt_string+"/"


# Ensure the parent directory exists before creating the subdirectory
os.makedirs(os.path.dirname(basefolder), exist_ok=True)

model_config={ # in this dict, we store object etc. (but not the hmat that is not storable for now)
    "Mesh":mesh,
    "Elasticity":elastic_m,
    "Friction model": frictionModel,
    "Material properties": mech_properties,
}
model_parameters={
    "Elasticity":{"Young":YoungM,"Nu":nu},
    "Injection":{"Injection rate": Qinj},
    "Flow": {"Hydraulic diffusivity": alpha_h, "Hydraulic conductivity":hyd_cond
             ,"Initial aperture":wh_o,"Fluid viscosity":fluid_visc,"Storage ":S_e},
    "Initial stress":[tau_xz,tau_yz,sigmap_o],
    "Friction coefficient":f_p, "Dilatancy coefficient":f_dil,
    "T parameter":T
}
    
my_simul=TimeIntegrationApp(basename, model_config, model_parameters, ts_options,onewayStepWrapper,description=Simul_description, basefolder=basefolder)
my_simul.setupSimulation(sol0,tend,dt=dt_ini,maxSteps=maxSteps,saveEveryNSteps=5,log_level='INFO',start_step_number=0) #
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

# %%
solN = res[-1]
alpha = alpha_h
from scipy.special import exp1
tts = np.array([res[i].time for i in range(len(res))])
# analytical solution for pressure at collocation points, Eq. 4 in Alexi JMPS
pressure = lambda r, t: (p0 + dp_star * exp1((r**2) / (4.0 * alpha * t)))
p_col = pressure(r_col, tts[-1]) 
plt.figure()
plt.plot(r_col, solN.pressure,'k.')
plt.plot(r_col, p_col,'r.')
plt.xlabel("r (m)")
plt.ylabel("p(r) / po")
plt.show()
# %% slip 
solN = res[-1]

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
