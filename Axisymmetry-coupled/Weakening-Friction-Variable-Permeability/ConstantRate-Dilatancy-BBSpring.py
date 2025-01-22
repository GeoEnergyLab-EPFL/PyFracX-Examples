#
# This file is part of PyFracX-Examples
#

#
# Coupled fluid injection into an axisymmetric frictional fault (0 Poisson's ratio, plane-strain problem)
# Constant injection rate then shut-in
# Linear weakening friction 
# Variable permeability due to dilatancy and nonlinear springs (Barton-Bandis model)
# There is no existing ref solution
#
#%%
import numpy as np
import time
from mesh.usmesh import usmesh
from utils.App import TimeIntegrationApp
from hm.HMFsolver import HMFSolution,hmf_coupled_step

from mechanics.H_Elasticity import Elasticity
from mechanics.mech_utils import MechanicalModel,QuasiDynamicsOperator
from mechanics.MixModeW2D import MixModeW2D,normalStiffness_BB
from mechanics.evolutionLaws import linearEvolution

from flow.flow_utils import *
from flow.FlowConstitutiveLaws import CubicLawNewtonian
from loads.Injection import *

from utils.helper_utils import uniform_interface_properties_BB

################################################ MODEL PARAMETERS ################################################

################################################   1. ROCK PROPERTIES   ######################################
#%% Material parameters, in-sity stress & over-pressure
# material elasticity
YoungM= 40e9  #65e9   # [Pa] Young's modulus
nu= 0.        # [-] Poisson's ratio
shear_g= YoungM/(2*(1+nu))
Kk = 2/3 * shear_g + 2*shear_g*nu/(1-2*nu)

c_s = np.sqrt(shear_g/2700.)        # shear wave speed
c_p = c_s * np.sqrt(2*(1-nu)/(1.-2*nu)) # P -wave speed
eta_s = 0.5 *shear_g / c_s   # Quasi-dynamic s coef
eta_p = 0.5 *shear_g / c_p   # Quasi-dynamic p coef

################################################   2. EFFECTIVE TRACTIONS   ###################################
# # uniform initial effective tractions in Pa
tauo=22.e6         # [Pa]
sigmap_o=36.5e6    # [Pa] 

################################################   3. INTERFACE STIFFNESS   ###################
# Barton-Bandis stiffness, linear slip weakening interface
### stiffnesses - 
ks = YoungM*100 
kni=3.02693e10
v_m=0.000439377
kn_at_o=normalStiffness_BB(-(sigmap_o), kni, v_m)# at sigmap_o
kn_at_opg=normalStiffness_BB(-(0e6), kni, v_m)
wc = v_m / (1 + v_m * kni / (sigmap_o)) #mechanical aperture at which Tn'=0

################################################   4. FRICTION AND DILATANCY   ###################################

#  friction properties (linear weakening)
f_p=0.66      # peak friction 
f_r=0.62       # residual friction 

#  dilatancy (linear decay)
psi_p = 0.1  # peak dilatancy
d_c = 0.0025   # critical slip distances
c_p=0.          # zero cohesion

sig_c = 0.
alpha =0.
beta = 0.

################################################   5. MESH   ################################################

#%% mesh and model creation 
# simple 1D mesh
r_w=0.
Rend = 600.0
h_x = 0.25
Nelts= int(Rend / h_x)
coor1D=np.linspace(r_w,Rend,Nelts+1)
coor = np.transpose(np.array([coor1D,coor1D*0.]))
conn=np.fromfunction(lambda i, j: i + j, (Nelts, 2), dtype=int)
me=usmesh(2,coor,conn,1)

################################################   6. HYDRAULIC PROPERTIES OF THE FAULT ZONE   ###################

# Assume that matrix is impermeable and all the fluid went into one fracture.

# hydraulic properties - cubic law model #
Tav = 1.95398e-15  # average transmibillity w^3/12    [m^3]
alpha_hydro = 0.0000193369  # hydraulic diffusivity  [m^2/s]    - unreliable value

fluid_visc = 2.5e-4  # [Pa*s] fluid dynamic viscosity (value from Andres)
wh_o = (12. * Tav) ** (1 / 3)  # [m] hydraulic width of the fault, cubic law
k_f = wh_o ** 2 / 12.  # permeability
cond_hyd = Tav / fluid_visc  # [m^3/(Pa*s)]
# Compute Storage coefficient / 'fracture+fluid' initial compressibility from value of alpha-> Unreliable
S_e_direct = Tav / (alpha_hydro * fluid_visc) / wh_o  # [1/Pa]

S_e = 0.000117593  # [1/Pa]. - more-realistic - used in the simulation
alpha_hydro = k_f / (S_e * fluid_visc)  # updated value of hydraulic diffusivity.

C_wd = 6.9 * 1.e-8  # [m^3/Pa] wellbore storage  

# %% Injection rate history
QinjTimeData = np.array([[0, 1e-1],[1000, 0.]])
list_times = QinjTimeData[::, 0]
# injection with a given variation of injection rate
the_inj = Injection(np.array([0., 0.]), QinjTimeData, "Rate", volumetric_compressibility=C_wd)

################################################   7. KERNEL   ################################################
# Elasticity discretization via boundary element - plane-strain piece-wise constant displacement discontinuity element
kernel = "Axi3DP0-H"       # axisymmetric simulation 
elas_properties=np.array([YoungM, nu])
elastic_m = Elasticity(kernel, elas_properties, max_leaf_size=32, eta=2., eps_aca=1.0e-6)
# Hmat creation
hmat = elastic_m.constructHmatrix(me)

################################################   8. POPULATE ALL PROPERTIES   ################################

# initial in-situ conditions over the mesh
in_situ_effective_tractions =np.full((Nelts,2),[-tauo,-sigmap_o])  # positive stress in traction !

# material ids
mat_id = np.zeros(me.nelts,dtype=int)

# mechanics
mat_properties = uniform_interface_properties_BB(
    me.nelts, ks, kni, v_m, sig_c, c_p, f_p, f_r, psi_p, wc, d_c, alpha, beta)

# interface
interface_model_BB_LW = MixModeW2D(mat_properties,me.nelts,
    linearEvolution,yield_atol=1.0e-5 * (f_p * sigmap_o - tauo),
    tol=1e-5)

# flow
stor_c = PropertyMap(mat_id, np.array([S_e]))
aperture = PropertyMap(np.zeros(me.nelts, dtype=int), np.array([wh_o]))  # aperture
flow_properties = {"Initial hydraulic width": aperture,
                   "Compressibility": stor_c, "Fluid viscosity": fluid_visc}
cubicModel = CubicLawNewtonian(flow_properties, Nelts)  # instantiate the permeability model
scale_r_f = 1.0  # scale for flow residuals

################################################   DIMENSIONLESS NUMBERS SEE L-W PAPER   ###################
# Important dimensionless numbers 
S_number = tauo/(f_p*sigmap_o)
F_number = f_r/f_p
P_number = 10e6/sigmap_o

dil_max= psi_p*d_c/2.  # max dilation 

print("Initial distance to failure = ", f_p * sigmap_o - tauo)
print("Residual distance to failure = ", f_r * sigmap_o - tauo)
print('Stress criticality ratio : %g ' %(S_number))
print('ratio between residual and peak friction: %g' %(F_number))
print('Max transmissibility increase : %g ' %((1+(dil_max/wh_o))**3))

################################################   ALL PARAMETERS SET   ###################

################################################   PREPARING FOR SIMULATION   ###################

# Testing matvec operation
zt = time.process_time()
for i in range(120):
    hmat @ np.ones(2 * Nelts)
elapsed = (time.process_time() - zt) / 120
print("elapsed time", elapsed)

# mechanical model
qd=QuasiDynamicsOperator(Nelts,2,eta_s,eta_p)

# creating the mechanical model: hmat, preconditioner, number of collocation points, constitutive model
scale_r_m = 1.0 # scale for mech residuals
mech_model = MechanicalModel(hmat, me.nelts,interface_model_BB_LW,precType="Jacobi",QDoperator=qd,scalingX=1.,scalingR=scale_r_m) #"Jacobi"

# flow model
flow_model = FlowModelFractureSegments_axis(me, cubicModel, the_inj, scalingR=scale_r_f, scalingX=1.)

#
model_config = {  # in this dict, we store object etc. (but not the hmat that is not storable for now)
    "Mesh": me,
    "Elasticity": elastic_m,
    "Flow properties": flow_properties,
    "Injection": the_inj,
    "Friction model": interface_model_BB_LW,
    "Material properties": mat_properties,
}

model_parameters = {
    "Elasticity": {"Young": YoungM, "Nu": nu},
    "Flow": {"who": wh_o, "Viscosity": fluid_visc,
             "Compressibility": S_e, "rock diffusivity": alpha_hydro, "Transmissibility": Tav},
    "Injection history": {"Constant Rate": QinjTimeData, "Wellbore storage": C_wd},
    "Initial stress": [sigmap_o, tauo],
    "Friction coefficient": {"peak": f_p, "residual": f_r, "d_c": d_c, "peak dilatancy": psi_p, "form": 'linear'},
    "Interface stiffness": {"ks": ks, "kni": kni, "vm": v_m}
}

#  Preparing folders and base dict for savings
from datetime import datetime
# dd/mm/YY H:M:S
Simul_description="Axisymmetric - Constant rate then shut-in - Variable Permeability - Linear weak. friction - Coupled simulation"
now=datetime.now()
dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
basename="Axisymmetric-ctQ-LinearWeakening-VarPermeability"
basefolder="./"+basename+"-"+dt_string+"/"

# initial solution
sol0=HMFSolution(time=0.,effective_tractions=in_situ_effective_tractions.flatten(), pressure=0.*np.zeros(me.nnodes, dtype=float),
                 DDs=0.*in_situ_effective_tractions.flatten(),
                 DDs_plastic=0.*in_situ_effective_tractions.flatten(),Internal_variables=np.zeros(2*me.nelts),
                 pressure_rate=0.*np.zeros(me.nnodes, dtype=float),
                 DDs_rate=0.*in_situ_effective_tractions.flatten(),
                 res_mech=0.*in_situ_effective_tractions.flatten())

# %% Solver  options
from utils.options_utils import TimeIntegration_options,NonLinear_step_options,NonLinearSolve_options,IterativeLinearSolve_options
# newton solve options
res_atol = 1e-4*scale_r_m* max(np.linalg.norm(in_situ_effective_tractions.flatten()), 1e3)
print("res_atol: %g" %(res_atol))

newton_solver_options=NonLinearSolve_options(max_iterations=20,residuals_atol=res_atol,
                                             residuals_rtol=np.inf,dx_atol=np.inf,dx_rtol=1e-3,
                                             line_search=True,line_search_type="cheap")

# options for the jacobian solver
jac_solve_options = IterativeLinearSolve_options(
    max_iterations=200,
    restart_iterations=150,
    absolute_tolerance=0.,
    relative_tolerance=1e-6,
    preconditioner_side="Left",
    schur_ilu_fill_factor=20,
    schur_ilu_drop_tol=1e-2,
    mech_rtol=1.e-6,
    mech_atol=0.,
    mech_max_iterations=round(me.nelts/2)
)

# combining the 2 as option for the non-linear time-step
step_solve_options=NonLinear_step_options(jacobian_solver_type="GMRES",
                                          jacobian_solver_opts=jac_solve_options,non_linear_start_factor=0.,
                                          non_linear_solver_opts=newton_solver_options)

# %% Function wrapping the one step / coupled solver for this case
def coupledStepWrapper(solF,dt):
    solFNew = hmf_coupled_step(solF, dt, mech_model, flow_model, step_solve_options)
    return solFNew

# - initial time step from pure diffusion 
dt_ini= h_x**2 / alpha_hydro # setting the initial time-step to have something "moving"
tend=1500
maxSteps=5000

# options of the time inegration !  note that we also pass the step_solve_options for consistency
ts_options=TimeIntegration_options(max_attempts=10,dt_reduction_factor=4.,max_dt_increase_factor=5,lte_goal=0.001,
                                   acceptance_a_tol=res_atol,minimum_dt=dt_ini/1000.0,maximum_dt=1e3,
                                   stepper_opts=step_solve_options)

from ts.Event import EnforceTimes
events=EnforceTimes(list_times)
events.setTimeStepAfterEvent(1e-2) # do a small time step after an event, modify if necessary

my_simul=TimeIntegrationApp(basename, model_config, model_parameters, ts_options,coupledStepWrapper,events,description=Simul_description, basefolder=basefolder)
my_simul.setupSimulation(sol0,tend,dt=dt_ini,maxSteps=maxSteps,saveEveryNSteps=1,log_level='INFO',start_step_number=0) #
my_simul.saveParametersToJSon()
me.saveToJson(basefolder+"Mesh.json")

# %% now we are ready to run the simulation
zt= time.process_time()
res,status_ts=my_simul.run()
elapsed=time.process_time()-zt
print("elapsed time",elapsed)

#  This is a coupled simulation. There is NO existing analytical solution to compare with

# %% Postprocess the results
# compute the radial coordinates of the mesh nodes.
x = me.coor[:,0]

Qinj=QinjTimeData[0][1]
tt=np.array([res[j].time for j in range(len(res))])
# # find index of coordinate close to 1. meter
# ind=np.argmin(np.absolute(x-1.))

import matplotlib.pyplot as plt
####################################################################
# plots
timestep=-2
num_pressure=res[timestep].pressure

####################################################################
# %% Overpressure at the well
pw_t=np.array([res[j].pressure[2] for j in range(len(res))])

fig, ax = plt.subplots()
ax.plot(tt[:], 1e-6*pw_t[:],'.b')
plt.xlabel("Time (s)")
plt.ylabel("Well Pressure MPa")
plt.xlim([0,res[-1].time])
plt.show()

#%% ##### rupture radius
nyi=np.array([res[i].Nyielded for i in range(len(res))])
cr_front = nyi*(coor1D[1]-coor1D[0])
fig, ax = plt.subplots()
ax.plot( tt ,cr_front ,'.y')

plt.xlabel("Time (s)")
plt.ylabel("rupture length (m)")
plt.show()

#%%
#### slip at origin

slip_0 = np.abs(np.array([res[i].DDs[0] for i in range(len(res))]))
slip_p_0 = np.abs(np.array([res[i].DDs_plastic[0] for i in range(len(res))]))

fig, ax = plt.subplots()
ax.plot( tt ,slip_0 ,'.k', label='Total Slip')
ax.plot( tt ,slip_p_0 ,'.r', label='Plastic Slip')
plt.xlabel("Time (s)")
plt.ylabel("Slip @ injection point (m)")
plt.legend()
plt.show()

# %% width at origin
width_0 = np.abs(np.array([res[i].DDs[1] for i in range(len(res))]))
width_p_0 = np.abs(np.array([res[i].DDs_plastic[1] for i in range(len(res))]))

fig, ax = plt.subplots()
ax.plot( tt ,width_0 ,'.k', label='Total Opg')
ax.plot( tt ,width_p_0 ,'.b', label='Plastic Opg')
plt.xlabel("Time (s)")
plt.ylabel("Width @ injection point (m)")
plt.legend()
plt.show()

# %%
fig, ax = plt.subplots()
ax.plot( slip_0 ,width_0 ,'.k', label='Total slip (x) VS Total Opg (y)')
ax.plot( slip_p_0 ,width_p_0 ,'.b', label='Plastic slip (x) VS Plastic Opg (y)')
plt.xlabel("Slip @ injection point (m)")
plt.ylabel("Width @ injection point (m)")
plt.legend()
plt.show()

#%%
### profiles
timestep=-1
sol_to_p= res[timestep]
x=me.coor[:,0]
num_pressure=sol_to_p.pressure
fig, ax = plt.subplots()
ax.plot(x[1:400], num_pressure[1:400],'.b')
plt.xlabel("  Radius (m)")
plt.ylabel("Fluid pressure (Pa)")
plt.show()

#%%
# width profile
w_profile=sol_to_p.DDs[1::2]
w_profile_p=sol_to_p.DDs_plastic[1::2]
fig, ax = plt.subplots()
ax.plot((x[1:]+x[0::-2])/2., w_profile[:],'.k')
ax.plot((x[1:]+x[0::-2])/2., w_profile_p[:],'.b')
plt.xlabel("  Radius (m)")
plt.ylabel(" Width (m)")
plt.show()

fig, ax = plt.subplots()

ax.plot((x[1:]+x[0::-2])/2., (1+w_profile[:]/wh_o)**3,'.k')
plt.xlabel("  Radius (m)")
plt.ylabel(" Hydraulic transmissibility increase (-)")
plt.show()

#%%
# slip profile
slip_profile=sol_to_p.DDs[0::2]
slip_profile_p=sol_to_p.DDs_plastic[0::2]
fig, ax = plt.subplots()
# ax.plot(x[1:],true_sol[1:],'r')
ax.plot((x[1:]+x[0::-2])/2., -slip_profile[:],'.k')
ax.plot((x[1:]+x[0::-2])/2., -slip_profile_p[:],'.b')
plt.xlabel("  r (m)")
plt.ylabel("slip (m)")
plt.show()

# dilatancy profile
fig, ax = plt.subplots()
app_dila_c = -sol_to_p.DDs_rate[1::2]/sol_to_p.DDs_rate[0::2]
ax.plot((x[1:]+x[0::-2])/2.,app_dila_c,'.b')
plt.xlabel("  r (m)")
plt.ylabel(" w/slip (-)")
plt.show()

#%%
# effective tracitons profile
tau =sol_to_p.effective_tractions[0::2]
sig_n =sol_to_p.effective_tractions[1::2]

fig, ax = plt.subplots()
# ax.plot(x[1:],true_sol[1:],'r')
ax.plot((x[1:]+x[0::-2])/2., -tau[:],'.k', label='Shear component')
ax.plot((x[1:]+x[0::-2])/2., -sig_n[:],'.b', label='Normal component')
plt.xlabel("  r (m)")
plt.ylabel("effective tractions (Pa)")
plt.legend()
plt.show()


fric_c= [linearEvolution(slip_profile_p[i],d_c,f_p,f_r) for i in range(Nelts)]
dila_c = [linearEvolution(slip_profile_p[i],d_c,psi_p,0.) for i in range(Nelts)]
fig, ax = plt.subplots()
ax.plot((x[1:]+x[0::-2])/2., fric_c,'.k', label='Friction evol.')
ax.plot((x[1:]+x[0::-2])/2., dila_c,'.b', label='Dilatancy evol.')

plt.xlabel("  r (m)")
plt.ylabel("friction coef. & dilatancy coef.")
plt.legend()
plt.show()

###
fig, ax = plt.subplots()
ax.plot((x[1:]+x[0::-2])/2.,np.abs(tau)+fric_c*sig_n,'.b')

plt.xlabel("  r (m)")
plt.ylabel(" F_mc")
plt.show()