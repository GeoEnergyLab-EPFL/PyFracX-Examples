#
# This file is part of PyFracX-Examples
# Fluid injection at constant rate into a slip-weakening frictional fault in 3D (modelled as axisymmetric problem).
# Reference results from Sáez & Lecampion (2023) 

# %%
#Importing the necessary python libraries and managing the python path

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path

from ReferenceSolutions.FDFR.frictional_ruptures_3D_constant_friction import *

#%%
# Set the dimensionless parameters of the simulation defined in Section 4 in Sáez & Lecampion (2023)

S_fac = 0.6 #Pre-stress ratio or stress criticiality
F = 0.7 #Residual to Peak friction ratio
Op = 0.035 #Overpressure ratio

Op_c = 0.1*(1-S_fac) #Critical overpressure ratio

# Checking the stability of the fault

print("Pre-stress ratio, S = ",S_fac)
print("Residual to peak friction ratio, F = ", F)
print("Overpressure ratio, P =", Op)
print("Critical overpressure ratio, P_c =", Op_c)

if Op > Op_c:
    print("Slip activates")
    if S_fac < F:
        print("Ultimately stable regime, Tr > 0")
    else:
        print("Unstable regime, Tr < 0")
else:
    print("No slip activation")


# %% Elastic Parameters of the simulation

G = 30e9  # [Pa], Shear modulus
nu = 0.0  # Poisson's ratio = 0 because there is analytical solution only for circular ruptures
E = 2 * G * (1 + nu)  # Youngs Modulus
alpha = 0.1  #  [m^2/s], rock hyd diffusivity

 # %% Flow Parameters of the simulation

# S: Storage [1 / Pa], mu : viscosity [Pa s], wh: hydraulic aperture [m]
# alpha : rock hyd diffusivity,  wh^2/(S mu) [m^2/s]

mu = 8.9e-4  #[Pa s], viscosity
wh = (12 * 3e-12) ** (1/3)  # [m], hydraulic aperture
cond_hyd = wh**2 / (12 * mu)  # (wh^2 / (12 mu)) [m^2/Pa s], intrinsic perm over viscosity
S = cond_hyd / alpha  # [1/Pa], storage

# %% Fault Parameters

f_p = 0.6 #peak-friction coefficient
f_r = F * f_p #residual friction coefficient
f_d = 0.0 #dilatancy coefficient
cohesion = 0.0  # [Pa]
sigma0 = 120e6  # [Pa], normal stress 
p0 = 0  # [Pa], initial pore pressure
tau0 = S_fac * f_p * (sigma0-p0)  # [Pa], shear stress 2.0235414193112407e7
dp_star = Op * (sigma0-p0)  #Characteristic pressure drop [Pa]
Qinj =  dp_star* (4 * cond_hyd * np.pi * wh) #wellbore flow rate: m^3/s

#tau0 = f_p * (sigma0 - p0) - f_p * T_expected * dp_star

#Computing the stress-injection parameters

T_p = (f_p * (sigma0 - p0) - tau0) / (f_p * dp_star)
T_r = (f_r * (sigma0 - p0) - tau0) / (f_r * dp_star)

print("Peak stress injection parameter, T_p = ", T_p)
print("Residual stress injection parameter, T_r= ", T_r)
print("Initial shear stress, tau0 = ", tau0)

#Corresponding amplification factors

lam_p = lambda_analytical(T_p)
lam_r = lambda_analytical(T_r)

print("Lambda_p = ", lam_p)
print("Lambda_r = ", lam_r)
# %% Mesh parameters
# At late time it will approach the lambda expected corresponding to the residual stress injection parameter
# resolving the rupture length/ diffusion length scale at t = t_first_step with 10 elements
# defining the critical slip distance

d_c = 9.e-4  # [m], critical slip distance 

# Computing the rupture length scale defined in Section 4.1 in Sáez & Lecampion (2023)

Rw = (d_c * G) / ((f_p-f_r) * (sigma0-p0))

# Set the time of simulation to reproduce plots in Figure 5 of Sáez & Lecampion (2023)
# Get x_ord (sqrt(4 * alpha * t)/ Rw) from the plot you want to reproduce in Figure 5 of Sáez & Lecampion (2023)

x_ord = 400

t_end = (x_ord **2 * Rw **2) / (4 * alpha) # [s], end time of simulation
t_first_step = t_end/100 # [s], first time step

# Computing the domain size and element size

domain_size = 1.5*np.max([lam_r * np.sqrt(4 * alpha * t_end), np.sqrt(4 * alpha * t_end)])

elem_size = Rw/10

Nelts = round(domain_size/elem_size)

print("domain size", domain_size)
print("nelem", Nelts)
print("size of elem", elem_size)
print("Critical slip distance", d_c)
print("Rupture length scale", Rw)
print("Time of simulation", t_end)
print("First time step", t_first_step)

# %% Mesh generation
# simple AxiSymm 1D mesh

from mesh.usmesh import usmesh
from scipy import linalg

coor1D = np.linspace(0.0, domain_size, Nelts + 1)
coor = np.transpose(np.array([coor1D, coor1D * 0.0]))
conn = np.fromfunction(lambda i,j: i + j, (Nelts, 2), dtype=int)
mesh = usmesh(2, coor, conn, 0)

Nelts = mesh.nelts
Nnodes = mesh.nnodes
coor = np.asarray(mesh.coor)
conn = np.asarray(mesh.conn)
colPts = (coor1D[1:] + coor1D[0:-1]) / 2.0  # collocation points for P0

r = np.array([linalg.norm(coor[i]) for i in range(Nnodes)])
r_col = np.array([linalg.norm(colPts[i]) for i in range(Nelts)])

# %% 
# Elasticity discretization via boundary element
from mechanics.H_Elasticity import Elasticity

kernel = "Axi3DP0"
elas_properties = np.array([E, nu])
elastic_m = Elasticity(kernel, elas_properties, max_leaf_size = 128, eta = 3.0, eps_aca=1.0e-3)

# BE H-Matrix construction for the elastic problem
hmat = elastic_m.constructHmatrix(mesh)

# %%

#### Populate material properties


from MaterialProperties import PropertyMap
from mechanics.friction2D import *
from mechanics.evolutionLaws import *
from mechanics.mech_utils import *

# Preparing  properties for simulation
friction_p = PropertyMap(np.zeros(Nelts, dtype=int), 
                         np.array([f_p]))  #peak friction coefficient
friction_r = PropertyMap(np.zeros(Nelts, dtype=int), 
                         np.array([f_r]))  #residual friction coefficient
dilatant_c = PropertyMap(np.zeros(Nelts, dtype=int), 
                         np.array([f_d]))  # dilatancy coefficient
cohesion_p=PropertyMap(np.zeros(Nelts,dtype=int),
                       np.array([cohesion]))
slip_dc = PropertyMap(np.zeros(Nelts,dtype=int),
                      np.array([d_c]))

k_sn = PropertyMap(
    np.zeros(Nelts, dtype=int),
    np.array([[100 * G, 100* E]]),
)  # springs shear, normal

mat_properties = {
    "Peak friction": friction_p,
    "Residual friction": friction_r,
    "Peak dilatancy coefficient": dilatant_c,
    "Critical slip distance":slip_dc,
    "Peak cohesion": cohesion_p,
    "Spring Cij": k_sn,
    "Elastic parameters": {"Young": E, "Poisson": nu},
}

# instantiating the slip-weakening friction model

##### MANUAL INPUT OF THE EVOLUTION LAWS #####

fric_model = "Linear" #or "Linear" or "Exponential"

#### "linearEvolution" : Linear slip-weakening evolution law ######
#### "expEvolution" : Exponential slip-weakening evolution law ######

frictionModel = FrictionVar2D(mat_properties, Nelts, linearEvolution)#####

# quasi-dynamics term (put here to a small value)
c_s = np.sqrt(G/1.)
c_p = c_s * np.sqrt(2*(1-nu)/(1.-2*nu))
eta_s = 1.e-4*0.5 * G / c_s
eta_p = 1.e-4*0.5 * G / c_p
qd = QuasiDynamicsOperator(Nelts, 2, eta_s, eta_p)

# creating the mechanical model: hmat, preconditioner, number of collocation points, constitutive model
mech_model = MechanicalModel(hmat, mesh.nelts, frictionModel, QDoperator=qd)

# %% Flow model
from flow.FlowConstitutiveLaws import ConstantPerm
from flow.flow_utils import FlowModelFractureSegments_axis
from loads.Injection import Injection

cond_c = PropertyMap(
    np.zeros(mesh.nelts, dtype=int), np.array([cond_hyd * wh])
)  # uniform hydraulic conductivity
stor_c = PropertyMap(
    np.zeros(mesh.nelts, dtype=int), np.array([S * wh])
)  # uniform hydraulic storage
flow_properties = {"Conductivity": cond_c, "Storage": stor_c}
constPerm = ConstantPerm(
    flow_properties, mesh.nelts
)  # instantiate a constant permeability/storage model

# injection under constant pressure
the_inj = Injection(np.array([0.0, 0.0]), np.array([[0.0, Qinj]]), "Rate")

flow_model = FlowModelFractureSegments_axis(mesh, constPerm, the_inj)

# %%
from hm.HMFsolver import HMFSolution

#initial pore pressure over the nodes and the collocation points
po_nodes = np.zeros(mesh.nnodes, dtype=float) + p0
po_col = np.zeros(mesh.nelts, dtype=float) + p0

#initial in-situ tractions over the mesh
in_situ_tractions = np.full(
    (Nelts, 2), [-tau0, -sigma0]
)  # positive stress in traction ! tension positive convention
effective_tractions = in_situ_tractions.copy()
effective_tractions[:, 1] += po_col


# Initial solution
sol0 = HMFSolution(
    time=0.0,
    effective_tractions=effective_tractions.flatten(),
    pressure= po_nodes,
    DDs=0.0 * in_situ_tractions.flatten(),
    DDs_plastic=0.0 * in_situ_tractions.flatten(),
    pressure_rate=np.zeros(mesh.nnodes),
    DDs_rate=0.0 * in_situ_tractions.flatten(),
    Internal_variables=np.zeros(2*mesh.nelts),
    res_mech=0.0 * in_situ_tractions.flatten(),
)

# %% Solver options
# newton solve options
from utils.options_utils import (
    TimeIntegration_options,
    NonLinearSolve_options,
    IterativeLinearSolve_options,
    NonLinear_step_options,
)
res_atol = 1.e-3*max(np.linalg.norm(in_situ_tractions.flatten()),1e3)
print("res_atol: %g" %(res_atol))

newton_solver_options = NonLinearSolve_options(
    max_iterations=100,
    residuals_atol=res_atol,
    residuals_rtol=np.inf,
    dx_atol=np.inf,
    dx_rtol=5e-3,
    line_search=True,
    line_search_type="cheap",
    verbose=True,
)
# options for the jacobian solver
jac_solve_options = IterativeLinearSolve_options(
    max_iterations=50,
    restart_iterations=50,
    absolute_tolerance=np.inf,
    relative_tolerance=1e-4,
    preconditioner_side="Left",
    schur_ilu_fill_factor=50,
    schur_ilu_drop_tol=1e-4,
    mech_rtol=1e-4,
    mech_atol=1,
    mech_max_iterations=mesh.nelts/2
)
# combining the 2 as option for the non-linear time-step
step_solve_options = NonLinear_step_options(
    jacobian_solver_type="GMRES",
    jacobian_solver_opts=jac_solve_options,
    non_linear_start_factor=1.0,
    non_linear_solver_opts=newton_solver_options,
)

# %%
## function wrapping the one-way H-M solver for this case
from hm.HMFsolver import hmf_one_way_flow_numerical, HMFSolution

Dtinf = np.zeros(2 * mesh.nelts)

def onewayStepWrapper(solN: HMFSolution, dt: float) -> HMFSolution:
    solTnew = hmf_one_way_flow_numerical(
        solN, dt, mech_model, Dtinf, flow_model, step_solve_options
    )
    return solTnew

# %%
model_config = {  # in this dict, we store object etc. (but not the hmat that is not storable for now)
    "Mesh": mesh,
    "Elasticity": elastic_m,
    "Friction model": frictionModel,
    "Material properties": mat_properties,
}
model_parameters = {
    "Elasticity": {"Young": E, "Nu": nu},
    "Injection": {"Injection rate": Qinj},
    "Friction model": {"Peak friction": f_p, "Residual friction": f_r, "Model": fric_model},
    "Flow": {"Hydraulic diffusivity": alpha, "Hydraulic conductivity": cond_hyd},
    "Initial stress": [sigma0, tau0],
    "Hydraulic aperture": wh,
    "Peak to residual friction":F,"Dilatancy peak":f_d,"Critical slip":d_c,
    "T peak": T_p,
    "T residual": T_r,
    "Pre-stress ratio": S_fac,
    "Overpressure ratio": Op,
    "Rupture length scale": Rw
}

# %%
from utils.App import TimeIntegrationApp
from utils.options_utils import TimeIntegration_options

new_dt = t_first_step/100
maxSteps = 1e10
tend = t_end

Simul_description = (
    "AxiSymm - ct rate - linear weakening friction - one way simulation - flow numerical"
)
now = datetime.now()

dt_string = now.strftime("%d-%m-%Y-%H:%M:%S")

basename = "AxiSymm-ctRate-NumFlow-lwfric-S_0.6-P_0.035"

basefolder = "./" + basename + "-" + dt_string+"/"

ts_options = TimeIntegration_options(
    max_attempts=10,
    dt_reduction_factor=1.5,
    max_dt_increase_factor = 1.03,
    lte_goal = 0.001,
    minimum_dt = new_dt/100,
    maximum_dt = new_dt*20,
    acceptance_a_tol= res_atol,
    stepper_opts=step_solve_options,
)


my_simul = TimeIntegrationApp(
    basename,
    model_config,
    model_parameters,
    ts_options,
    onewayStepWrapper,
    description=Simul_description,
    basefolder=basefolder,
)
my_simul.setAdditionalStoppingCriteria(lambda sol: sol.Nyielded > int(0.95* mesh.nelts))

my_simul.setupSimulation(
    sol0,
    tend,
    dt=new_dt,
    maxSteps=maxSteps,
    saveEveryNSteps=10,
    log_level="INFO",
)
my_simul.saveParametersToJSon()
my_simul.saveConfigToBinary()
mesh.saveToJson(basefolder + "Mesh.json")

# %% Simulation
# now we are ready to run the simulation
import time

zt = time.process_time()
res, status_ts = my_simul.run()
elapsed = time.process_time() - zt
print("End of simulation in ", elapsed)

# %%
from scipy.special import exp1

# analytical solution for pressure at collocation points, Eq. 4 in Alexi JMPS
pressure = lambda r, t: (p0 + dp_star * exp1((r**2) / (4.0 * alpha * t)))
# %% Post-processing
tts = np.array([res[i].time for i in range(len(res))])
nyi = np.array([res[i].Nyielded for i in range(len(res))])
#tts = tts[nyi > 1]
# find position of last yielded element (slip becomes zero)
rmax = 0.0 * tts

for i in range(len(res)):
    aux_k = np.where(res[i].yieldedElts)[0]
    
    if len(aux_k) > 0:
        rmax[i] = r_col[aux_k].max()
    else:
        rmax[i] = 0.0  # or some other appropriate value

# %% pressure plot
time_step = -1   
p_anal = pressure(coor1D[:], res[time_step].time)
fig, ax = plt.subplots()
ax.plot(coor1D[:], res[time_step].pressure[:], ".")
ax.plot(coor1D[:], p_anal, "-g")
plt.xlabel("x (m)")
plt.ylabel(" Pressure (Pa) ")
plt.gca().legend(["Analytical", "Numerics"])
plt.title("Pressure at t = %.3f secs" % (res[time_step].time))
plt.show()

# %% slip plot
fig, ax = plt.subplots()
ax.plot(colPts, -res[-1].DDs[0:-1:2], ".")
# ax.plot(coor1D[:],p_anal,'-g')
plt.xlabel("r (m)")
plt.ylabel(" slip ")
plt.title("Total slip at t = %.3f secs" % (res[-1].time))
plt.show()
# %% Analytical solution
#lam = lambda_analytical(T)
#print("lambda = ", lam)
#print("Predicted Rmax = ", np.sqrt(4.0 * alpha * tend) * lam)
t_ = np.linspace(tts[0], tts.max())
#R = lam * np.sqrt(4.0 * alpha * t_)
plt.figure()
#plt.plot(t_, R, "r")
plt.plot(tts, rmax, ".")
plt.semilogx()
#plt.semilogy()
plt.xlabel("Time (s)")
plt.ylabel("Rupture radius (m)")
plt.legend(["Numerics"])
plt.show()
# %%
plt.figure()
plt.plot(tts, lambda_analytical(T_p)* np.ones(len(tts)), "-k")
plt.plot(tts, lambda_analytical(T_r)* np.ones(len(tts)), "-b")
plt.plot(tts, rmax / np.sqrt(4 * alpha * tts), "-r")
plt.semilogx()
plt.xlabel("Time (s)")
plt.ylabel(r"$\lambda$ = Rupture radius / $\sqrt{4 \alpha t}$")
plt.gca().legend(["$\lambda_p$ ","$\lambda_r$","Numerics"])
#plt.title(r"T = %.3f, $\lambda = %.2f$" % (T_p, lam))
# %%
# %%
plt.figure()
plt.plot(tts, lambda_analytical(T_p)* np.ones(len(tts)), "-k")
plt.plot(tts, lambda_analytical(T_r)* np.ones(len(tts)), "-b")
plt.plot(tts, rmax / np.sqrt(4 * alpha * tts), "-r")
plt.semilogx()
plt.xlabel("Time (s)")
plt.ylabel(r"$\lambda$ = Rupture radius / $\sqrt{4 \alpha t}$")
plt.gca().legend(["$\lambda_p$ ","$\lambda_r$","Numerics"])
#plt.title(r"T = %.3f, $\lambda = %.2f$" % (T_p, lam))

#%%
x_cor = np.sqrt(4*alpha*tts)
x_cor = x_cor/Rw
y_cor = rmax/Rw
plt.figure()
plt.plot(x_cor, y_cor, "-.r",label = 'numerical: LW')
plt.xlabel(r"$\sqrt{4 \alpha t}/R_w$")
plt.ylabel(r"$R/R_w$")
plt.legend()

# %%
