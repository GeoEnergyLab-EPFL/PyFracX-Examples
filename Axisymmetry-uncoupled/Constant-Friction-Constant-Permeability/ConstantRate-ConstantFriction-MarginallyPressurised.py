#
# This file is part of PyFracX-Examples
#
#
# Fluid injection at constant rate into a frictional fault in 3D (modelled as axisymmetric problem).
# Reference results from Sáez & Lecampion (2022) 

# %%+
#Importing the necessary python libraries and managing the python path

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path

from ReferenceSolutions.FDFR.frictional_ruptures_3D_constant_friction import *

# %%

# Defining the stress injection parameter value for marginally pressurised simulation

T_exp = 4

# Analytical solution for amplification factor lambda = R(t)/L(t), Eq. 21 from Sáez & Lecampion (2022) 

lam = lambda_analytical(T_exp)
print("Amplification factor expected, lambda = ", lam)

# Plotting the analytical solution for amplification factor with a range of stress injection parameter, 
# Fig. 2 from Sáez & Lecampion (2022) 

T = np.linspace(5e-4, 1e1, 100)
plt.figure()
plt.title("Analytical solution for lambda")
plt.ylabel("T")
plt.xlabel(r"$\lambda$")
plt.plot(lambda_approx_cs(T), T, "--k", label="critically stressed")
plt.plot(lambda_approx_mp(T), T, "--g", label="marginally pressurized")
plt.plot(np.vectorize(lambda_analytical)(T), T, "-m", label="analytical")
plt.semilogy()
plt.semilogx()
# plt.xlim([1e-2,3e1])
# plt.ylim([5e-4,2e1])
plt.legend()

# %% Decide simulation parameters
# Here, we calculate the expected rupture length and decide the mesh size and number of elements

t_first_step = 100  # [s], first time step
alpha = 0.1  #  [m^2/s], rock hyd diffusivity
t_end = 86400  # [s], end time of simulation

# resolving the rupture length/ diffusion length scale at t = t_first_step with 10 elements

hmin = (
    np.min(
        [
            lam * np.sqrt(4 * alpha * t_first_step),
            np.sqrt(4 * alpha * t_first_step),
        ]
    )
    / 10
)

max_rupture_length = lam * np.sqrt(4 * alpha * t_end)
domain_size = 1.5 * np.max([max_rupture_length, np.sqrt(4 * alpha * t_end)])
Nelts = round(domain_size / hmin)

print("domain size", domain_size)
print("nelem", Nelts)
print("size of elem", domain_size / Nelts)

# %% Mesh Generation
# simple AxiSymmetric 1D mesh
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

#radial coordinates of the nodes
r = np.array([linalg.norm(coor[i]) for i in range(Nnodes)])

#radial coordinates of the collocation points
r_col = np.array([linalg.norm(colPts[i]) for i in range(Nelts)])

# %% Elastic Parameters of the simulation
G = 30e9  # [Pa], Shear modulus
nu = 0.0  # Poisson's ratio = 0 because there is analytical solution only for circular ruptures
E = 2 * G * (1 + nu)  # Youngs Modulus
f_p = 0.6  # friction coefficient
f_d = 0.0  # dilatancy coefficient

# %% Flow Parameters of the simulation
# S: Storage [1 / Pa], mu : viscosity [Pa s], wh: hydraulic aperture [m]
# alpha : rock hyd diffusivity,  wh^2/(S mu) [m^2/s]
# Parameters taken from Fig. 3 of Sáez & Lecampion (2022) 

mu = 8.9e-4  # [Pa s], viscosity
wh = (12 * 3e-12) ** (1 / 3)  # [m], hydraulic aperture
cond_hyd = wh**2 / (
    12 * mu
)  # (wh^2 / (12 mu)) [m^2/Pa s], intrinsic perm over viscosity
S = cond_hyd / alpha  # [1/Pa], storage

Qinj = 1.8 / 60  # 1.8 [m^3/min] Wellbore flow rate

# %% Fault Parameters

sigma0 = 120e6  # [Pa], normal stress
# tau0 = 47.958e6  # [Pa], shear stress
p0 = 40e6  # [Pa], background pore pressure
dp_star = Qinj / (4 * cond_hyd * np.pi * wh)

#Calculating the initial shear stress corresponding to the stress injection parameter T
tau0 = f_p * (sigma0 - p0) - f_p * T_exp * dp_star
T = (f_p * (sigma0 - p0) - tau0) / (f_p * dp_star)
print("Stress injection parameter, T = ", T)
print("Shear stress, tau0 = ", tau0)

# %%
from scipy.special import exp1

# analytical solution for pressure at collocation points for constant rate injection
# Eq. 4 in Sáez & Lecampion (2022) 
pressure = lambda r, t: (p0 + dp_star * exp1((r**2) / (4.0 * alpha * t)))
p_col = pressure(r_col, t_end) / p0
plt.figure()
plt.plot(r_col, p_col, ".r")
plt.xlabel("r (m)")
plt.ylabel("p(r) / po")
plt.show()

# %% 
# Elasticity discretization via boundary element - plane-strain piece-wise constant displacement discontinuity element
from mechanics.H_Elasticity import Elasticity

kernel = "Axi3DP0"
elas_properties = np.array([E, nu])
elastic_m = Elasticity(kernel, elas_properties, max_leaf_size=20, eta=4.0, eps_aca=1.0e-3)

# BE H-Matrix construction for the elastic problem
hmat = elastic_m.constructHmatrix(mesh)
# %%
#### Populate material properties

from MaterialProperties import PropertyMap
from mechanics.friction2D import FrictionCt2D
from mechanics.mech_utils import MechanicalModel

friction_c = PropertyMap(
    np.zeros(Nelts, dtype=int), np.array([f_p])
)  # friction coefficient

dilatant_c = PropertyMap(
    np.zeros(Nelts, dtype=int), np.array([f_d])
)  # dilatancy coefficient

k_sn = PropertyMap(
    np.zeros(Nelts, dtype=int),
    np.array([[100 * G, 100 * E]]),
)  # springs shear, normal

mat_properties = {
    "Friction coefficient": friction_c,
    "Dilatancy coefficient": dilatant_c,
    "Spring Cij": k_sn,
    "Elastic parameters": {"Young": E, "Poisson": nu},
}

# instantiating the constant friction model
frictionModel = FrictionCt2D(mat_properties, Nelts)

# creating the mechanical model: hmat, preconditioner, number of collocation points, constitutive model
mech_model = MechanicalModel(hmat, mesh.nelts, frictionModel)
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

# properties are constant, so we construct the FE matrix once and for all.
flow_model.setConductivity(None)
flow_model.setStorage(None)

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
    pressure=po_nodes,
    DDs=0.0 * in_situ_tractions.flatten(),
    DDs_plastic=0.0 * in_situ_tractions.flatten(),
    pressure_rate=np.zeros(mesh.nnodes),
    DDs_rate=0.0 * in_situ_tractions.flatten(),
    Internal_variables=np.zeros(0),
    res_mech=0.0 * in_situ_tractions.flatten(),
)

# %% Solver options

from utils.options_utils import (
    NonLinearSolve_options,
    IterativeLinearSolve_options,
    NonLinear_step_options,
)

# newton solve options
newton_solver_options = NonLinearSolve_options(
    max_iterations=25,
    residuals_atol=0.5,
    residuals_rtol=1.0,
    dx_atol=np.inf,
    dx_rtol=1e-3,
    line_search=True,
)
# options for the jacobian solver
jac_solve_options = IterativeLinearSolve_options(
    max_iterations=300,
    absolute_tolerance=1e-6,
    relative_tolerance=1e-6,
    restart_iterations=150,
)
# combining the 2 as option for the non-linear time-step
step_solve_options = NonLinear_step_options(
    jacobian_solver_opts=jac_solve_options,
    non_linear_start_factor=0.0,
    non_linear_solver_opts=newton_solver_options,
)

# %%
## function wrapping the one-way H-M / uncoupled solver for this case
from hm.HMFsolver import hmf_one_way_flow_numerical, HMFSolution

Dtinf = np.zeros(2 * mesh.nelts)

def onewayStepWrapper(solN: HMFSolution, dt: float) -> HMFSolution:
    solTnew = hmf_one_way_flow_numerical(
        solN, dt, mech_model, Dtinf, flow_model, step_solve_options
    )
    return solTnew
# %%
#Storing the configuration and properties of the model
model_config = {  # in this dict, we store object etc. (but not the hmat that is not storable for now)
    "Mesh": mesh,
    "Elasticity": elastic_m,
    "Friction model": frictionModel,
    "Material properties": mat_properties,
}
model_parameters = {
    "Elasticity": {"Young": E, "Nu": nu},
    "Injection": {"Injection rate": Qinj},
    "Flow": {"Hydraulic diffusivity": alpha, "Hydraulic conductivity": cond_hyd},
    "Hydraulic aperture": wh,
    "Initial stress": [sigma0, tau0],
    "Friction coefficient": f_p,
    "T parameter": T,
}
# %%
from utils.App import TimeIntegrationApp
from utils.options_utils import TimeIntegration_options

Simul_description = (
    "Axi Symm - ct rate - ct friction - one way simulation - flow numerical - marginally pressurized"
)
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y-%H:%M:%S")
basename = "AxiSymm-ctRate-ctFriction-oneWayFlowNumerical-marpress"
basefolder = "./"+basename+"-"+dt_string+"/"

#Path(basefolder).mkdir(parents=True, exist_ok=True)

# prepare the time-stepper simulations
new_dt = t_first_step    # initial time-step
maxSteps = 100           #120 max number of stpes of the simulation
tend = t_end             # max time to simulate

# options of the time inegration !  note that we also pass the step_solve_options
ts_options = TimeIntegration_options(
    max_attempts=4,
    dt_reduction_factor=1,
    max_dt_increase_factor=1.3,
    minimum_dt = new_dt,
    acceptance_a_tol=0.1,
    stepper_opts=step_solve_options,
    lte_goal=0.01,
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

#my_simul.setAdditionalStoppingCriteria(lambda sol: sol.Nyielded == mesh.nelts // 2)

my_simul.setupSimulation(
    sol0,
    tend,
    dt=new_dt,
    maxSteps=maxSteps,
    saveEveryNSteps=1,
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

# %% Post-processing
tts = np.array([res[i].time for i in range(len(res))])
nyi = np.array([res[i].Nyielded for i in range(len(res))])
#tts = tts[nyi > 1]
# find position of last yielded element (slip becomes zero)
rmax = 0.0 * tts
'''
for i in range(len(tts)):
    aux_k = np.where(res[i].yieldedElts)[0]
    rmax[i] = r_col[aux_k].max()
'''

for i in range(len(res)):
    aux_k = np.where(res[i].yieldedElts)[0]
    
    if len(aux_k) > 0:
        rmax[i] = r_col[aux_k].max()
    else:
        rmax[i] = 0.0  # or some other appropriate value

# %% Analytical solution
lam = lambda_analytical(T)
print("lambda = ", lam)
print("Predicted Rmax = ", np.sqrt(4.0 * alpha * tend) * lam)
t_ = np.linspace(tts[0], tts.max())
R = lam * np.sqrt(4.0 * alpha * t_)
plt.figure()
plt.plot(t_, R, "r")
plt.plot(tts, rmax, ".")
plt.semilogx()
plt.semilogy()
plt.xlabel("Time (s)")
plt.ylabel("Rupture radius (m)")
plt.legend(["Analytical solution", "Numerics"])
plt.show()
# %% pressure plot

time_step = 1   
p_anal = pressure(coor1D[:], res[time_step].time)
fig, ax = plt.subplots()
ax.plot(coor1D[:], res[time_step].pressure[:] / p0, ".")
ax.plot(coor1D[:], p_anal / p0, "-g")

plt.xlabel("x (m)")
plt.ylabel(" Pressure / p0")
plt.show()


# %% slip plot
fig, ax = plt.subplots()
ax.plot(colPts, -res[-1].DDs[0:-1:2], ".")
# ax.plot(coor1D[:],p_anal,'-g')

plt.xlabel("r (m)")
plt.ylabel(" slip ")
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
    (np.abs(lam - rmax[-1] / np.sqrt(4 * alpha * tts[-1])) / lam) * 100.0,
)

#%%
#Computing the complete analytical solution given at 
# "Asymptotic solutions for self-similar fault slip induced by fluid injection at constant rate"
# by Viesca (2024)
import scipy.special
dp = Qinj/(4*np.pi*cond_hyd*wh)
alpha_new = 4*alpha
lam = lambda_analytical(T)
rmax_analytical = lam * np.sqrt(4 * alpha * tts[-1])
t = tts[-1]

# %%
fig, ax = plt.subplots()
ax.plot(r_col, 
        -res[-1].DDs_plastic[0:-1:2], 
        "--r", 
        label = "Numerical")

ax.plot(
    r_col,
    complete_slip_profile_mp(G, f_p, dp, alpha, T, tts[-1], lambda_analytical, r_col),
    "-k",
    ms=1.2,
    label="Analytical",
)
plt.xlabel("r")
plt.ylabel("slip")
plt.legend()

#%%
fig, ax = plt.subplots()
ax.plot(r_col, 
        -res[-1].DDs_plastic[1::2]+res[-1].DDs[1::2], 
        "--r", 
        label = "Numerical Opening")

plt.xlabel("r")
plt.ylabel("Opening")
plt.legend()

#%%
fig, ax = plt.subplots()
ax.plot(r_col, 
        -res[-1].DDs_plastic[0::2]+res[-1].DDs[0::2], 
        "--r", 
        label = "Elastic Slip")
ax.plot(r_col, 
        -res[-1].DDs_plastic[0::2], 
        "--b", 
        label = "Plastic Slip")
plt.xlabel("r")
plt.ylabel("Slip")
plt.legend()
#%%
fig, ax = plt.subplots()
ax.plot(r_col, 
        res[-1].yieldF, 
        "--r", 
        label = "Yield Function")
ax.plot(r_col, 
        -res[-1].effective_tractions[0::2]+f_p*res[-1].effective_tractions[1::2], 
        "--r", 
        label = "Yield Function")

plt.xlabel("r")
plt.ylabel("Yield Function")
plt.legend()
