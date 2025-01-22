# %%
##!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %%
import os
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt


# %% Mesh
from mesh.usmesh import usmesh

# simple 1D mesh uniform
Nelts = 1000
Rinf = 20
coor1D = np.linspace(0, Rinf, Nelts + 1)
coor = np.transpose(np.array([coor1D, coor1D * 0.0]))
conn = np.fromfunction(lambda i, j: i + j, (Nelts, 2), dtype=int)
mesh = usmesh(2, coor, conn, 0)

colPts = (coor1D[1:] + coor1D[0:-1]) / 2.0  # collocation points for P0
col_pts = np.c_[colPts, np.zeros(colPts.shape[0])]
h_x = coor[1, 0] - coor[0, 0]
print("Mesh size", h_x)

# %% Elasticity Parameters
G = 20e9  # Young's Modulus [Pa]
nu = 0.0 # Poisson's ratio
E = 2 * G * (1 + nu)  # Shear modulus
E_prime = E / (1 - nu * nu)

# %% Fluid Flow Parameters
k = 1.5e-17  # [m^2] permeability
H = 550  # [m] shear zone thickness
wo = (12 * k * H) ** (1 / 3)  # [m] assuming cubic law
fluid_visc = 8.9e-4  # [Pa s] fluid viscosity
cf = 4.5e-10  # [Pa^-1] compressibility of the fluid
fluid_visc_prime = fluid_visc * 12


# %% Field Parameters
# fault friction
fp = 0.6 # peak friction
fd = 0.0 # dilatancy

# dimensionless numbers
taubyfsigop = 0.0 # \tau / (f \sigma_o^p), should be zero for no shear crack @INPUT
pcbysigop = 100 # p_c / \sigma_o^p @INPUT
beta_s = 0.869 # Skempton coefficient @INPUT

# far-field effective tractions
sigo = 15e6  # [Pa]
po = 5e6  # [Pa]
sigop = sigo - po
tauo = taubyfsigop * (fp * sigop)

Qinj = pcbysigop * (k * H * sigop) / (fluid_visc)

ks = 1e3 * G  # shear spring [Pa/m]
# kn = max(
#     2e2 * E_prime, 1e1 * sigop / wo, 1e1 / ((cf) * wo), 1e1 * ks
# )  # (sig_o_p / wh_o)  # opening spring [Pa/m]
# kn = 1e1 * kn
kn = (1 - beta_s) / (beta_s * cf  * wo)


# %% # Elasticity model
from mechanics.H_Elasticity import Elasticity
import time

kernel = "Axi3DS0-H"
elas_properties = np.array([E, nu])
elastic_m = Elasticity(
    kernel, elas_properties, max_leaf_size=32, eta=3.0, eps_aca=1.0e-4,n_openMP_threads=8
)
# hmat creation
hmat = elastic_m.constructHmatrix(mesh)

# Testing matvec operation
zt = time.process_time()
for i in range(20):
    hmat.dot(np.ones(2 * Nelts))
elapsed = (time.process_time() - zt) / 20
# print("number of active threads for MatVec", hmat.get_omp_threads())
print("elapsed time", elapsed)

# %% Interface law
from MaterialProperties import PropertyMap
from mechanics.friction2D import FrictionCt2D

friction_c = PropertyMap(
    np.zeros(Nelts, dtype=int), np.array([fp])
)  # friction coefficient
dilatant_c = PropertyMap(
    np.zeros(Nelts, dtype=int), np.array([fd])
)  # dilatancy coefficient
k_sn = PropertyMap(
    np.zeros(Nelts, dtype=int),
    np.array([[ks, kn]]),
)  # springs shear, normal
mat_properties = {
    "Friction coefficient": friction_c,
    "Dilatancy coefficient": dilatant_c,
    "Spring Cij": k_sn,
    "Elastic parameters": {"Young": E, "Poisson": nu},
}
interface_Model = FrictionCt2D(mat_properties, Nelts, yield_atol=1.0e-6 * sigop)

# %% Mechanical Model
from mechanics.mech_utils import MechanicalModel

mech_model = MechanicalModel(hmat, mesh.nelts, interface_Model)

# %% Flow Model
from flow.flow_utils import FlowModelFractureSegments_axis
from flow.FlowConstitutiveLaws import CubicLawNewtonian
from loads.Injection import Injection


# Injection under constant rate
the_inj = Injection(np.array([0.0, 0.0]), np.array([[0.0, Qinj]]), "Rate")

# Creation of the flow model :: flow cubic law
wolist = PropertyMap(
    np.zeros(mesh.nelts, dtype=int), np.array([wo])
)  # uniform hydraulic width
cflist = PropertyMap(np.zeros(mesh.nelts, dtype=int), np.array([cf]))
wc = sigop / kn

flow_properties = {
    "Fluid viscosity": fluid_visc,
    "Initial hydraulic width": wolist,
    "Compressibility": cflist,
}
# instantiate a cubic law model
cubicModel = CubicLawNewtonian(flow_properties, mesh.nelts)
flow_model = FlowModelFractureSegments_axis(mesh, cubicModel, the_inj)


# %% Initial conditions
from hm.HMFsolver import HMFSolution, hmf_coupled_step

# time stepping parameters
tinitial = 0
maxSteps = 1500
tend = 1e6

DDs_initial = np.zeros((mesh.nelts, 2))
pressure0_nodes = np.zeros(mesh.nnodes) + po

effective_traction = np.full((mesh.nelts, 2), [-tauo, -sigop])
yieldF = np.abs(effective_traction[:, 0]) + fp * effective_traction[:, 1]
yieldElts = np.zeros(mesh.nelts, dtype=bool)

sol0 = HMFSolution(
    time=tinitial,
    effective_tractions=effective_traction.flatten(),
    pressure=pressure0_nodes,
    DDs=DDs_initial.flatten(),
    DDs_plastic=DDs_initial.flatten(),
    pressure_rate=0.0 * pressure0_nodes,
    DDs_rate=0.0 * DDs_initial.flatten(),
    Internal_variables=np.zeros(0),
    yieldF=yieldF,
    yieldedElts=yieldElts,
    res_mech=effective_traction.flatten() * 0.0,
    res_flow=pressure0_nodes * 0.0,
)

# %% Solver options
from utils.options_utils import NonLinear_step_options
from utils.options_utils import IterativeLinearSolve_options
from utils.options_utils import NonLinearSolve_options


# max 1 kPa, norm(t0)
res_atol = 1e-3 * max(np.linalg.norm(effective_traction.flatten()), 1e3 * mesh.nelts)

print("res_atol", res_atol)
# newton solve options
newton_solver_options = NonLinearSolve_options(
    max_iterations=20,
    dx_rtol=1e-3,
    dx_atol=np.inf,
    residuals_atol=res_atol,
    residuals_rtol=np.inf,
    line_search=False,
    line_search_type="None",
)

# options for the jacobian solver inside Newton step
jac_solve_options = IterativeLinearSolve_options(
    max_iterations=200,
    restart_iterations=200,
    absolute_tolerance=0.,
    relative_tolerance=1e-5,
    preconditioner_side="Left",
    schur_ilu_fill_factor=10,
    schur_ilu_drop_tol=1e-4,
    mech_rtol=1e-5,
    mech_atol=0.,
    mech_max_iterations=300,
)

# combining the 2 as option for the non-linear time-step
step_solve_options = NonLinear_step_options(
    jacobian_solver_opts=jac_solve_options,
    non_linear_start_factor=0.0,
    non_linear_solver_opts=newton_solver_options,
)

# %% Define solver wrapper
from hm.HMFsolver import HMFSolution, hmf_coupled_step


def StepWrapper(solN: HMFSolution, dt: float) -> HMFSolution:
    # if solN.time < 1e2:
    #     step_solve_options.non_linear_solver_opts.dx_rtol = 1e-2
    # else:
    #     step_solve_options.non_linear_solver_opts.dx_rtol = 1e-3
    solNew = hmf_coupled_step(solN, dt, mech_model, flow_model, step_solve_options)
    return solNew


# %% Save simulation parameters in json
model_config = {  # in this dict, we store object etc. (but not the hmat that is not storable for now)
    "Mesh": mesh,
    "Elasticity": elastic_m,
    "Friction model": interface_Model,
    "Material properties": mat_properties,
    "Flow properties": flow_properties,
    "Injection": the_inj,
}
# in this dict, we store parameters and description...
model_parameters = {
    "Elasticity": {"Young": E, "Nu": nu, "Kernel": kernel},
    "Flow": {
        "who": wo,
        "Viscosity": fluid_visc,
        "Compressibility": cf,
    },
    "Injection": {"Constant Rate": Qinj},
    "Initial stress": [tauo, sigo],
    "Pore pressure": po,
    "Friction coef": fp,
    "Dilation coef": fd,
    "Shear spring": ks,
    "Normal spring": kn,
}

# %% Create the simulation folder
# dd/mm/YY H:M:S
folder ="./" # + "/data/" # @INPUT, where to save the data
Simul_description = "3D Axis Symm - constant rate - zero toughness Hydraulic Fracture"
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
basename = "3DAxiSymmHF"
basefolder = folder + basename + "-" + dt_string + "/"

Path(basefolder).mkdir(parents=True, exist_ok=True)



# %% Simulation options

from utils.options_utils import TimeIntegration_options
from utils.App import TimeIntegrationApp

ts_options = TimeIntegration_options(
    max_attempts=10,
    dt_reduction_factor=2.0,
    max_dt_increase_factor=1.035,
    lte_goal=1e-3,
    minimum_dt=1.0e-6,
    maximum_dt=1e3,
    stepper_opts=step_solve_options,
    acceptance_a_tol=step_solve_options.non_linear_solver_opts.residuals_atol,
)

# %% 
h_r = Rinf / Nelts
alpha = wo**2 / (12 * fluid_visc * cf)
dt_ini = min(1 * h_r**2 / alpha, 0.1)
print("dt_ini", dt_ini)

# %% Create accepatnce criteria

norm = lambda x: np.linalg.norm(x, ord=np.inf)
norm2 = lambda x: np.linalg.norm(x)


# %%
my_event = None

my_simul = TimeIntegrationApp(
    basename,
    model_config,
    model_parameters,
    ts_options,
    StepWrapper,
    event_handler=my_event,
    description=Simul_description,
    basefolder=basefolder,
)
my_simul.setAdditionalStoppingCriteria(
    lambda sol: sol.Nyielded > int(0.99 * mesh.nelts)
)


# %%
my_simul.setupSimulation(
    sol0,
    tend,
    dt=dt_ini,
    maxSteps=maxSteps,
    saveEveryNSteps=10,
    log_level="INFO",
)
my_simul.ts.save_to_file(0)
my_simul.saveConfigToBinary()
my_simul.saveParametersToJSon()
mesh.saveToJson(basefolder + "Mesh.json")

# %%
import logging
logging.info("Simulation folder created at : " + basefolder)
logging.info("Geometry and Mesh: %d / %d" % (Rinf, Nelts))
logging.info("dpc/sigop = %e" % (pcbysigop))
logging.info("taubyfsigop = %.2f" % (taubyfsigop))
logging.info("cf sigop = %e" % (cf * sigop))
logging.info("skempton coeff = %e" % (1/(1 + kn * cf * wo)))


# %% Run Simulation
import time
import shutil

# save run script in folder
shutil.copy2(__file__, basefolder)

zt = time.process_time()
# res, status_ts = my_simul.run(custom_printing)
res, status_ts = my_simul.run()
elapsed = time.process_time() - zt
print("elapsed time", elapsed)

# %%
