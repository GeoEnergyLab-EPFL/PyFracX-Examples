#
# This file is part of PyFracX-Examples
#
#
# UnCoupled fluid injection at constant rate into a frictional fault in 3D (modelled as axisymmetric problem).
# Reference results from Sáez & Lecampion (2022)

# %%+
# Importing the necessary python libraries and managing the python path

import os
import sys
import numpy as np
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy.special import exp1
import time
import scipy.special
from scipy import linalg

from pyfracx.mesh.usmesh import UnstructuredMesh
from pyfracx.mechanics.H_Elasticity import Elasticity
from pyfracx.MaterialProperties import PropertyMap
from pyfracx.mechanics.friction2D import FrictionCt2D
from pyfracx.mechanics.mech_utils import MechanicalModel
from pyfracx.flow.FlowConstitutiveLaws import ConstantPerm
from pyfracx.flow.flow_utils import FlowModelFractureSegments_axis
from pyfracx.loads.Injection import Injection
from pyfracx.hm.HMFsolver import HMFSolution
from pyfracx.utils.options_utils import (
    NonLinearSolve_options,
    IterativeLinearSolve_options,
    NonLinear_step_options,
)
from pyfracx.hm.HMFsolver import hmf_one_way_flow_numerical, HMFSolution
from pyfracx.utils.App import TimeIntegrationApp
from pyfracx.utils.options_utils import TimeIntegration_options

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from ReferenceSolutions.FDFR.frictional_ruptures_3D_constant_friction import *

# %%

Simul_description = "Axi Symm - ct rate - ct friction - one way simulation - flow numerical - crit stress"
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
basename = "AxiSymm-ctRate-ctFriction-oneWayFlowNumerical-critstress"
res_dir = os.path.join(os.path.dirname(__file__), "res_data")
os.makedirs(res_dir, exist_ok=True)
basefolder = os.path.join(res_dir, f"{basename}_{dt_string}")


# %%

# Defining the stress injection parameter value for criticallty stressed simulation

T_exp = 0.01

# Analytical solution for amplification factor lambda = R(t)/L(t), Eq. 21 from Sáez & Lecampion (2022)

lam = lambda_analytical(T_exp)
print("Amplification factor expected, lambda = ", lam)

# Plotting the analytical solution for amplification factor with a range of stress injection parameter,
# Fig. 2 from Sáez & Lecampion (2022)

T = np.linspace(5e-4, 1e1, 100)


# %% Decide simulation parameters
# Here, we calculate the expected rupture length and decide the mesh size and number of elements

t_first_step = 100  # [s], first time step
alpha = 0.1  #  [m^2/s], rock hyd diffusivity
t_end = 1e4  # [s], end time of simulation

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


coor1D = np.linspace(0.0, domain_size, Nelts + 1)
coor = np.transpose(np.array([coor1D, coor1D * 0.0]))
conn = np.fromfunction(lambda i, j: i + j, (Nelts, 2), dtype=int)
mesh = UnstructuredMesh(2, coor, conn, 0)

Nelts = mesh.nelts
Nnodes = mesh.nnodes
coor = np.asarray(mesh.coor)
conn = np.asarray(mesh.conn)
colPts = (coor1D[1:] + coor1D[0:-1]) / 2.0  # collocation points for P0

# radial coordinates of the nodes
r = np.array([linalg.norm(coor[i]) for i in range(Nnodes)])

# radial coordinates of the collocation points
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

# Calculating the initial shear stress corresponding to the stress injection parameter T
tau0 = f_p * (sigma0 - p0) - f_p * T_exp * dp_star
T = (f_p * (sigma0 - p0) - tau0) / (f_p * dp_star)
print("Stress injection parameter, T = ", T)
print("Shear stress, tau0 = ", tau0)

# %%

# analytical solution for pressure at collocation points for constant rate injection
# Eq. 4 in Sáez & Lecampion (2022)
pressure = lambda r, t: (p0 + dp_star * exp1((r**2) / (4.0 * alpha * t)))
p_col = pressure(r_col, t_end) / p0

# %%
# Elasticity discretization via boundary element - Axisymmetric piece-wise constant displacement discontinuity element

kernel = "Axi3DS0-H"
elas_properties = np.array([E, nu])
elastic_m = Elasticity(
    kernel, elas_properties, max_leaf_size=128, eta=3.0, eps_aca=1.0e-3
)

# BE H-Matrix construction for the elastic problem
hmat = elastic_m.constructHmatrix(mesh)
# %%
#### Populate material properties


friction_c = PropertyMap(
    np.zeros(Nelts, dtype=int), np.array([f_p])
)  # friction coefficient

dilatant_c = PropertyMap(
    np.zeros(Nelts, dtype=int), np.array([f_d])
)  # dilatancy coefficient

k_sn = PropertyMap(
    np.zeros(Nelts, dtype=int),
    np.array([[10 * G, 10 * G]]),
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

# initial pore pressure over the nodes and the collocation points
po_nodes = np.zeros(mesh.nnodes, dtype=float) + p0
po_col = np.zeros(mesh.nelts, dtype=float) + p0

# initial in-situ tractions over the mesh
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


# newton solve options
res_atol = 1.0e-3 * max(np.linalg.norm(in_situ_tractions.flatten()), 1e3)
print("res_atol: %g" % (res_atol))

newton_solver_options = NonLinearSolve_options(
    max_iterations=25,
    residuals_atol=res_atol,
    residuals_rtol=np.inf,
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

Dtinf = np.zeros(2 * mesh.nelts)


def onewayStepWrapper(solN: HMFSolution, dt: float) -> HMFSolution:
    solTnew = hmf_one_way_flow_numerical(
        solN, dt, mech_model, Dtinf, flow_model, step_solve_options
    )
    return solTnew


# %%
# Storing the configuration and properties of the model
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


# Path(basefolder).mkdir(parents=True, exist_ok=True)

# prepare the time-stepper simulations
new_dt = t_first_step  # initial time-step
maxSteps = 100  # 120 max number of stpes of the simulation
tend = t_end  # max time to simulate

# options of the time inegration !  note that we also pass the step_solve_options
ts_options = TimeIntegration_options(
    max_attempts=10,
    dt_reduction_factor=1,
    max_dt_increase_factor=1.3,
    minimum_dt=new_dt,
    acceptance_a_tol=res_atol,
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

# my_simul.setAdditionalStoppingCriteria(lambda sol: sol.Nyielded == mesh.nelts // 2)

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
mesh.saveToJson(os.path.join(basefolder, "Mesh.json"))


# %% Simulation
# now we are ready to run the simulation

zt = time.process_time()
res, status_ts = my_simul.run()
elapsed = time.process_time() - zt
print("End of simulation in ", elapsed)
