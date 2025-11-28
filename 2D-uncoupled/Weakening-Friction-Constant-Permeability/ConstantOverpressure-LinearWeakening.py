#
# This file is part of PyFracX-Examples
#

#
# UnCoupled fluid injection into a planar frictional fault (plane-strain problem) due to a constant over-pressure
# Linear weakening friction reference results from Germanovich & Garagash (2012)
# parameters taken from Ciardo et al IJNME 2021 - do not change if you want to compare with the ref solution
#

# %% imports
import sys
from scipy import special
import time, sys
from datetime import datetime

from pyfracx.mesh.usmesh import usmesh
from pyfracx.mechanics.H_Elasticity import *
from pyfracx.MaterialProperties import PropertyMap
from pyfracx.mechanics.mech_utils import *
from pyfracx.mechanics.friction2D import *
from pyfracx.mechanics.evolutionLaws import *
from pyfracx.hm.HMFsolver import HMFSolution, hmf_one_way_step
from pyfracx.ts.TimeStepper import *
from pyfracx.utils.App import TimeIntegrationApp
from pyfracx.utils.options_utils import (
    TimeIntegration_options,
    NonLinear_step_options,
    NonLinearSolve_options,
    IterativeLinearSolve_options,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from ReferenceSolutions.FDFR.Plane_TwoD_frictional_ruptures import *

# %%

Simul_description = (
    "2D plane-strain - ct pressure - linear weak. friction - one way simulation"
)
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
basename = "2D-ctP-LinearWeakening-MarginallyPressurized"
res_dir = os.path.join(os.path.dirname(__file__), "res_data")
os.makedirs(res_dir, exist_ok=True)
basefolder = os.path.join(res_dir, f"{basename}_{dt_string}")


# %% Material parameters, in-sity stress & over-pressure
# material elasticity
YoungM = 1.0
nu = 0.0
shear_prime = YoungM / (2 * (1 + nu) * 2 * (1 - nu))
shear_g = YoungM / (2 * (1 + nu))
c_s = np.sqrt(shear_g / (1.0))
c_p = c_s * np.sqrt(2 * (1 - nu) / (1.0 - 2 * nu))
# fault friction
f_p = 1.0
psi_p = 0.0
f_r = 0.6 * f_p
d_c = 0.4
cohesion = 0.0
###
# initial effective tractions
sigmap_o = 1.0
tau_o = 0.55
# injection overpressure
dpcenter = 0.5
alpha_hyd = 10.0  # hydraulic diffusivity of the fault

# corresponding parameters dimensionless parameters
tau_o - f_p * (sigmap_o + dpcenter)
T = (1 - tau_o / (f_p * sigmap_o)) * sigmap_o / dpcenter
tau_o / (f_p * sigmap_o)
T_r = (1 - tau_o / (f_r * sigmap_o)) * sigmap_o / dpcenter

# %% mesh and model creation
# simple 1D mesh
Nelts = 2000
coor1D = np.linspace(-10.0, 10.0, Nelts + 1)
coor = np.transpose(np.array([coor1D, coor1D * 0.0]))
conn = np.fromfunction(lambda i, j: i + j, (Nelts, 2), dtype=int)
me = usmesh(2, coor, conn, 0)
colPts = (coor1D[1:] + coor1D[0:-1]) / 2.0  # collocation points for P0

# analytical solution for pressure at collocation points for a constant over-pressure at the center
pressure = lambda x, t, Dpcenter: Dpcenter * special.erfc(
    np.abs(x) / ((4.0 * alpha_hyd * t) ** 0.5)
)
pressureAtColPts = lambda t, Dpcenter: Dpcenter * special.erfc(
    np.abs(colPts) / ((4.0 * alpha_hyd * t) ** 0.5)
)

# Elasticity discretization via boundary element - plane-strain piece-wise constant displacement discontinuity element
kernel = "2DS0-H"
elas_properties = np.array([YoungM, nu])
elastic_m = Elasticity(kernel, elas_properties, max_leaf_size=80, eta=3, eps_aca=1.0e-4)
# BE hierarchical matrix creation
h1 = elastic_m.constructHmatrix(me)

#### Populate properties
# initial in-situ conditions  over the mesh
in_situ_tractions = np.full(
    (Nelts, 2), [-tau_o, -sigmap_o]
)  # positive stress in traction !

# setting frictional properties for the linear weakening elasto-plastic interface model
friction_p = PropertyMap(
    np.zeros(Nelts, dtype=int), np.array([f_p])
)  # friction coefficient
friction_r = PropertyMap(
    np.zeros(Nelts, dtype=int), np.array([f_r])
)  # friction coefficient
dilatant_p = PropertyMap(
    np.zeros(Nelts, dtype=int), np.array([psi_p])
)  # dilatancy coefficient
cohesion_p = PropertyMap(np.zeros(Nelts, dtype=int), np.array([0.0]))
slip_dc = PropertyMap(np.zeros(Nelts, dtype=int), np.array([d_c]))
k_sn = PropertyMap(
    np.zeros(Nelts, dtype=int),
    np.array([[100 * YoungM / (2 * (1 + nu)), 100 * YoungM]]),
)  # springs shear, normal

mat_properties = {
    "Peak friction": friction_p,
    "Residual friction": friction_r,
    "Peak dilatancy coefficient": dilatant_p,
    "Peak cohesion": cohesion_p,
    "Critical slip distance": slip_dc,
    "Spring Cij": k_sn,
    "Elastic parameters": {"Young": YoungM, "Poisson": nu},
}

frictionModel_LW = FrictionVar2D(
    mat_properties, Nelts, linearEvolution, tol=1.0e-6, yield_atol=1e-6 * sigmap_o
)  # linear frictional weakening model

# quasi-dynamics term (put here to a small value)
eta_s = 1.0e-3 * 0.5 * shear_g / c_s
eta_p = 1.0e-3 * 0.5 * shear_g / c_p
qd = QuasiDynamicsOperator(Nelts, 2, eta_s, eta_p)

# creating the mechanical model: hmat, preconditioner, number of collocation points, constitutive model
mech = MechanicalModel(h1, me.nelts, frictionModel_LW, precType="Jacobi", QDoperator=qd)

# initial solution
sol0 = HMFSolution(
    time=0.0,
    effective_tractions=in_situ_tractions.flatten(),
    pressure=np.zeros(me.nnodes),
    DDs=0.0 * in_situ_tractions.flatten(),
    DDs_plastic=0.0 * in_situ_tractions.flatten(),
    pressure_rate=np.zeros(me.nnodes),
    DDs_rate=0.0 * in_situ_tractions.flatten(),
    Internal_variables=np.zeros(2 * me.nelts),
    res_mech=0.0 * in_situ_tractions.flatten(),
)

# stepper options
# newton solve options
res_a_tol = 1.0e-5 * max(
    sigmap_o * me.nelts, 1e3
)  # absolute convergence tolerance on mechanical residualts.
newton_solver_options = NonLinearSolve_options(
    max_iterations=20,
    residuals_atol=res_a_tol,
    residuals_rtol=np.inf,
    dx_atol=np.inf,
    dx_rtol=1e-3,
    line_search=True,
    line_search_type="cheap",
)
# options for the jacobian solver
jac_solve_options = IterativeLinearSolve_options(
    max_iterations=200,
    absolute_tolerance=1.0e-12,
    relative_tolerance=1.0e-6,
    restart_iterations=150,
)
# combining the 2 as option for the non-linear time-step
step_solve_options = NonLinear_step_options(
    jacobian_solver_type="GMRES",
    jacobian_solver_opts=jac_solve_options,
    non_linear_start_factor=0.0,
    non_linear_solver_opts=newton_solver_options,
)

## function wrapping the one-way / uncoupled solver for this case
Dtinf = np.zeros(2 * me.nelts)


def onewayStepWrapper(solN, dt):
    time_s = solN.time + dt
    # compute pressure increment at nodes form the analytical solution
    dp = pressure(coor1D, time_s, dpcenter) - solN.pressure
    # at collocation points
    if solN.time == 0.0:
        dpcol = pressureAtColPts(time_s, dpcenter)
    else:
        dpcol = pressureAtColPts(time_s, dpcenter) - pressureAtColPts(
            solN.time, dpcenter
        )
    solTnew = hmf_one_way_step(solN, dt, mech, dpcol, Dtinf, step_solve_options)
    solTnew.pressure = solTnew.pressure + dp
    return solTnew


#  Preparing folders and base dict for savings

model_config = {  # in this dict, we store object etc. (but not the hmat that is not storable for now)
    "Mesh": me,
    "Elasticity": elastic_m,
    "Friction model": frictionModel_LW,
    "Material properties": mat_properties,
}
model_parameters = {
    "Elasticity": {"Young": YoungM, "Nu": nu},
    "Flow": {"Hydraulic diffusivity": alpha_hyd},
    "Injection": {"P center": dpcenter},
    "Initial stress": [sigmap_o, tau_o],
    "Friction peak": f_p,
    "Friction residual": f_r,
    "Dilatancy peak": psi_p,
    "Dc": d_c,
    "T parameter": T,
}

# prepare the time-stepper simulations
new_dt = 1.0e-4  # initial time-step
maxSteps = 160  # 120 max number of stpes of the simulation
tend = 1e6  # max time to simulate

# options of the time inegration !  note that we also pass the step_solve_options
ts_options = TimeIntegration_options(
    max_attempts=4,
    dt_reduction_factor=2.0,
    max_dt_increase_factor=1.25,
    lte_goal=0.005,
    acceptance_a_tol=res_a_tol,
    minimum_dt=new_dt / 100.0,
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
my_simul.setAdditionalStoppingCriteria(lambda sol: sol.Nyielded == me.nelts)

my_simul.setupSimulation(sol0, tend, dt=new_dt, maxSteps=maxSteps, saveEveryNSteps=1)
my_simul.saveParametersToJSon()
# my_simul.saveConfigToBinary()
me.saveToJson(os.path.join(basefolder, "Mesh.json"))
# if (os.path.exists(simul_definition["Simulation folder"])==False):
#     os.mkdir(simul_definition["Simulation folder"])


# %% now we are ready to run the simulation
zt = time.process_time()
res, status_ts = my_simul.run()
elapsed = time.process_time() - zt
print("elapsed time", elapsed)
