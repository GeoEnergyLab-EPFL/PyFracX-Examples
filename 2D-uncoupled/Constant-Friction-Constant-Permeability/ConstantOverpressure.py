#
# This file is part of PyFracX-Examples
#

#
# UnCoupled fluid injection into a 2D planar frictional fault (plane-strain problem) due to a constant over-pressure
# Constant friction reference results from Viesca (2021)
# All properties are constant

# %% imports
import sys
import numpy as np
from scipy import special
import time
from datetime import datetime


from pyfracx.mesh.usmesh import UnstructuredMesh
from pyfracx.mechanics.H_Elasticity import *
from pyfracx.MaterialProperties import PropertyMap
from pyfracx.mechanics.friction2D import *
from pyfracx.mechanics.mech_utils import *
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

Simul_description = "2D plane-strain - ct pressure - ct friction - one way simulation"
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
basename = "2D-ctP-ctFriction"
res_dir = os.path.join(os.path.dirname(__file__), "res_data")
os.makedirs(res_dir, exist_ok=True)
basefolder = os.path.join(res_dir, f"{basename}_{dt_string}")


# %% Material parameters, in-sity stress & over-pressure
# material elasticity
YoungM = 48.0e9
nu = 0.2
shear_prime = YoungM / (2 * (1 + nu) * 2 * (1 - nu))

# fault friction
f_p = 0.6
# dilatancy
psi_p = 0
# initial effective tractions
sigmap_o = 90.0e6
tau_o = 50.0e6
# injection overpressure
T = 0.1
dpcenter = (1.0 - tau_o / (f_p * sigmap_o)) * sigmap_o / T
alpha_hyd = 10.0e-2

# corresponding parameters dimensionless parameters
# T=(1-tau_o/(f_p*sigmap_o))*sigmap_o/dpcenter
print("T=", T)

# %% mesh and model creation
# simple 1D mesh
Nelts = 4000
coor1D = np.linspace(-10.0, 10.0, Nelts + 1)
coor = np.transpose(np.array([coor1D, coor1D * 0.0]))
conn = np.fromfunction(lambda i, j: i + j, (Nelts, 2), dtype=int)
colPts = (coor1D[1:] + coor1D[0:-1]) / 2.0  # collocation points for P0
# BE hierarchical matrix creation
me = UnstructuredMesh(2, coor, conn, 0)

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
elastic_m = Elasticity(
    kernel,
    elas_properties,
    max_leaf_size=16,
    eta=10.0,
    eps_aca=1.0e-6,
    n_openMP_threads=16,
)
# BE hierarchical matrix creation
h = elastic_m.constructHmatrix(me)

#### Populate properties
# initial in-situ conditions  over the mesh
in_situ_tractions = np.full(
    (Nelts, 2), [-tau_o, -sigmap_o]
)  # positive stress in traction !

# setting frictional properties for the constant friction elasto-plastic interface model
friction_c = PropertyMap(
    np.zeros(Nelts, dtype=int), np.array([f_p])
)  # friction coefficient
dilatant_c = PropertyMap(
    np.zeros(Nelts, dtype=int), np.array([psi_p])
)  # dilatancy coefficient
k_sn = PropertyMap(
    np.zeros(Nelts, dtype=int),
    np.array([[100 * YoungM / (2 * (1 + nu)), 100 * YoungM]]),
)  # springs shear, normal

mat_properties = {
    "Friction coefficient": friction_c,
    "Dilatancy coefficient": dilatant_c,
    "Spring Cij": k_sn,
    "Elastic parameters": {"Young": YoungM, "Poisson": nu},
}

frictionModel = FrictionCt2D(mat_properties, Nelts, yield_atol=1.0e-6 * sigmap_o)

# creating the mechanical model: hmat, preconditioner, number of collocation points, constitutive model
mech = MechanicalModel(h, me.nelts, frictionModel, precType="ILUT")  # "Jacobi+")
mech.always_update_pc_ = True

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
    res_flow=np.zeros(me.nnodes),
)

# stepper options
# newton solve options
res_a_tol = 1.0e-3 * max(
    sigmap_o * me.nelts, 1e3
)  # absolute convergence tolerance on mechanical residualts.
newton_solver_options = NonLinearSolve_options(
    max_iterations=20,
    residuals_atol=res_a_tol,
    residuals_rtol=np.inf,
    dx_atol=np.inf,
    dx_rtol=1e-4,
    line_search=True,
    line_search_type="cheap",
)
# options for the jacobian solver
jac_solve_options = IterativeLinearSolve_options(
    max_iterations=200,
    absolute_tolerance=1e-6,
    relative_tolerance=1e-6,
    restart_iterations=150,
)
# combining the 2 as option for the non-linear time-step
step_solve_options = NonLinear_step_options(
    jacobian_solver_type="Bicgstab",
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
    "Friction model": frictionModel,
    "Material properties": mat_properties,
}
model_parameters = {
    "Elasticity": {"Young": YoungM, "Nu": nu},
    "Flow": {"Hydraulic diffusivity": alpha_hyd},
    "Injection": {"P center": dpcenter},
    "Initial stress": [sigmap_o, tau_o],
    "Friction": f_p,
    "Dilatancy": psi_p,
    "T parameter": T,
}

# prepare the time-stepper simulations
new_dt = 1e-1  # initial time-step
maxSteps = 90  # 120 max number of stpes of the simulation
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
me.saveToJson(os.path.join(basefolder, "Mesh.json"))

# %% now we are ready to run the simulation
zt = time.time()  # time.process_time()
res, status_ts = my_simul.run()
elapsed = time.time() - zt
print("elapsed time", elapsed)
