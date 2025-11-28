# %% SCRIPT FOR THE SIMULATION OF THE BASEL-1 well Stimulation
import time
import json
from datetime import datetime
import numpy as np
import scipy.special as sc


import os, sys

home = os.environ["HOME"]
dir_path = os.path.dirname(os.path.realpath(__file__))
#
# sys.path.append("/home/fakhretd/BigWham/build/interfaces/python/")
# sys.path.append("/home/fakhretd/PyFracX/src/")
# sys.path.append(dir_path)

# %%  Import required modules
# import qtconsole.styles

from pyfracx.mesh.usmesh import usmesh
from pyfracx.utils.App import TimeIntegrationApp
from pyfracx.hm.HMFsolver import HMFSolution, hmf_coupled_step
from pyfracx.mechanics.H_Elasticity import Elasticity
from pyfracx.mechanics.mech_utils import MechanicalModel, QuasiDynamicsOperator
from pyfracx.mechanics.MixModeW2D import MixModeW2D, normalStiffness_BB
from pyfracx.mechanics.evolutionLaws import linearEvolution
from pyfracx.flow.flow_utils import *
from pyfracx.flow.FlowConstitutiveLaws import ShearZoneFlowCubicLawNewtonian
from pyfracx.loads.Injection import *
from pyfracx.utils.helper_utils import uniform_interface_properties_BB
from pyfracx.utils.options_utils import (
    TimeIntegration_options,
    NonLinear_step_options,
    NonLinearSolve_options,
    IterativeLinearSolve_options,
)
from pyfracx.utils.json_dict_dataclass_utils import *
from pyfracx.ts.Event import EnforceTimes

# %%

Simul_description = "Basel 1 simulation Coupled - Axisymmetric - Variable rate - Variable Perm - BartonBandis stiffness"
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
basename = "Axisymmetric-Basel-NonLinearStiffness"
res_dir = os.path.join(os.path.dirname(__file__), "res_data")
os.makedirs(res_dir, exist_ok=True)
basefolder = os.path.join(res_dir, f"{basename}_{dt_string}")


# %% Simple 1D mesh
r_w = 0.0
Rend = 600.0
h_x = 2.0  # 0.25
Nelts = int(Rend / h_x)
coor1D = np.linspace(r_w, Rend, Nelts + 1)
coor = np.transpose(np.array([coor1D, coor1D * 0.0]))
conn = np.fromfunction(lambda i, j: i + j, (Nelts, 2), dtype=int)
me = usmesh(2, coor, conn, 1)

# %%  Model and parameters.

kernel = "Axi3DS0-H"  # axisymmetric simulation
# Elastic properties of the bulk matrix
YoungM = 40.0e9  # 65e9   # [Pa] Young's modulus
nu = 0.0  # [-] Poisson's ratio
shear_g = YoungM / (2 * (1 + nu))
Kk = 2 / 3 * shear_g + 2 * shear_g * nu / (1 - 2 * nu)
c_s = np.sqrt(shear_g / 2700.0)  # shear wave speed
c_p = c_s * np.sqrt(2 * (1 - nu) / (1.0 - 2 * nu))  # P -wave speed
eta_s = 0.5 * shear_g / c_s  # Quasi-dynamic s coef
eta_p = 0.5 * shear_g / c_p  # Quasi-dynamic p coef

# in-situ effective stress  #
# # uniform initial effective tractions in Pa
tauo = 22.0e6  # [Pa]
sigmap_o = 37.5e6  # [Pa]
in_situ_effective_tractions = np.full((Nelts, 2), [-tauo, -sigmap_o])
# initial displacement discontinuities
DDs_initial = 0.0 * in_situ_effective_tractions

## Properties of the pre-existing fracture/fault #
# Assume that matrix is impermeable and all the fluid went into one fracture.

# hydraulic properties from Pre-Stimulation test - cubic law model #
Tav = 2.0e-15  # average transmibillity w^3/12    [m^3]
# S_e_oedo = 2.e-8   # VALUE THAT WORKED REALLY WELL! Compressibility of fracture together with the shear zone    [Pa^{-1}]
S_e_oedo = (
    8.0e-9  # Compressibility of fracture together with the shear zone    [Pa^{-1}]
)
# alpha_hydroSZ = 0.0023  # hydraulic diffusivity of the shear zone from pre-stim test
h_thickness = 1.0  # [m] shear zone thickness, just a guess!

fluid_visc = 2.5e-4  # [Pa*s] fluid dynamic viscosity (value from Andres)
# S_e_oedo = Tav / (fluid_visc * alpha_hydroSZ * h_thickness)
alpha_hydroSZ = Tav / (fluid_visc * S_e_oedo * h_thickness)
print("S_e_oedometric = : %g " % (S_e_oedo))
print("Shear zone hydraulic diffusivity is = : %g " % (alpha_hydroSZ))
perm = alpha_hydroSZ * S_e_oedo * fluid_visc
wh_o = 1.0  # [m] hydraulic width of the fault, here it is a shear model, so we do not need wh_o, but I keep it for postprocessing without changes

cond_hyd = Tav / fluid_visc  # [m^3/(Pa*s)]

C_wd = 6.5 * 1.0e-8  # [m^3/Pa] wellbore storage from Andres

#  interface mechanical properties  #
# Barton-Bandis stiffness, linear slip weakening interface
### stiffnesses -
ks = YoungM * 100

# %%
# v_m=0.00048544393247491896
# kni=1.28553e10
kni = 3.02693e10
v_m = 0.000439377
kn_at_o = normalStiffness_BB(-(sigmap_o), kni, v_m)  # at sigmap_o
kn_at_opg = normalStiffness_BB(-(0e6), kni, v_m)
wc = v_m / (1 + v_m * kni / (sigmap_o))  # mechanical aperture at which Tn'=0
print("kn(s_o)/YoungM : %g" % (kn_at_o / YoungM))
print("kn(T =0)/YoungM : %g" % (kn_at_opg / YoungM))
print("w(at T=0)/wh_o: %g " % (wc / wh_o))

#  friction properties
chi = 3.2  # approximate slope of the seismic radius
f_p = 0.66  # peak friction
f_r = (
    2
    * np.pi
    * chi**2
    * S_e_oedo
    * h_thickness
    * tauo
    / (2 * np.pi * chi**2 * S_e_oedo * h_thickness * sigmap_o - 1)
)  # residual friction
f_r = 0.64
print("Residual friction is : %g " % (f_r))

psi_p = 0.1  # peak dilatancy
d_c = 0.0025  # critical slip distances
c_p = 0.0  # zero cohesion

sig_c = 0.0
alpha = 0.0
beta = 0.0

# Important dimensionless numbers
S_number = tauo / (f_p * sigmap_o)
F_number = f_r / f_p
P_number = 10e6 / sigmap_o

dil_max = psi_p * d_c / 2.0  # max dilation

print("Stress criticality ratio : %g " % (S_number))
print("ratio between residual and peak friction: %g" % (F_number))
print("Max transmissibility increase : %g " % ((1 + (dil_max / wh_o)) ** 3))

# %% Properties map creation  & interface model
mat_id = np.zeros(me.nelts, dtype=int)

# flow in the SHEAR zone. We must set S*h and
stor_c = PropertyMap(mat_id, np.array([S_e_oedo]))
thickness = PropertyMap(
    np.zeros(me.nelts, dtype=int), np.array([h_thickness])
)  # shear zone thickness
kh_o = PropertyMap(
    np.zeros(me.nelts, dtype=int), np.array([Tav])
)  # [m^3]   (k_o h)   Initial hydraulic transmissibility

shear_flow_properties = {
    "Initial hydraulic transmissibility": kh_o,  # property map
    "Shear zone thickness": thickness,  # property map
    "Storage coefficient": stor_c,  # property map
    "Fluid viscosity": fluid_visc,
}  # float

shear_zone_flow = ShearZoneFlowCubicLawNewtonian(
    shear_flow_properties, Nelts
)  # instantiate the permeability model

# mechanics
mat_properties = uniform_interface_properties_BB(
    me.nelts, ks, kni, v_m, sig_c, c_p, f_p, f_r, psi_p, wc, d_c, alpha, beta
)
interface_model_BB_LW = MixModeW2D(
    mat_properties,
    me.nelts,
    linearEvolution,
    yield_atol=1.0e-5 * (f_p * sigmap_o - tauo),
    tol=1e-6,
)
# %% Elasticity model
elas_properties = np.array([YoungM, nu])
elastic_m = Elasticity(
    kernel,
    elas_properties,
    max_leaf_size=32,
    eta=3.0,
    eps_aca=1.0e-6,
    n_openMP_threads=16,
)
# Hmat creation
hmat = elastic_m.constructHmatrix(me)
# Testing matvec operation
zt = time.process_time()
for i in range(120):
    hmat @ np.ones(2 * Nelts)
elapsed = (time.process_time() - zt) / 120
print("number of active threads for MatVec", hmat.get_omp_threads())
print("elapsed time", elapsed)

# %% Flow model
#  Import injection rate history /local_dev/Basel
file = open(
    os.path.join(os.path.dirname(__file__), "Stimulation_Q_VS_time_Version_3.json")
)
QinjTimeData = np.array(json.load(file))
file.close()
list_times = QinjTimeData[::, 0]
# injection with a given variation of injection rate
the_inj = Injection(
    np.array([0.0, 0.0]), QinjTimeData, "Rate", volumetric_compressibility=C_wd
)
#
vol_dt = np.array(
    [
        the_inj.volume(list_times[i], list_times[i + 1])
        for i in range(len(list_times) - 1)
    ]
)
vol = np.add.accumulate(vol_dt)
vol = np.hstack(([0.0], vol))
Q_dt = [
    the_inj.volume(list_times[i], list_times[i + 1])
    / (list_times[i + 1] - list_times[i])
    for i in range(len(list_times) - 1)
]

# -----

myt_t = np.hstack((list_times[:6], list_times[13:]))

# myt_t = np.array([ 4*3600.*i for i in range(25)])

vol_dt = [the_inj.volume(myt_t[i], myt_t[i + 1]) for i in range(len(myt_t) - 1)]
volc = np.add.accumulate(vol_dt)
volc = np.hstack(([0.0], volc))
Q_dt = [
    the_inj.volume(myt_t[i], myt_t[i + 1]) / (myt_t[i + 1] - myt_t[i])
    for i in range(len(myt_t) - 1)
]

# use a smoother injection schedulte
newQinjTimeData = np.array([np.array(myt_t[:-1]), Q_dt])
the_inj = Injection(
    np.array([0.0, 0.0]), newQinjTimeData.T, "Rate", volumetric_compressibility=C_wd
)


# %% models

# flow model
scale_r_f = 1.0  # scale for flow residuals

flow_model = FlowModelFractureSegments_axis(
    me, shear_zone_flow, the_inj, scalingR=scale_r_f, scalingX=1.0
)
# mechanical model
qd = QuasiDynamicsOperator(Nelts, 2, eta_s, eta_p)
# creating the mechanical model: hmat, preconditioner, number of collocation points, constitutive model
scale_r_m = 1.0  # scale for mech residuals
mech_model = MechanicalModel(
    hmat,
    me.nelts,
    interface_model_BB_LW,
    precType="Jacobi",
    QDoperator=qd,
    scalingX=1.0,
    scalingR=scale_r_m,
)  # "Jacobi"

# %% reate the simulation object

#
model_config = {  # in this dict, we store object etc. (but not the hmat that is not storable for now)
    "Mesh": me,
    "Elasticity": elastic_m,
    "Flow properties": shear_flow_properties,
    "Injection": the_inj,
    "Friction model": interface_model_BB_LW,
    "Material properties": mat_properties,
}

model_parameters = {
    "Elasticity": {"Young": YoungM, "Nu": nu},
    "Flow": {
        "who": wh_o,
        "Viscosity": fluid_visc,
        "Compressibility of the shear zone": S_e_oedo,
        "rock diffusivity": alpha_hydroSZ,
        "Transmissibility": Tav,
    },
    "Injection history": {"Constant Rate": QinjTimeData, "Wellbore storage": C_wd},
    "Initial stress": [sigmap_o, tauo],
    "Friction coefficient": {
        "peak": f_p,
        "residual": f_r,
        "d_c": d_c,
        "peak dilatancy": psi_p,
        "form": "linear",
    },
    "Interface stiffness": {"ks": ks, "kni": kni, "vm": v_m},
}

# %% initial solution


sol0 = HMFSolution(
    time=0.0,
    effective_tractions=in_situ_effective_tractions.flatten(),
    pressure=0.0 * np.zeros(me.nnodes, dtype=float),
    DDs=0.0 * in_situ_effective_tractions.flatten(),
    DDs_plastic=0.0 * in_situ_effective_tractions.flatten(),
    Internal_variables=np.zeros(2 * me.nelts),
    pressure_rate=0.0 * np.zeros(me.nnodes, dtype=float),
    DDs_rate=0.0 * in_situ_effective_tractions.flatten(),
    res_mech=0.0 * in_situ_effective_tractions.flatten(),
    res_flow=0.0 * np.zeros(me.nnodes, dtype=float),
)
# %% restart case
# /home/fakhretd/PyFracX/Basel_scripts/Scripts/res_data/Axisymmetric-Basel-NonLinearStiffness _04_03_2024___23_19_42
restart = False
if restart:
    basename = "Axisymmetric-Basel-NonLinearStiffness "
    basefolder = "./res_data/" + basename + "_04_03_2024___23_19_42/"

    data_sol = json_read(basefolder + "Axisymmetric-Basel-NonLinearStiffness -23375")
    s0 = 23375
    # switch to numpy array
    for k, v in data_sol.items():
        # print(k,"->",v)
        if type(v) == list:
            data_sol[k] = np.array(v)
    sol0 = from_dict_to_dataclass(HMFSolution, data_sol)
else:
    s0 = 0

# %% Solver  options
# newton solve options
# newton solve options
res_atol = (
    1e-4 * scale_r_m * max(np.linalg.norm(in_situ_effective_tractions.flatten()), 1e3)
)
print("res_atol: %g" % (res_atol))

newton_solver_options = NonLinearSolve_options(
    max_iterations=15,
    residuals_atol=res_atol,
    residuals_rtol=np.inf,
    dx_atol=np.inf,
    dx_rtol=1e-3,
    line_search=True,
    line_search_type="cheap",
    verbose=True,
)

# options for the jacobian solver
jac_solve_options = IterativeLinearSolve_options(
    max_iterations=300,
    restart_iterations=150,
    absolute_tolerance=0.0,
    relative_tolerance=1e-5,
    preconditioner_side="Left",
    schur_ilu_fill_factor=50,
    schur_ilu_drop_tol=1e-3,
    mech_rtol=1e-5,
    mech_atol=0.0,
    mech_max_iterations=int(me.nelts / 2),
)
# combining the 2 as option for the non-linear time-step
step_solve_options = NonLinear_step_options(
    jacobian_solver_type="BICGSTAB",
    jacobian_solver_opts=jac_solve_options,
    non_linear_start_factor=0.0,
    non_linear_solver_opts=newton_solver_options,
)


# %% Preparing the time-stepper
def onewayStepWrapper(solF, dt):
    solFNew = hmf_coupled_step(solF, dt, mech_model, flow_model, step_solve_options)
    return solFNew


# - initial time step from pure diffusion
dt_ini = (
    1.0 * h_x
) ** 2 / alpha_hydroSZ  # setting the initial time-step to have something "moving"
# dt_ini=1.
tend = 5.4e5
maxSteps = 2000  # 500000000
min_time_step = 6.0

# options of the time inegration !  note that we also pass the step_solve_options for consistency
ts_options = TimeIntegration_options(
    max_attempts=10,
    dt_reduction_factor=2.0,
    max_dt_increase_factor=1.035,
    lte_goal=0.001,
    acceptance_a_tol=1.0 * res_atol,
    minimum_dt=min_time_step,
    maximum_dt=600.0,
    stepper_opts=step_solve_options,
)

# we try here a simulation without enforcing events
# events=EnforceTimes(list_times)
# events.setTimeStepAfterEvent(4.*min_time_step) # do a small time step after an event
events = None

my_simul = TimeIntegrationApp(
    basename,
    model_config,
    model_parameters,
    ts_options,
    onewayStepWrapper,
    events,
    description=Simul_description,
    basefolder=basefolder,
)
my_simul.setupSimulation(
    sol0,
    tend,
    dt=dt_ini,
    maxSteps=maxSteps,
    saveEveryNSteps=5,
    log_level="INFO",
    start_step_number=s0,
)  #
if not (restart):
    my_simul.saveParametersToJSon()
    me.saveToJson(os.path.join(basefolder, "Mesh.json"))

# %% now we are ready to run the simulation
zt = time.process_time()
res, status_ts = my_simul.run()
elapsed = time.process_time() - zt
print("elapsed time", elapsed)
