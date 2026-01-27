# %% This file is part of PyFracX.
#
# Created by Brice Lecampion on 08.01.25.
# Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2025.  All rights reserved.
# See the LICENSE.TXT file for more details.
#
#
# ct friction case

# %% General Imports
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib
import gmsh
import scipy.special as sc
import time
from scipy.special import exp1
import mpmath


#  Imports from PyFracX
from pyfracx.mesh.usmesh import UnstructuredMesh
from pyfracx.mesh.mesh_utils import *
from pyfracx.MaterialProperties import PropertyMap

# from fe.assemble import assemble,assembleLoadFun
from pyfracx.flow.FlowConstitutiveLaws import *
from pyfracx.flow.flow_utils import FlowModelFractureSurfaces
from pyfracx.loads.Injection import *
from pyfracx.mechanics.mech_utils import *
from pyfracx.mechanics.H_Elasticity import *
from pyfracx.mechanics.friction3D import FrictionCt3D
from pyfracx.hm.HMFsolver import HMFSolution, hmf_coupled_step
from pyfracx.ts.TimeStepper import *
from pyfracx.utils.App import TimeIntegrationApp
from pyfracx.mechanics import H_Elasticity
from pyfracx.utils.options_utils import (
    TimeIntegration_options,
    NonLinear_step_options,
    NonLinearSolve_options,
    IterativeLinearSolve_options,
)

# %%

Simul_description = (
    "3D - ct injection rate - ct friction - cubic law model - coupled simulation"
)
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
basename = "3D-ctFriction-CubicLaw-benchmark_hexFracture"
res_dir = os.path.join(os.path.dirname(__file__), "res_data")
os.makedirs(res_dir, exist_ok=True)
basefolder = os.path.join(res_dir, f"{basename}_{dt_string}")


# %%
# analytical solution for pressure
def pres(r, t, c=1.0):  # divided by dp_c
    return (-1.0 / (4 * np.pi)) * sc.expi(-r * r / (4 * c * t))


# %%   Mesh a polygon centered on 0,0
center_res = 0.005
out_res = 0.08
Lx_ext = 1.0

p = 6
x = [
    ([Lx_ext * np.cos(2 * np.pi * i / p), Lx_ext * np.sin(2 * np.pi * i / p), 0.0])
    for i in range(p)
]

gmsh.initialize()
for i in range(p):
    gmsh.model.geo.addPoint(x[i][0], x[i][1], x[i][2], out_res, i + 1)

for i in range(p - 1):
    gmsh.model.geo.addLine(i + 1, i + 2, i + 1)

gmsh.model.geo.addLine(p, 1, p)

ll = [i + 1 for i in range(p)]
gmsh.model.geo.addCurveLoop(ll, 1)
gmsh.model.geo.addPlaneSurface([1], 1)


# We define a new point for the origin to enforce the mesh
gmsh.model.geo.addPoint(0.0, 0.0, 0.0, center_res, p + 1)

gmsh.model.geo.synchronize()
gmsh.model.mesh.embed(0, [p + 1], 2, 1)

gmsh.model.mesh.generate(5)

## post-processing the gmsh generated triangulation
dim = -1
tag = -1
nodeTags, coords, parametricCoord = gmsh.model.mesh.getNodes(dim, tag)
coords = coords.reshape((-1, 3))

defined_elt_type = gmsh.model.mesh.getElementTypes()

eletype = 2  # triangle
tag = -1
eleTags, nodeTags = gmsh.model.mesh.getElementsByType(eletype, tag)
nodeTags = nodeTags.reshape((-1, 3)) - 1
# gmsh.fltk.run()
mesh = UnstructuredMesh(coor=coords, conn=nodeTags, dimension=3)
gmsh.finalize()

Nelts = mesh.nelts


# %%
# injection rate [m3/s]
Qinj = 1.8 / 60
# hydraulic properties - cubic law

fluid_visc = 8.9e-4  # fluid viscosity
wh_o = 3.3e-4  # initial hydraulic width
alpha_h = 0.01
hyd_cond = wh_o**2 / (12.0 * fluid_visc)
S_e = hyd_cond / alpha_h  # storage [1/Pa]
dp_star = Qinj / ((4.0 * np.pi) * (hyd_cond * wh_o))

# elastic properties
G = 30e9
nu = 0.3
YoungM = 2 * G * (1 + nu)

# spring factor
beta_spring = 100  # kn, ks are beta_spring elastic stiffness

# ct friction
f_p = 0.6
f_dil = 0  # zero dilatancy

# ---- in-situ conditions
# Fault Parameters
T_expected = 0.01
sigmap_o = 80e6  # effective normal stress
p0 = 0  # background pore pressure
tau_xz = f_p * (sigmap_o) - f_p * T_expected * dp_star
tau_yz = 0
T = (f_p * (sigmap_o) - tau_xz) / (f_p * dp_star)
print("T = ", T)
print("tau_xz = ", tau_xz)
print("tau_yz = ", tau_yz)

# %%  Mechanical model & initial tractions
# Elasticity model

kernel = "3DT0-H"
elas_properties = np.array([YoungM, nu])
elastic_m = Elasticity(
    kernel, elas_properties, max_leaf_size=32, eta=3, eps_aca=1.0e-5, n_openMP_threads=8
)
# hmat creation
h1 = elastic_m.constructHmatrix(mesh)

# Preparing  properties for simulation
friction_c = PropertyMap(
    np.zeros(mesh.nelts, dtype=int), np.array([f_p])
)  # friction coefficient
dilatant_c = PropertyMap(
    np.zeros(mesh.nelts, dtype=int), np.array([f_dil])
)  # dilatancy coefficient
k_sn = PropertyMap(
    np.zeros(mesh.nelts, dtype=int),
    np.array([[beta_spring * YoungM / (2 * (1 + nu)), beta_spring * YoungM]]),
)  # springs shear, normal

mech_properties = {
    "Friction coefficient": friction_c,
    "Dilatancy coefficient": dilatant_c,
    "Spring Cij": k_sn,
    "Elastic parameters": {"Young": YoungM, "Poisson": nu},
}
# instantiating the constant friction model
frictionModel = FrictionCt3D(mech_properties, mesh.nelts)

# creating the mechanical model: hmat, preconditioner, number of collocation points, constitutive model
mech_model = MechanicalModel(h1, mesh.nelts, frictionModel, precType="Jacobi")

# In-situ traction and pore pressure field
# insitu tractions
insitu_tractions_global = np.full(
    (mesh.nelts, 3), [-tau_xz, 0.0, -sigmap_o]
)  # positive stress in traction ! tension positive convention
# flattened array
insitu_tractions_local = h1.convert_to_local(insitu_tractions_global.flatten())
# initial pore pressure array at nodes
pp_0 = np.zeros(mesh.nnodes, dtype=float) + p0

# %% Flow model

stor_c = PropertyMap(np.zeros(mesh.nelts, dtype=int), np.array([S_e]))
aperture = PropertyMap(np.zeros(mesh.nelts, dtype=int), np.array([wh_o]))  # aperture
flow_properties = {
    "Initial hydraulic width": aperture,
    "Compressibility": stor_c,
    "Fluid viscosity": fluid_visc,
}
cubicModel = CubicLawNewtonian(
    flow_properties, Nelts
)  # instantiate the permeability model
# mechanics

the_inj = Injection(np.array([0.0, 0.0, 0.0]), np.array([[0.0, Qinj]]), "Rate")

flow_model = FlowModelFractureSurfaces(
    mesh, cubicModel, the_inj, scalingR=1, scalingX=1
)


# %% initiial solution

sol0 = HMFSolution(
    time=0.0,
    effective_tractions=insitu_tractions_local.flatten(),
    pressure=0.0 * np.zeros(mesh.nnodes, dtype=float),
    DDs=0.0 * insitu_tractions_local.flatten(),
    DDs_plastic=0.0 * insitu_tractions_local.flatten(),
    Internal_variables=np.zeros(2 * mesh.nelts),
    pressure_rate=0.0 * np.zeros(mesh.nnodes, dtype=float),
    DDs_rate=0.0 * insitu_tractions_local.flatten(),
    res_mech=0.0 * insitu_tractions_local.flatten(),
    res_flow=0.0 * np.zeros(mesh.nnodes, dtype=float),
)


# %% stepper options
# newton solve options
res_atol = 1.0e-3 * max(np.linalg.norm(insitu_tractions_local.flatten()), 1e3)
print("res_atol: %g" % (res_atol))

newton_solver_options = NonLinearSolve_options(
    max_iterations=20,
    residuals_atol=res_atol,
    residuals_rtol=np.inf,
    dx_atol=np.inf,
    dx_rtol=1e-3,
    line_search=False,
    line_search_type="None",
    verbose=True,
)

# options for the jacobian solver
jac_solve_options = IterativeLinearSolve_options(
    max_iterations=300,
    restart_iterations=150,
    absolute_tolerance=0.0,
    relative_tolerance=1e-6,
    preconditioner_side="Left",
    schur_ilu_fill_factor=1,
    schur_ilu_drop_tol=1e-4,
    mech_rtol=1e-6,
    mech_atol=0.0,
    mech_max_iterations=mesh.nelts,
)

# combining the 2 as option for the non-linear time-step
step_solve_options = NonLinear_step_options(
    jacobian_solver_type="BICGSTAB",
    jacobian_solver_opts=jac_solve_options,
    non_linear_start_factor=0.0,
    non_linear_solver_opts=newton_solver_options,
)


#  Preparing the time-stepper
def StepWrapper(solF, dt):
    solFNew = hmf_coupled_step(solF, dt, mech_model, flow_model, step_solve_options)
    return solFNew


# - initial time step from pure diffusion
h_x = center_res  # resolution...

dt_ini = (
    1.0 * h_x
) ** 2 / alpha_h  # setting the initial time-step to have something "moving"
tend = 0.17 * 10
maxSteps = 100

# options of the time inegration !  note that we also pass the step_solve_options for consistency
ts_options = TimeIntegration_options(
    max_attempts=6,
    dt_reduction_factor=1.5,
    max_dt_increase_factor=1.3,
    lte_goal=0.001,
    acceptance_a_tol=res_atol,
    minimum_dt=dt_ini / 100.0,
    maximum_dt=dt_ini * 100,
    stepper_opts=step_solve_options,
)

# %%
# dd/mm/YY H:M:S


model_config = {  # in this dict, we store object etc. (but not the hmat that is not storable for now)
    "Mesh": mesh,
    "Elasticity": elastic_m,
    "Friction model": frictionModel,
    "Material properties": mech_properties | flow_properties,
}
model_parameters = {
    "Elasticity": {"Young": YoungM, "Nu": nu},
    "Injection": {"Injection rate": Qinj},
    "Flow": {
        "fracture diffusivity": alpha_h,
        "Hydraulic conductivity": hyd_cond,
        "Initial aperture": wh_o,
        "Fluid viscosity": fluid_visc,
        "Storage ": S_e,
    },
    "Initial stress": [tau_xz, tau_yz, sigmap_o],
    "Friction coefficient": f_p,
    "Dilatancy coefficient": f_dil,
    "T parameter": T,
}

my_simul = TimeIntegrationApp(
    basename,
    model_config,
    model_parameters,
    ts_options,
    StepWrapper,
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
    start_step_number=0,
)  #

my_simul.setAdditionalStoppingCriteria(lambda sol: sol.Nyielded == mesh.nelts)

restart = False
if not (restart):
    my_simul.saveParametersToJSon()
    mesh.saveToJson(os.path.join(basefolder, "Mesh.json"))

# %% now we are ready to run the simulation
zt = time.process_time()
res, status_ts = my_simul.run()
elapsed = time.process_time() - zt
print("elapsed time", elapsed)
