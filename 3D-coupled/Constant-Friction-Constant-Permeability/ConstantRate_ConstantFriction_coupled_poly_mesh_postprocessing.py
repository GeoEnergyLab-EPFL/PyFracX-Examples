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
import re
import scienceplots

plt.style.use(["science", "grid"])

#  Imports from PyFracX
from pyfracx.mesh.usmesh import UnstructuredMesh
from pyfracx.utils.json_dict_dataclass_utils import *
from pyfracx.hm.HMFsolver import HMFSolution
from pyfracx.loads.Injection import *
from pyfracx.mechanics.H_Elasticity import *


# %% loading numerical results
# here you need to change basefolder to the folder name of your simulation
# set it to None to look for the most recent simulation
basefolder = None
basename = "3D-ctFriction-CubicLaw-benchmark_hexFracture"

# Looking for the most recent simulation
if basefolder is None:
    res_dir = "res_data"

    # Pattern to match: basename_DD-MM-YYYY-HH-MM-SS
    pattern = re.compile(
        rf"^{re.escape(basename)}_(\d{{2}}-\d{{2}}-\d{{4}}-\d{{2}}-\d{{2}}-\d{{2}})$"
    )

    # Get all folders in res_dir matching the pattern
    matching_folders = []
    for folder_name in os.listdir(res_dir):
        folder_path = os.path.join(res_dir, folder_name)
        if os.path.isdir(folder_path):
            match = pattern.match(folder_name)
            if match:
                datetime_str = match.group(1)
                # Parse the datetime string: DD-MM-YYYY-HH-MM-SS
                try:
                    dt = datetime.strptime(datetime_str, "%d-%m-%Y-%H-%M-%S")
                    matching_folders.append((folder_name, dt))
                except ValueError:
                    continue

    # Check if we found any folders
    if not matching_folders:
        raise FileNotFoundError(
            f"No folder found matching pattern '{basename}-DD-MM-YYYY-HH-MM-SS' in '{res_dir}'"
        )

    # Sort by datetime (most recent last) and pick the most recent
    matching_folders.sort(key=lambda x: x[1])
    most_recent_folder = matching_folders[-1][0]

    basefolder = os.path.join(res_dir, most_recent_folder)

print(f"Using results folder: {basefolder}")

# %% Setup figure directory
figure_dir = f"figures_{basename}"
os.makedirs(figure_dir, exist_ok=True)

# we load all available timesteps by detecting files in the folder
res = []

# Get all files in the basefolder and extract step numbers
pattern = re.compile(rf"^{re.escape(basename)}-(\d+)\.json$")
step_numbers = []

for filename in os.listdir(basefolder):
    match = pattern.match(filename)
    if match:
        step_num = int(match.group(1))
        step_numbers.append(step_num)

# Convert to numpy array and sort
step_numbers = np.array(step_numbers)
step_numbers = np.sort(step_numbers)

print(f"Step range : {step_numbers[0]}-{step_numbers[-1]}")

# Loop through detected timesteps
for step in step_numbers:
    tt = from_dict_to_dataclass(
        HMFSolution, json_read(os.path.join(basefolder, f"{basename}-{step}"))
    )
    res.append(tt)

param = json_read(os.path.join(basefolder, "Parameters"))

mm = json_read(os.path.join(basefolder, "Mesh"))
mesh = UnstructuredMesh(2, np.array(mm["Coordinates"]), np.array(mm["Connectivity"]), 0)

Qinj = param["Injection"]["Injection rate"]
YoungM = param["Elasticity"]["Young"]
nu = param["Elasticity"]["Nu"]
wh_o = param["Flow"]["Initial aperture"]
fluid_visc = param["Flow"]["Fluid viscosity"]
hyd_cond = param["Flow"]["Hydraulic conductivity"]
c_f = param["Flow"]["Storage "]
f_p = param["Friction coefficient"]
T = param["T parameter"]

p0 = 0.0
dp_star = Qinj / ((4.0 * np.pi) * (hyd_cond * wh_o))

tend = res[-1].time

G = YoungM / (1 + nu) / 2

k_frac = wh_o**2 / 12
alpha_h = k_frac / (fluid_visc * c_f)

the_inj = Injection(np.array([0.0, 0.0]), np.array([[0.0, Qinj]]), "Rate")

triang = matplotlib.tri.Triangulation(
    mesh.coor[:, 0], mesh.coor[:, 1], triangles=mesh.conn, mask=None
)

# %% We rebuild the hmat to get the local - global conversions

kernel = "3DT0-H"
elas_properties = np.array([YoungM, nu])
elastic_m = Elasticity(
    kernel, elas_properties, max_leaf_size=32, eta=3, eps_aca=1.0e-5, n_openMP_threads=8
)
# hmat creation
h1 = elastic_m.constructHmatrix(mesh)

# %% time evolution of inlet pressure
inj_pressure = np.zeros(len(res))
timestamp = np.zeros(len(res))
i_inj = the_inj.locate_in_mesh(mesh)
for k in range(len(res)):
    inj_pressure[k] = res[k].pressure[i_inj]
    timestamp[k] = res[k].time

fig1, ax1 = plt.subplots()
plt.plot(timestamp, inj_pressure, label="Numerical")
plt.xlabel("Time (s)")
plt.ylabel("Injection pressure (Pa)")
plt.legend()
plt.savefig(
    os.path.join(figure_dir, "injection_pressure_vs_time.png"),
    dpi=200,
    bbox_inches="tight",
)
# plt.show()

# %%
solN = res[-1]

rcoor = (mesh.coor[:, 0] ** 2 + mesh.coor[:, 1] ** 2) ** (0.5)

fig1, ax1 = plt.subplots()
tri = ax1.tricontourf(triang, solN.pressure, cmap=plt.cm.rainbow, alpha=0.5)
ax1.axis("equal")
plt.colorbar(tri)
plt.title("Fluid pressure (Pa)")
plt.savefig(
    os.path.join(figure_dir, "pressure_contour.png"), dpi=200, bbox_inches="tight"
)
# plt.show()

fig, ax = plt.subplots()
ax.plot(rcoor, solN.pressure, ".", label="Numerical")
plt.xlabel("r (m)")
plt.ylabel("Pressure (Pa)")
plt.legend()
plt.savefig(
    os.path.join(figure_dir, "pressure_vs_radius.png"), dpi=200, bbox_inches="tight"
)
# plt.show()

# %% slip

global_dds = h1.convert_to_global(solN.DDs)

fig1, ax1 = plt.subplots()
tri = ax1.tripcolor(triang, global_dds[0::3], cmap=plt.cm.rainbow, alpha=0.5)
ax1.axis("equal")
plt.colorbar(tri)
plt.title("Slip along x (m)")
plt.savefig(
    os.path.join(figure_dir, "slip_x_contour.png"), dpi=200, bbox_inches="tight"
)
# plt.show()

fig1, ax1 = plt.subplots()
tri = ax1.tripcolor(triang, global_dds[1::3], cmap=plt.cm.rainbow, alpha=0.5)
ax1.axis("equal")
plt.colorbar(tri)
plt.title("Slip along y (m)")
plt.savefig(
    os.path.join(figure_dir, "slip_y_contour.png"), dpi=200, bbox_inches="tight"
)
# plt.show()

fig1, ax1 = plt.subplots()
tri = ax1.tripcolor(triang, global_dds[2::3], cmap=plt.cm.rainbow, alpha=0.5)
ax1.axis("equal")
plt.colorbar(tri)
plt.title("Opening (m)")
plt.savefig(
    os.path.join(figure_dir, "opening_contour.png"), dpi=200, bbox_inches="tight"
)
# plt.show()

col_pts = np.asarray(
    [np.mean(mesh.coor[mesh.conn[e, :], :], axis=0) for e in range(mesh.nelts)]
)  # h1.getCollocationPoints()
rcoor_mid = np.sqrt(col_pts[:, 0] ** 2 + col_pts[:, 1] ** 2)
fig, ax = plt.subplots()
ax.plot(rcoor_mid, global_dds[2::3], ".", label="Numerical")
plt.xlabel("r (m)")
plt.ylabel("Opening (m)")
plt.legend()
plt.savefig(
    os.path.join(figure_dir, "opening_vs_radius.png"), dpi=200, bbox_inches="tight"
)
# plt.show()

fig, ax = plt.subplots()
ax.plot(rcoor_mid, global_dds[0::3], ".", label="Numerical")
plt.xlabel("r (m)")
plt.ylabel("Slip (m)")
plt.legend()
plt.savefig(
    os.path.join(figure_dir, "slip_vs_radius.png"), dpi=200, bbox_inches="tight"
)
# plt.show()


# %% yield function

fig1, ax1 = plt.subplots()
tri = ax1.tripcolor(triang, solN.yieldF, cmap=plt.cm.rainbow, alpha=0.5)
ax1.axis("equal")
plt.colorbar(tri)
plt.title("Yield function (-)")
plt.savefig(
    os.path.join(figure_dir, "yield_function_contour.png"), dpi=200, bbox_inches="tight"
)
# plt.show()

# %% traction

T_eff_n = h1.convert_to_global(solN.effective_tractions)

fig1, ax1 = plt.subplots()
tri = ax1.tripcolor(triang, T_eff_n[2::3], cmap=plt.cm.rainbow, alpha=0.5)
ax1.axis("equal")
plt.colorbar(tri)
plt.title("Normal effective traction (Pa)")
plt.savefig(
    os.path.join(figure_dir, "normal_traction_contour.png"),
    dpi=200,
    bbox_inches="tight",
)
# plt.show()

fig1, ax1 = plt.subplots()
tri = ax1.tripcolor(triang, T_eff_n[0::3], cmap=plt.cm.rainbow, alpha=0.5)
ax1.axis("equal")
plt.colorbar(tri)
plt.title("Shear effective traction (Pa)")
plt.savefig(
    os.path.join(figure_dir, "shear_traction_contour.png"), dpi=200, bbox_inches="tight"
)
# plt.show()

# %%
solN.yieldedElts


aux_k = np.where(solN.yieldedElts)[0]


# %%
alpha = alpha_h
tts = np.array([res[i].time for i in range(len(res))])
# analytical solution for pressure at collocation points, Eq. 4 in Alexi JMPS
pressure = lambda r, t: (p0 + dp_star * exp1((r**2) / (4.0 * alpha * t)))
p_col = pressure(rcoor, tts[-1])
plt.figure()
plt.plot(rcoor, solN.pressure, ".", label="Numerical")
plt.plot(rcoor, p_col, "r.", label="Analytical")
plt.xlabel("r (m)")
plt.ylabel("Pressure (Pa)")
plt.legend()
plt.savefig(
    os.path.join(figure_dir, "pressure_comparison.png"), dpi=200, bbox_inches="tight"
)
# plt.show()
# %%
eff_end = np.reshape(h1.convert_to_global(res[-1].effective_tractions), (-1, 3))

myF = eff_end[:, 0] + f_p * eff_end[:, 2]


fig1, ax1 = plt.subplots()
tri = ax1.tripcolor(triang, myF, cmap=plt.cm.rainbow, alpha=0.5)
ax1.axis("equal")
plt.colorbar(tri)
plt.title("Yield function (-)")
plt.savefig(
    os.path.join(figure_dir, "yield_function_computed.png"),
    dpi=200,
    bbox_inches="tight",
)
# plt.show()

# %%
colPt = h1.getCollocationPoints()

rcol = (colPt[:, 0] ** 2 + colPt[:, 1] ** 2) ** (0.5)

plastic_dds = h1.convert_to_global(res[-1].DDs_plastic)

fig, ax = plt.subplots()
ax.plot(rcol, -plastic_dds[0::3], ".r", label="Plastic slip")
ax.plot(rcol, -global_dds[0::3] + plastic_dds[0::3], ".b", label="Elastic slip")
plt.xlabel("r (m)")
plt.ylabel("Slip (m)")
plt.legend()
plt.savefig(
    os.path.join(figure_dir, "slip_plastic_elastic_comparison.png"),
    dpi=200,
    bbox_inches="tight",
)
# plt.show()


fig, ax = plt.subplots()
ax.plot(rcol, plastic_dds[2::3], ".r", label="Plastic opening")
ax.plot(rcol, global_dds[2::3] - plastic_dds[2::3], ".b", label="Elastic opening")
plt.xlabel("r (m)")
plt.ylabel("Opening (m)")
plt.legend()
plt.savefig(
    os.path.join(figure_dir, "opening_plastic_elastic_comparison.png"),
    dpi=200,
    bbox_inches="tight",
)
# plt.show()


# %%Verification for slip profile

lambda_crticallstress = lambda T: 1.0 / np.sqrt(2.0 * T)
# Marginally pressurized case, Eq. 23
lambda_marginallypressurized = lambda T: 0.5 * np.exp(
    (2.0 - float(mpmath.euler) - T) * 0.5
)


def lambda_approx(T):
    lam = 0
    if T > 2:
        lam = lambda_marginallypressurized(T)
    elif T < 0.3:
        lam = lambda_crticallstress(T)
    else:
        lam = (lambda_crticallstress(T) + lambda_marginallypressurized(T)) * 0.5

    return lam


def lambda_analytical(T):
    # Eq 21
    eq = (
        lambda lam2: 2
        - mpmath.euler
        + (2 / 3) * lam2 * mpmath.hyp2f2(1, 1, 2, 2.5, -lam2)
        - mpmath.log(4 * lam2)
        - T
    )
    return float(mpmath.sqrt(mpmath.findroot(eq, lambda_approx(T) ** 2).real))


tts = np.array([res[i].time for i in range(len(res))])
nyi = np.array([res[i].Nyielded for i in range(len(res))])

# find position of last yielded element (slip becomes zero)
rmax = 0.0 * tts
for i in range(len(res)):
    aux_k = np.where(res[i].yieldedElts)[0]
    rmax[i] = rcol[aux_k].max()

# %% Check for rupture radius
lam = lambda_analytical(T)
print("lambda = ", lam)
print("Predicted Rmax = ", np.sqrt(4.0 * alpha * tend) * lam)
t_ = np.linspace(tts[0], tts.max())
R = lam * np.sqrt(4.0 * alpha * t_)
plt.figure()
plt.plot(t_, R, "r")
plt.plot(tts, rmax, ".")
# plt.semilogx()
# plt.semilogy()
plt.xlabel("Time (s)")
plt.ylabel("Rupture radius (m)")
plt.legend(["Analytical solution", "Numerics"])
# plt.show()

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
# %% Analytical solution for slip profile Eq.~25, 26
# slip profile
fac = f_p * dp_star / G
# Eq 25, marginally pressurized regime
slip_marg = (
    lambda r: fac * (8 / np.pi) * (np.sqrt(1 - r**2) - np.abs(r) * np.arccos(np.abs(r)))
)
# Eq 26, critically stressed regime
slip_crit = (
    lambda r: fac
    * ((2 * np.sqrt(2 * T)) / np.pi)
    * (np.arccos(np.abs(r)) / np.abs(r) - np.sqrt(1 - r**2))
)
print(fac)
# %% Fig 3
# CS outer asymptotic solution, doesnt match for T = 0.1
lt = np.sqrt(4 * alpha * tts[-1])
rt = rmax[-1]
fac_lt = fac * lt
fac_rt = fac * rt
plt.figure()
if T <= 0.8:
    plt.plot(
        rcol / rt,
        slip_crit(rcol / rt) * lt / fac_lt,
        ".r",
        label="Reference solution",
    )
elif T >= 2.0:
    plt.plot(
        rcol / rt,
        slip_marg(rcol / rt) * rt / fac_lt,
        ".r",
        label="Analytical solution: MP",
    )
plt.plot(
    rcol / rt,
    -global_dds[0::3] / fac_lt,
    "*k",
    ms=1.2,
    label="Numerics",
)
plt.xlabel("r/Rmax")
plt.ylabel("normalized slip")
plt.legend()
# plt.xlim([0.0, 1.5])
plt.ylim([0.0, 1.5 * np.max(np.abs(res[-1].DDs.reshape(-1, 2)[:, 0]) / fac_lt)])

# %% Check for slip profile Eq.~25, 26
lam = lambda_analytical(T)
Lt = np.sqrt(4 * alpha * tts[-1])
Rt = lam * Lt
# slip profile
fac = f_p * dp_star / G
# Eq 25, marginally pressurized regime
slip_marg = (
    lambda r: fac * (8 / np.pi) * (np.sqrt(1 - r**2) - np.abs(r) * np.arccos(np.abs(r)))
)
# Eq 26, critically stressed regime
slip_crit = (
    lambda r: fac
    * ((2 * np.sqrt(2 * T)) / np.pi)
    * (np.arccos(np.abs(r)) / np.abs(r) - np.sqrt(1 - r**2))
)

plt.figure()
if T <= 0.8:
    plt.plot(
        rcol,
        slip_crit(rcol) * Rt,
        ".r",
        label="Reference solution",
    )
elif T >= 2.0:
    plt.plot(
        rcol,
        slip_marg(rcol) * Lt,
        ".r",
        label="Analytical solution: MP",
    )

plt.figure(0)
plt.plot(
    rcol,
    global_dds[2::3],
    "*k",
    ms=1.2,
    label="Numerics",
)
plt.xlabel("r")
plt.ylabel("Slip")
plt.legend()

# %%
