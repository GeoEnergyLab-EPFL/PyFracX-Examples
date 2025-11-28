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
import re
import time
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import gmsh
import scienceplots

plt.style.use(["science", "grid"])


#  Imports from PyFracX
from pyfracx.mesh.usmesh import usmesh
from pyfracx.utils.json_dict_dataclass_utils import *
from pyfracx.hm.HMFsolver import HMFSolution
from pyfracx.loads.Injection import *
from pyfracx.mechanics.H_Elasticity import *

# %% loading numerical results
# here you need to change basefolder to the folder name of your simulation
# set it to None to look for the most recent simulation
basefolder = None
basename = "3D-HF-ReOpening-benchmark"

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
mesh = usmesh(2, np.array(mm["Coordinates"]), np.array(mm["Connectivity"]), 0)

Qinj = param["Injection"]["Injection rate"]
YoungM = param["Elasticity"]["Young"]
nu = param["Elasticity"]["Nu"]
wh_o = param["Flow"]["Initial aperture"]
fluid_visc = param["Flow"]["Fluid viscosity"]
hyd_cond = param["Flow"]["Hydraulic conductivity"]
c_f = param["Flow"]["Storage "]
f_p = param["Friction coefficient"]


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
ax1.plot(timestamp, inj_pressure, label="Numerical")
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


fig, ax = plt.subplots()
ax.plot(rcoor_mid, (global_dds[2::3] / wh_o) ** 3.0, ".", label="Numerical")
plt.xlabel("r (m)")
plt.ylabel("Transmissivity increase (-)")
plt.legend()
plt.savefig(
    os.path.join(figure_dir, "transmissivity_increase_vs_radius.png"),
    dpi=200,
    bbox_inches="tight",
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
