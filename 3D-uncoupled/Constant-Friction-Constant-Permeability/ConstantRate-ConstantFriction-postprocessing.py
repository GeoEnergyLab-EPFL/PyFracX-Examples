# %%
# Importing the necessary python libraries and managing the python path
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path
import scipy
import matplotlib
from scipy import linalg
import re
import scienceplots

plt.style.use(["science", "grid"])

from pyfracx.mesh.usmesh import usmesh
from pyfracx.utils.json_dict_dataclass_utils import *
from pyfracx.hm.HMFsolver import HMFSolution
from pyfracx.mechanics.H_Elasticity import *
from pyfracx.mechanics import H_Elasticity

# %% loading numerical results
# here you need to change basefolder to the folder name of your simulation
# set it to None to look for the most recent simulation
basefolder = None
basename = "3D-ctFriction-ctperm-benchmark"

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
# %%
alpha_hyd = param["Flow"]["Hydraulic diffusivity"]
cond_hyd = param["Flow"]["Hydraulic conductivity"]
Qinj = param["Injection"]["Injection rate"]
wh = param["Flow"]["Initial aperture"]
T = param["T parameter"]
f = param["Friction coefficient"]
E = param["Elasticity"]["Young"]
nu = param["Elasticity"]["Nu"]
G = E / (2 * (1 + nu))

Nelts = mm["Nelts"]
conn = np.array(mm["Connectivity"])
coor = np.array(mm["Coordinates"])
Nnodes = len(coor)
mesh = usmesh(3, coor, conn, 0)

colPts = [
    (coor[conn[i][0]] + coor[conn[i][1]] + coor[conn[i][2]]) / 3.0 for i in range(Nelts)
]  # put it in usmesh
r = np.array([scipy.linalg.norm(coor[i]) for i in range(Nnodes)])
r_col = np.array([scipy.linalg.norm(colPts[i]) for i in range(Nelts)])

# %%
## plotting the unstructured mesh
triang = matplotlib.tri.Triangulation(coor[:, 0], coor[:, 1], triangles=conn, mask=None)
fig1, ax1 = plt.subplots()
ax1.set_aspect("equal")
ax1.triplot(triang, "b-", lw=1)
ax1.plot(0.0, 0.0, "ko", label="Injection point")
plt.legend()
plt.title("3D Mesh")
plt.savefig(os.path.join(figure_dir, "mesh_3d.png"), dpi=200, bbox_inches="tight")
# plt.show()

# %%
# Create hmatrix for using the functions


kernel = "3DT0-H"
elas_properties = np.array([E, nu])
elastic_m = Elasticity(
    kernel, elas_properties, max_leaf_size=64, eta=3.0, eps_aca=1.0e-3
)
# hmat creation
h1 = elastic_m.constructHmatrix(mesh)

# %%
solN = res[-1]

rcoor = (coor[:, 0] ** 2 + coor[:, 1] ** 2) ** (0.5)

fig, ax = plt.subplots()
ax.plot(r_col, solN.pressure, ".", label="Numerical")
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
    [np.mean(coor[conn[e, :], :], axis=0) for e in range(Nelts)]
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

# %%
plastic_dds = h1.convert_to_global(res[-1].DDs_plastic)

fig, ax = plt.subplots()
ax.plot(r_col, -plastic_dds[0::3], ".r", label="Plastic slip")
ax.plot(r_col, -global_dds[0::3] + plastic_dds[0::3], ".b", label="Elastic slip")
plt.xlabel("r (m)")
plt.ylabel("Slip (m)")
plt.legend()
# plt.xlim(0, 15)
plt.savefig(
    os.path.join(figure_dir, "slip_plastic_elastic_comparison.png"),
    dpi=200,
    bbox_inches="tight",
)
# plt.show()


fig, ax = plt.subplots()
ax.plot(r_col, plastic_dds[2::3], ".r", label="Plastic opening")
ax.plot(r_col, global_dds[2::3] - plastic_dds[2::3], ".b", label="Elastic opening")
plt.xlabel("r (m)")
plt.ylabel("Opening (m)")
plt.legend()
plt.savefig(
    os.path.join(figure_dir, "opening_plastic_elastic_comparison.png"),
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
