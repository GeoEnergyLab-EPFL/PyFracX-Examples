#
# This file is part of PyFracX-Examples
#
# POSTPROCESSING FILES FOR
# Fluid injection at constant rate into a slip weakening frictional fault in 3D (modelled as axisymmetric problem).
# Reference results from Sáez & Lecampion (2023)
#
# %%+
# Importing the necessary python libraries and managing the python path

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path
import json
from scipy import linalg
import re
import scienceplots

plt.style.use(["science", "grid"])

from pyfracx.mesh.usmesh import UnstructuredMesh
from pyfracx.utils.json_dict_dataclass_utils import *
from pyfracx.hm.HMFsolver import HMFSolution

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from ReferenceSolutions.FDFR.frictional_ruptures_3D_constant_friction import *


# %% loading numerical results
# here you need to change basefolder to the folder name of your simulation
# set it to None to look for the most recent simulation
basefolder = None
basename = "AxiSymm-ctRate-coupled-lwfric-S_0.6-P_0.035"

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
            f"No folder found matching pattern '{basename}_DD-MM-YYYY-HH-MM-SS' in '{res_dir}'"
        )

    # Sort by datetime (most recent last) and pick the most recent
    matching_folders.sort(key=lambda x: x[1])
    most_recent_folder = matching_folders[-1][0]

    basefolder = os.path.join(res_dir, most_recent_folder)

print(f"Using results folder: {basefolder}")


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
# %% Setup figure directory
figure_dir = f"figures_{basename}"
os.makedirs(figure_dir, exist_ok=True)


# %%
# Extracting the parameters of the simulation from the json file

alpha_hyd = param["Flow"]["Hydraulic diffusivity"]
cond_hyd = param["Flow"]["Hydraulic conductivity"]
Qinj = param["Injection"]["Injection rate"]
wh = param["Hydraulic aperture"]
T = param["T peak"]
S = param["Pre-stress ratio"]
P = param["Overpressure ratio"]
F = param["Peak to residual friction"]
f_p = param["Friction model"]["Peak friction"]
fric_model = param["Friction model"]["Model"]
print("Friction model:", fric_model)
f_r = F * f_p
E = param["Elasticity"]["Young"]
nu = param["Elasticity"]["Nu"]
G = E / (2 * (1 + nu))
Rw = param["Rupture length scale"]
Nelts = mm["Nelts"]
Nnodes = Nelts + 1
coor1D = np.array(mm["Coordinates"])[:, 0]
colPts = (coor1D[1:] + coor1D[0:-1]) / 2.0  # collocation points for P0

# radial coordinates of the nodes
r = np.array([linalg.norm(coor1D[i]) for i in range(Nnodes)])

# radial coordinates of the collocation points
r_col = np.array([linalg.norm(colPts[i]) for i in range(Nelts)])

# %%

arg_fric_mod = "lin"  # 'lin' or 'exp' based on the friction model used

# Construct the file path (modify it according to your file structure)

base_path = f"../../ReferenceSolutions/FDFR/SlipWeakeningBenchmarks/{fric_model}/"
file_name = f"P{P}/S_{S}_P_{P}_F_{F}_{arg_fric_mod}_QD_pz50_L15_"
file_path = base_path + file_name

# Open and load the JSON file
with open(file_path, "r") as file:
    data = json.load(file)

# Example extraction of 'tlistNorm' and 'RfNorm' from the first entry (adjust according to your JSON structure)
tlistNorm = data["tlistNorm"]
RfNorm = data["RfNorm"]

# %%
##### VERIFICATION OF THE RESULTS WITH THE REFERENCE SOLUTIONS  #####

tts = np.array([res[i].time for i in range(len(res))])
nyi = np.array([res[i].Nyielded for i in range(len(res))])

# find position of last yielded element (slip becomes zero)
rmax = 0.0 * tts

for i in range(len(res)):
    aux_k = np.where(res[i].yieldedElts)[0]

    if len(aux_k) > 0:
        rmax[i] = r_col[aux_k].max()
    else:
        rmax[i] = 0.0  # or some other appropriate value

x_cor = np.sqrt(4 * alpha_hyd * tts)
x_cor = x_cor / Rw
y_cor = rmax / Rw

plt.figure()
plt.plot(x_cor, y_cor, "-.r", label="Numerics")
plt.plot(tlistNorm, RfNorm, "-k", label="Benchmark")
plt.xlabel(r"$\sqrt{4 \alpha t}/R_w$")
plt.ylabel(r"$R/R_w$")
plt.legend()
plt.savefig(
    os.path.join(figure_dir, "rupture_radius_vs_time.png"), dpi=200, bbox_inches="tight"
)
# plt.show()
# %%

# VISULISATION OF THE PLASTIC AND ELASTIC DISPLACEMENT DISCONTINUITIES
# slip plot

jj = -1
fig, ax = plt.subplots()
ax.plot(colPts, -np.array(res[jj].DDs[0:-1:2]), ".")
# ax.plot(coor1D[:],p_anal,'-g')
plt.xlabel("r (m)")
plt.ylabel(" slip ")
plt.title("Total slip at t = %.3f secs" % (res[-1].time))
plt.savefig(
    os.path.join(figure_dir, "total_slip_profile.png"), dpi=200, bbox_inches="tight"
)
# plt.show()

fig, ax = plt.subplots()
ax.plot(r_col, -np.array(res[jj].DDs_plastic[0::2]))
plt.xlabel("x (m)")
plt.ylabel("Plastic slip profile (m) at time last step")
plt.savefig(
    os.path.join(figure_dir, "plastic_slip_profile.png"), dpi=200, bbox_inches="tight"
)
# plt.show()

fig, ax = plt.subplots()
ax.plot(r_col, -(np.array(res[jj].DDs)[0::2] - np.array(res[jj].DDs_plastic)[0::2]))
plt.xlabel("x (m)")
plt.ylabel("elastic slip profile (m) at time last step")
plt.savefig(
    os.path.join(figure_dir, "elastic_slip_profile.png"), dpi=200, bbox_inches="tight"
)
# plt.show()

fig, ax = plt.subplots()
ax.plot(r_col, res[jj].DDs_plastic[1::2])
plt.xlabel("x (m)")
plt.ylabel("Plastic opening profile (m) at time last step")
plt.savefig(
    os.path.join(figure_dir, "plastic_opening_profile.png"),
    dpi=200,
    bbox_inches="tight",
)
# plt.show()

fig, ax = plt.subplots()
ax.plot(r_col, -(np.array(res[jj].DDs)[1::2] - np.array(res[jj].DDs_plastic)[1::2]))
plt.xlabel("x (m)")
plt.ylabel("elastic Opg (m) at time last step")
plt.savefig(
    os.path.join(figure_dir, "elastic_opening_profile.png"),
    dpi=200,
    bbox_inches="tight",
)
# plt.show()


# %%
#
#
# VISULISATION OF THE STRESS PROFILES
#
#
fig, ax = plt.subplots()
ax.plot(r_col, res[jj].effective_tractions[1::2], ".-")
plt.xlabel("x (m)")
plt.ylabel("Effective normal stress (Pa) at time last step")
plt.savefig(
    os.path.join(figure_dir, "eff_normal_traction_profile.png"),
    dpi=200,
    bbox_inches="tight",
)
# plt.show()

fig, ax = plt.subplots()
ax.plot(r_col, res[jj].effective_tractions[0::2], ".-")
plt.xlabel("x (m)")
plt.ylabel("Shear stress (Pa) at time last step")
plt.savefig(
    os.path.join(figure_dir, "shear_traction_profile.png"),
    dpi=200,
    bbox_inches="tight",
)
# plt.show()

fig, ax = plt.subplots()
ax.plot(r_col, res[jj].yieldF)
plt.xlabel("x (m)")
plt.ylabel("Yield function (Pa) at time last step")
plt.savefig(
    os.path.join(figure_dir, "yield_function_profile.png"),
    dpi=200,
    bbox_inches="tight",
)
# plt.show()


# %%
