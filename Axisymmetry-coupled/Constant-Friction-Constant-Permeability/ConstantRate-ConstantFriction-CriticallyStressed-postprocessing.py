#
# This file is part of PyFracX-Examples
#
# POSTPROCESSING FILES FOR
# UnCoupled fluid injection at constant rate into a frictional fault in 3D (modelled as axisymmetric problem).
# Reference results from Sáez & Lecampion (2022)
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
import scipy.special
from scipy import linalg
import re
import scienceplots

plt.style.use(["science", "grid"])

from pyfracx.mesh.usmesh import usmesh
from pyfracx.utils.json_dict_dataclass_utils import *
from pyfracx.hm.HMFsolver import HMFSolution

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from ReferenceSolutions.FDFR.frictional_ruptures_3D_constant_friction import *


# %% loading numerical results
# here you need to change basefolder to the folder name of your simulation
# set it to None to look for the most recent simulation
basefolder = None
basename = "AxiSymm-ctRate-ctFriction-coupled-critstress"

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

alpha_hyd = param["Flow"]["Hydraulic diffusivity"]
cond_hyd = param["Flow"]["Hydraulic conductivity"]
Qinj = param["Injection"]["Injection rate"]
wh = param["Hydraulic aperture"]
T = param["T parameter"]
f = param["Friction coefficient"]
E = param["Elasticity"]["Young"]
nu = param["Elasticity"]["Nu"]
G = E / (2 * (1 + nu))

Nelts = mm["Nelts"]
Nnodes = Nelts + 1
coor1D = np.array(mm["Coordinates"])[:, 0]
colPts = (coor1D[1:] + coor1D[0:-1]) / 2.0  # collocation points for P0

# radial coordinates of the nodes
r = np.array([linalg.norm(coor1D[i]) for i in range(Nnodes)])

# radial coordinates of the collocation points
r_col = np.array([linalg.norm(colPts[i]) for i in range(Nelts)])
# %%
tts = np.array([res[i].time for i in range(len(res))])
nyi = np.array([res[i].Nyielded for i in range(len(res))])
rmax = 0.0 * tts

for i in range(len(res)):
    aux_k = np.where(res[i].yieldedElts)[0]

    if len(aux_k) > 0:
        rmax[i] = r_col[aux_k].max()
    else:
        rmax[i] = 0.0  # or some other appropriate value

# %%
#
# VERIFICATION OF THE RESULTS WITH ANALYTICAL SOLUTIONS
#
#
tend = tts[-1]
lam = lambda_analytical(T)
print("lambda = ", lam)
print("Predicted Rmax = ", np.sqrt(4.0 * alpha_hyd * tend) * lam)
t_ = np.linspace(tts[0], tts.max())
R = lam * np.sqrt(4.0 * alpha_hyd * t_)
plt.figure()
plt.plot(t_, R, "r", label="Analytical solution")
plt.plot(tts, rmax, ".", label="Numerical")
plt.semilogx()
plt.semilogy()
plt.xlabel("Time (s)")
plt.ylabel("Rupture radius (m)")
plt.legend()
plt.savefig(
    os.path.join(figure_dir, "rupture_radius_vs_time.png"), dpi=200, bbox_inches="tight"
)
# plt.show()

plt.figure()
plt.plot(tts, lam * np.ones(len(tts)), "-r", label="Analytical solution")
plt.plot(tts, rmax / np.sqrt(4 * alpha_hyd * tts), "ko", label="Numerical")
plt.semilogx()
plt.xlabel("Time (s)")
plt.ylabel(r"$\lambda$ = Rupture radius / $\sqrt{4 \alpha t}$ (-)")
plt.legend()
plt.title(r"T = %.3f, $\lambda = %.2f$" % (T, lam))
plt.savefig(
    os.path.join(figure_dir, "lambda_vs_time.png"), dpi=200, bbox_inches="tight"
)
# plt.show()
print(
    "Relative error from analytical prediction in Percentage : ",
    (np.abs(lam - rmax[-1] / np.sqrt(4 * alpha_hyd * tts[-1])) / lam) * 100.0,
)

# "Asymptotic solutions for self-similar fault slip induced by fluid injection at constant rate"
# by Viesca (2024)


dp = Qinj / (4 * np.pi * cond_hyd * wh)
alpha_new = 4 * alpha_hyd
lam = lambda_analytical(T)
rmax_analytical = lam * np.sqrt(4 * alpha_hyd * tts[-1])
t = tts[-1]

fig, ax = plt.subplots()
ax.plot(r_col, -np.array(res[-1].DDs_plastic)[0:-1:2], "--r", label="Numerical")

##########    ATTENTION    ##### MODIFY THE FUNCTION USED HERE #########
ax.plot(
    r_col,
    complete_slip_profile_cs(G, f, dp, alpha_hyd, T, tts[-1], lambda_analytical, r_col),
    "-k",
    ms=1.2,
    label="Analytical",
    # for critically stressed case: use the function complete_slip_profile_cs
    # for marginally stressed case: use the function complete_slip_profile_mp)
)

plt.xlabel("r (m)")
plt.ylabel("Slip (m)")
plt.legend()
plt.savefig(
    os.path.join(figure_dir, "slip_profile_comparison.png"),
    dpi=200,
    bbox_inches="tight",
)
# plt.show()

# %%

# VISULISATION OF THE PLASTIC AND ELASTIC DISPLACEMENT DISCONTINUITIES

jj = -1
fig, ax = plt.subplots()
ax.plot(r_col, -np.array(res[jj].DDs_plastic[0::2]), label="Numerical")
plt.xlabel("r (m)")
plt.ylabel("Plastic slip (m)")
plt.legend()
plt.savefig(
    os.path.join(figure_dir, "plastic_slip_profile.png"), dpi=200, bbox_inches="tight"
)
# plt.show()

fig, ax = plt.subplots()
ax.plot(
    r_col,
    -(np.array(res[jj].DDs)[0::2] - np.array(res[jj].DDs_plastic)[0::2]),
    label="Numerical",
)
plt.xlabel("r (m)")
plt.ylabel("Elastic slip (m)")
plt.legend()
plt.savefig(
    os.path.join(figure_dir, "elastic_slip_profile.png"), dpi=200, bbox_inches="tight"
)
# plt.show()

fig, ax = plt.subplots()
ax.plot(r_col, res[jj].DDs_plastic[1::2], label="Numerical")
plt.xlabel("r (m)")
plt.ylabel("Plastic opening (m)")
plt.legend()
plt.savefig(
    os.path.join(figure_dir, "plastic_opening_profile.png"),
    dpi=200,
    bbox_inches="tight",
)
# plt.show()

fig, ax = plt.subplots()
ax.plot(
    r_col,
    -(np.array(res[jj].DDs)[1::2] - np.array(res[jj].DDs_plastic)[1::2]),
    label="Numerical",
)
plt.xlabel("r (m)")
plt.ylabel("Elastic opening (m)")
plt.legend()
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
ax.plot(r_col, res[jj].effective_tractions[1::2], ".-", label="Numerical")
plt.xlabel("r (m)")
plt.ylabel("Effective normal stress (Pa)")
plt.legend()
plt.savefig(
    os.path.join(figure_dir, "effective_normal_stress.png"),
    dpi=200,
    bbox_inches="tight",
)
# plt.show()

fig, ax = plt.subplots()
ax.plot(r_col, res[jj].effective_tractions[0::2], ".-", label="Numerical")
plt.xlabel("r (m)")
plt.ylabel("Shear stress (Pa)")
plt.legend()
plt.savefig(os.path.join(figure_dir, "shear_stress.png"), dpi=200, bbox_inches="tight")
# plt.show()

fig, ax = plt.subplots()
ax.plot(r_col, res[jj].yieldF, label="Numerical")
plt.xlabel("r (m)")
plt.ylabel("Yield function (Pa)")
plt.legend()
plt.savefig(
    os.path.join(figure_dir, "yield_function.png"), dpi=200, bbox_inches="tight"
)
# plt.show()


# %%
