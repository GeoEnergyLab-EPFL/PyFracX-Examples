#
# This file is part of PyFracX-Examples
#

#
# Coupled fluid injection into a 2D planar frictional fault (plane-strain problem) due to a constant over-pressure
# Constant friction reference results from Viesca (2021)
# All properties are constant
# -> the problem is therefore uncoupled (but we solve it as coupled)  - verification test
#
# %% imports
import os
import sys
import re
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
import scienceplots

plt.style.use(["science", "grid"])

from pyfracx.mesh.usmesh import usmesh
from pyfracx.utils.json_dict_dataclass_utils import *
from pyfracx.hm.HMFsolver import HMFSolution

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from ReferenceSolutions.FDFR.Plane_TwoD_frictional_ruptures import *


# %% loading numerical results
# here you need to change basefolder to the folder name of your simulation
# set it to None to look for the most recent simulation
basefolder = None
basename = "2D-ctP-ctFriction"

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

alpha_hyd = param["Flow"]["Hydraulic diffusivity"]
T = param["T parameter"]
f_p = param["Friction"]
sigmap_o = param["Initial stress"][0]
tau_o = param["Initial stress"][1]
YoungM = param["Elasticity"]["Young"]
nu = param["Elasticity"]["Nu"]

shear_prime = YoungM / (2 * (1 + nu) * 2 * (1 - nu))
dpcenter = (1.0 - tau_o / (f_p * sigmap_o)) * sigmap_o / T

mm = json_read(os.path.join(basefolder, "Mesh"))
Nelts = mm["Nelts"]
coor1D = np.array(mm["Coordinates"])[:, 0]

# %% Setup figure directory
figure_dir = f"figures_{basename}"
os.makedirs(figure_dir, exist_ok=True)

# %% plots


# Select time-step to plot profiles later
jj = len(res) - 2
t = res[jj].time

# numerical results
tts = np.array([res[i].time for i in range(len(res))])
nyi = np.array([res[i].Nyielded for i in range(len(res))])
cr_front = nyi / 2.0 * (coor1D[1] - coor1D[0])  # only for uniform mesh

# analytical solution
if T < 0.4:
    lam = 2 / (np.pi ** (3 / 2)) * 1 / T
    print("Critically stressed case lambda=", lam)
else:
    lam = (np.pi ** (3 / 2)) / 4.0 * (1 - T)
    print("Marginally pressurized case lambda=", lam)

y_ = lam * np.sqrt(4.0 * alpha_hyd * tts)  # crack front from analytical solution

# %% Plot: Overpressure
colPts = (coor1D[1:] + coor1D[0:-1]) / 2.0  # collocation points for P0
pressureAtColPts = lambda t, Dpcenter: Dpcenter * special.erfc(
    np.abs(colPts) / ((4.0 * alpha_hyd * t) ** 0.5)
)

jj = len(res) - 2
fig, ax = plt.subplots()
solp = pressureAtColPts(res[jj].time, dpcenter)
ax.plot(colPts, solp, "r-", label="Analytical")
ax.plot(coor1D, res[jj].pressure, "b-", label="Numerical")
plt.xlabel("x (m)")
plt.ylabel("Over-pressure (Pa)")
plt.xlim([-cr_front[jj], cr_front[jj]])
ax.legend()
plt.savefig(
    os.path.join(figure_dir, "overpressure_profile.png"), dpi=200, bbox_inches="tight"
)
# plt.show()


# %%  Plot: crack half length VS time
fig, ax = plt.subplots()
ax.loglog(tts, y_, "r", label="Analytical")
ax.loglog(tts, cr_front, ".", label="Numerical")
plt.xlabel("Time (s)")
plt.ylabel("Crack half-length (m)")
ax.legend()
plt.savefig(
    os.path.join(figure_dir, "crack_halflength_vs_time.png"),
    dpi=200,
    bbox_inches="tight",
)
# plt.show()

# #%% Plot: Slip profile (Dimensionless)
# if T > 0.4:
#     xx = np.linspace(-1, 1, Nelts)
#     #analytical_slip = marginallyStressed_slip(xx)
#     analytical_slip = marginallyPressurized_slip(xx, lam)
#     slip_scale_MP =lam**2*np.sqrt(4*alpha_hyd*t)*f_p*dpcenter/shear_prime
#     numerical_slip = -np.array(res[jj].DDs)[0::2] / slip_scale_MP
#     fig, ax = plt.subplots()
#     ax.plot(xx, analytical_slip,'r.')
#     ax.plot((coor1D[1:] + coor1D[0:-1]) / (2. * cr_front[jj]), numerical_slip,'b-')
#     plt.xlabel("x (-)")
#     plt.ylabel("Non-dimensional Slip profile (-))")
#     ax.legend(['Analytical','Numerical'])
#     ax.set_xlim([-1., 1.])  # range for x-axis
#     # plt.show()
# else:
#     xx = (coor1D[1:] + coor1D[0:-1]) / 2
#     xx_new = xx[(xx < cr_front[jj]) & (xx > - cr_front[jj])]

#     slip_scale_CS =np.sqrt(4*alpha_hyd*t)*f_p*dpcenter/shear_prime
#     numerical_slip = -np.array(res[jj].DDs)[0::2] / slip_scale_CS

#     fig, ax = plt.subplots()
#     ax.plot( xx / cr_front[jj], numerical_slip, 'b.')

#     x_ = xx_new[(xx_new > cr_front[jj] / 2.) | (xx_new < -cr_front[jj] / 2)] / cr_front[jj]
#     analytical_slip_Out = criticallyStressed_slip_Outer(x_)
#     ax.plot(x_, analytical_slip_Out, 'r--')

#     x__ = xx[(xx < cr_front[jj] / 2.) & (xx > -cr_front[jj] / 2)]
#     xhat__ = x__ / np.sqrt(4 * alpha_hyd * t)
#     analytical_slip_In = criticallyStressed_slip_Inner(xhat__ , lam)
#     ax.plot(x__ / cr_front[jj], analytical_slip_In, 'g--')

#     ax.legend(['Numerical','Analytical outer','Analytical inner'])
#     ax.set_xlim([-1., 1.])  # range for x-axis
#     plt.xlabel("x (-)")
#     plt.ylabel("Non-dimensional Slip profile (-))")
#     # plt.show()

# %% Plot: Slip profile (Dimensional)

if T > 0.4:
    xx = (coor1D[1:] + coor1D[0:-1]) / 2
    analytical_slip = marginallyStressed_slip_dimensional(
        xx, T, alpha_hyd, t, f_p, dpcenter, shear_prime
    )
    numerical_slip = -np.array(res[jj].DDs)[0::2]
    fig, ax = plt.subplots()
    ax.plot(xx, analytical_slip, "r-", label="Analytical")
    ax.plot(xx, numerical_slip, "b.", label="Numerical")
    plt.xlabel("x (m)")
    plt.ylabel("Slip (m)")
    ax.legend()
    ax.set_xlim([-cr_front[jj], cr_front[jj]])  # range for x-axis
    plt.savefig(
        os.path.join(figure_dir, "slip_profile_dimensional.png"),
        dpi=200,
        bbox_inches="tight",
    )
    # plt.show()
else:
    xx = (coor1D[1:] + coor1D[0:-1]) / 2
    xx_new = xx[(xx < cr_front[jj]) & (xx > -cr_front[jj])]

    slip_scale_CS = np.sqrt(4 * alpha_hyd * t) * f_p * dpcenter / shear_prime
    numerical_slip = -np.array(res[jj].DDs)[0::2]

    fig, ax = plt.subplots()
    ax.plot(xx, numerical_slip, label="Numerical")

    x_ = xx_new[(xx_new > cr_front[jj] / 2.0) | (xx_new < -cr_front[jj] / 2)]
    analytical_slip_Out = slip_scale_CS * criticallyStressed_slip_Outer(
        x_ / cr_front[jj]
    )
    ax.plot(x_, analytical_slip_Out, "r--", label="Analytical outer")

    x__ = xx[(xx < y_[jj] / 2.0) & (xx > -y_[jj] / 2)]
    analytical_slip_In = slip_scale_CS * criticallyStressed_slip_Inner(
        x__ / np.sqrt(4.0 * alpha_hyd * t), lam
    )
    ax.plot(x__, analytical_slip_In, "g--", label="Analytical inner")

    ax.legend()
    ax.set_xlim([-cr_front[jj], cr_front[jj]])  # range for x-axis
    plt.xlabel("x (m)")
    plt.ylabel("Slip (m)")
    plt.savefig(
        os.path.join(figure_dir, "slip_profile_dimensional.png"),
        dpi=200,
        bbox_inches="tight",
    )
    # plt.show()


# %% time - stepping

dts = np.array([res[i].timestep for i in range(len(res))])
fig, ax = plt.subplots()
ax.loglog(tts, dts, ".")
plt.xlabel("Time (s)")
plt.ylabel("Time-step (s)")
plt.savefig(
    os.path.join(figure_dir, "timestep_evolution.png"), dpi=200, bbox_inches="tight"
)
# plt.show()
#  PLOTTING slip profile ----

ltes = np.array([res[i].lte for i in range(len(res))])
fig, ax = plt.subplots()
ax.loglog(tts, ltes, ".")
plt.xlabel("Time (s)")
plt.ylabel("LTE estimate (-)")
plt.savefig(os.path.join(figure_dir, "lte_estimate.png"), dpi=200, bbox_inches="tight")
# plt.show()


# %%
jj = -1
fig, ax = plt.subplots()
ax.plot((coor1D[1:] + coor1D[0:-1]) / 2.0, -np.array(res[jj].DDs_plastic[0::2]))
plt.xlabel("x (m)")
plt.ylabel("Plastic slip (m)")
plt.savefig(
    os.path.join(figure_dir, "plastic_slip_profile.png"), dpi=200, bbox_inches="tight"
)
# plt.show()

# %%
fig, ax = plt.subplots()
ax.plot(
    (coor1D[1:] + coor1D[0:-1]) / 2.0,
    -(np.array(res[jj].DDs)[0::2] - np.array(res[jj].DDs_plastic)[0::2]),
)
plt.xlabel("x (m)")
plt.ylabel("Elastic slip (m)")
plt.savefig(
    os.path.join(figure_dir, "elastic_slip_profile.png"), dpi=200, bbox_inches="tight"
)
# plt.show()

fig, ax = plt.subplots()
ax.plot((coor1D[1:] + coor1D[0:-1]) / 2.0, res[jj].DDs_plastic[1::2])
plt.xlabel("x (m)")
plt.ylabel("Plastic opening (m)")
plt.savefig(
    os.path.join(figure_dir, "plastic_opening_profile.png"),
    dpi=200,
    bbox_inches="tight",
)
# plt.show()

fig, ax = plt.subplots()
ax.plot(
    (coor1D[1:] + coor1D[0:-1]) / 2.0,
    -(np.array(res[jj].DDs)[1::2] - np.array(res[jj].DDs_plastic)[1::2]),
)
plt.xlabel("x (m)")
plt.ylabel("Elastic opening (m)")
plt.savefig(
    os.path.join(figure_dir, "elastic_opening_profile.png"),
    dpi=200,
    bbox_inches="tight",
)
# plt.show()

# %%
fig, ax = plt.subplots()
ax.plot((coor1D[1:] + coor1D[0:-1]) / 2.0, res[jj].effective_tractions[1::2], ".-")
plt.xlabel("x (m)")
plt.ylabel("Effective normal stress (Pa)")
plt.savefig(
    os.path.join(figure_dir, "effective_normal_stress.png"),
    dpi=200,
    bbox_inches="tight",
)
# plt.show()

fig, ax = plt.subplots()
ax.plot((coor1D[1:] + coor1D[0:-1]) / 2.0, res[jj].effective_tractions[0::2], ".-")
plt.xlabel("x (m)")
plt.ylabel("Shear stress (Pa)")
plt.savefig(os.path.join(figure_dir, "shear_stress.png"), dpi=200, bbox_inches="tight")
# plt.show()

fig, ax = plt.subplots()
ax.plot((coor1D[1:] + coor1D[0:-1]) / 2.0, res[jj].yieldF)
plt.xlabel("x (m)")
plt.ylabel("Yield function (Pa)")
plt.savefig(
    os.path.join(figure_dir, "yield_function.png"), dpi=200, bbox_inches="tight"
)
# plt.show()


# %%
