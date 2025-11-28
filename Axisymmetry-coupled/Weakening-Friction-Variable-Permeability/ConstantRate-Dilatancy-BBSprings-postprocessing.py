#
# This file is part of PyFracX-Examples
#

#
# Coupled fluid injection into an axisymmetric frictional fault (0 Poisson's ratio, plane-strain problem)
# Constant injection rate then shut-in
# Linear weakening friction
# Variable permeability due to dilatancy and nonlinear springs (Barton-Bandis model)
# There is no existing ref solution
#

# %% imports
import json
import os
import re
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "grid"])

from pyfracx.mesh.usmesh import usmesh
from pyfracx.utils.json_dict_dataclass_utils import *
from pyfracx.hm.HMFsolver import HMFSolution
from pyfracx.mechanics.evolutionLaws import linearEvolution

# %% loading numerical results
# here you need to change basefolder to the folder name of your simulation
# set it to None to look for the most recent simulation
basefolder = None
basename = "Axisymmetric-ctQ-LinearWeakening-VarPermeability"


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

me = json_read(os.path.join(basefolder, "Mesh"))

f_p = param["Friction coefficient"]["peak"]
f_r = param["Friction coefficient"]["residual"]
d_c = param["Friction coefficient"]["d_c"]
psi_p = param["Friction coefficient"]["peak dilatancy"]

sigmap_o = param["Initial stress"][0]
tau_o = param["Initial stress"][1]

YoungM = param["Elasticity"]["Young"]
nu = param["Elasticity"]["Nu"]

QinjTimeData = param["Injection history"]["Constant Rate"]
C_wd = param["Injection history"]["Wellbore storage"]

who = param["Flow"]["who"]
mu = param["Flow"]["Viscosity"]
S_e = param["Flow"]["Compressibility"]
Tav = param["Flow"]["Transmissibility"]
alpha_hydro = param["Flow"]["rock diffusivity"]

ks = param["Interface stiffness"]["ks"]
kni = param["Interface stiffness"]["kni"]
v_m = param["Interface stiffness"]["vm"]

shear_prime = YoungM / (2 * (1 + nu) * 2 * (1 - nu))

Nelts = me["Nelts"]
coor1D = np.array(me["Coordinates"])[:, 0]

# %% plots
timestep = -2
num_pressure = res[timestep].pressure
tt = np.array([res[j].time for j in range(len(res))])

####################################################################
####################################################################
# %% Overpressure at the well
pw_t = np.array([res[j].pressure[2] for j in range(len(res))])

fig, ax = plt.subplots()
ax.plot(tt[:], 1e-6 * pw_t[:], ".b")
plt.xlabel("Time (s)")
plt.ylabel("Well Pressure MPa")
plt.xlim([0, res[-1].time])
plt.savefig(
    os.path.join(figure_dir, "injection_pressure.png"), dpi=200, bbox_inches="tight"
)
# plt.show()

# %% ##### rupture radius
nyi = np.array([res[i].Nyielded for i in range(len(res))])
cr_front = nyi * (coor1D[1] - coor1D[0])
fig, ax = plt.subplots()
ax.plot(tt, cr_front, ".y")

plt.xlabel("Time (s)")
plt.ylabel("rupture radius(m)")
plt.savefig(
    os.path.join(figure_dir, "rupture_radius.png"), dpi=200, bbox_inches="tight"
)
# plt.show()

# %%
#### slip at origin

slip_0 = np.abs(np.array([res[i].DDs[0] for i in range(len(res))]))
slip_p_0 = np.abs(np.array([res[i].DDs_plastic[0] for i in range(len(res))]))

fig, ax = plt.subplots()
ax.plot(tt, slip_0, ".k", label="Total Slip")
ax.plot(tt, slip_p_0, ".r", label="Plastic Slip")
plt.xlabel("Time (s)")
plt.ylabel("Slip @ injection point (m)")
plt.legend()
plt.savefig(
    os.path.join(figure_dir, "slip_at_injection_point.png"),
    dpi=200,
    bbox_inches="tight",
)
# plt.show()

# %% width at origin
width_0 = np.abs(np.array([res[i].DDs[1] for i in range(len(res))]))
width_p_0 = np.abs(np.array([res[i].DDs_plastic[1] for i in range(len(res))]))

fig, ax = plt.subplots()
ax.plot(tt, width_0, ".k", label="Total Opg")
ax.plot(tt, width_p_0, ".b", label="Plastic Opg")
plt.xlabel("Time (s)")
plt.ylabel("Width @ injection point (m)")
plt.legend()
plt.savefig(
    os.path.join(figure_dir, "width_at_injection_point.png"),
    dpi=200,
    bbox_inches="tight",
)
# plt.show()

# %%
fig, ax = plt.subplots()
ax.plot(slip_0, width_0, ".k", label="Total slip (x) VS Total Opg (y)")
ax.plot(slip_p_0, width_p_0, ".b", label="Plastic slip (x) VS Plastic Opg (y)")
plt.xlabel("Slip @ injection point (m)")
plt.ylabel("Width @ injection point (m)")
plt.legend()
plt.savefig(
    os.path.join(figure_dir, "slip_vs_width_at_injection_point.png"),
    dpi=200,
    bbox_inches="tight",
)
# plt.show()

# %%
### profiles
timestep = -1
sol_to_p = res[timestep]
x = np.array(me["Coordinates"])[:, 0]
num_pressure = sol_to_p.pressure
fig, ax = plt.subplots()
ax.plot(x[1:400], num_pressure[1:400], ".b")
plt.xlabel("  Radius (m)")
plt.ylabel("Fluid pressure (Pa)")
plt.savefig(
    os.path.join(figure_dir, "pressure_profile.png"), dpi=200, bbox_inches="tight"
)
# plt.show()

# %%
# width profile
w_profile = sol_to_p.DDs[1::2]
w_profile_p = sol_to_p.DDs_plastic[1::2]
fig, ax = plt.subplots()
ax.plot((x[1:] + x[0::-2]) / 2.0, w_profile[:], ".k")
ax.plot((x[1:] + x[0::-2]) / 2.0, w_profile_p[:], ".b")
plt.xlabel("  Radius (m)")
plt.ylabel(" Width (m)")
plt.savefig(
    os.path.join(figure_dir, "opening_profile.png"), dpi=200, bbox_inches="tight"
)
# plt.show()

fig, ax = plt.subplots()

colPts = np.array(x[1:] + x[0::-2]) / 2.0

ax.plot(colPts, (1 + np.array(w_profile)[:] / who) ** 3, ".k")
plt.xlabel("  Radius (m)")
plt.ylabel(" Hydraulic transmissibility increase (-)")
plt.savefig(
    os.path.join(figure_dir, "transmissibility_increase_profile.png"),
    dpi=200,
    bbox_inches="tight",
)
# plt.show()

# %%
# slip profile
slip_profile = sol_to_p.DDs[0::2]
slip_profile_p = sol_to_p.DDs_plastic[0::2]
fig, ax = plt.subplots()
# ax.plot(x[1:],true_sol[1:],'r')
ax.plot(colPts, -np.array(slip_profile)[:], ".k")
ax.plot(colPts, -np.array(slip_profile_p)[:], ".b")
plt.xlabel("  r (m)")
plt.ylabel("slip (m)")
plt.savefig(
    os.path.join(figure_dir, "slip_profile.png"),
    dpi=200,
    bbox_inches="tight",
)
# plt.show()

# dilatancy profile
fig, ax = plt.subplots()
app_dila_c = -np.array(sol_to_p.DDs_rate)[1::2] / np.array(sol_to_p.DDs_rate)[0::2]
ax.plot(colPts, app_dila_c, ".b")
plt.xlabel("  r (m)")
plt.ylabel(" w/slip (-)")
plt.savefig(
    os.path.join(figure_dir, "width_vs_slip_profile.png"),
    dpi=200,
    bbox_inches="tight",
)
# plt.show()

# %%
# effective tracitons profile
tau = np.array(sol_to_p.effective_tractions)[0::2]
sig_n = np.array(sol_to_p.effective_tractions)[1::2]

fig, ax = plt.subplots()
# ax.plot(x[1:],true_sol[1:],'r')
ax.plot(colPts, -tau[:], ".k", label="Shear component")
ax.plot(colPts, -sig_n[:], ".b", label="Normal component")
plt.xlabel("  r (m)")
plt.ylabel("effective tractions (Pa)")
plt.legend()
plt.savefig(
    os.path.join(figure_dir, "effective_traction_profile.png"),
    dpi=200,
    bbox_inches="tight",
)
# plt.show()


fric_c = [
    linearEvolution(np.array(slip_profile_p)[i], d_c, f_p, f_r) for i in range(Nelts)
]
dila_c = [
    linearEvolution(np.array(slip_profile_p)[i], d_c, psi_p, 0.0) for i in range(Nelts)
]
fig, ax = plt.subplots()
ax.plot(colPts, fric_c, ".k", label="Friction evol.")
ax.plot(colPts, dila_c, ".b", label="Dilatancy evol.")

plt.xlabel("  r (m)")
plt.ylabel("friction coef. and dilatancy coef.")
plt.legend()
plt.savefig(
    os.path.join(figure_dir, "friction_coeff_and_dilatancy_profile.png"),
    dpi=200,
    bbox_inches="tight",
)
# plt.show()

###
fig, ax = plt.subplots()
ax.plot(colPts, np.abs(tau) + fric_c * sig_n, ".b")

plt.xlabel("  r (m)")
plt.ylabel(" F_mc")
plt.savefig(
    os.path.join(figure_dir, "MC_failure_criterion_profile.png"),
    dpi=200,
    bbox_inches="tight",
)
# plt.show()

# %%
