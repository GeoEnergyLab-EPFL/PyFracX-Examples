# %% SCRIPT FOR THE SIMULATION OF THE BASEL-1 well Stimulation
import os
import re
from datetime import datetime
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "grid"])

from pyfracx.mesh.usmesh import UnstructuredMesh
from pyfracx.utils.json_dict_dataclass_utils import *
from pyfracx.hm.HMFsolver import HMFSolution
from pyfracx.mechanics.evolutionLaws import *


# %% loading numerical results
# set it to None to look for the most recent simulation
basefolder = None

# Looking for the most recent simulation
if basefolder is None:
    res_dir = "res_data"
    basename = "Axisymmetric-Basel-NonLinearStiffness"

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
coor1D = np.array(mm["Coordinates"])[:, 0]

Nelts = len(coor1D) - 1

d_c = param["Friction coefficient"]["d_c"]
f_p = param["Friction coefficient"]["peak"]
f_r = param["Friction coefficient"]["residual"]
psi_p = param["Friction coefficient"]["peak dilatancy"]


# %% Postprocess, compare with analytical solution
# compute the radial coordinates of the mesh nodes.
x = coor1D

tt = np.array([res[j].time for j in range(len(res))])
# # find index of coordinate close to 1. meter
# ind=np.argmin(np.absolute(x-1.))

####################################################################
# plots and checks


####################################################################
# plots and checks
timestep = -2
num_pressure = res[timestep].pressure
# true_sol=pressure(x,tt[timestep],Qinj)


####################################################################
# %% pressure well
pw_t = np.array([res[j].pressure[0] for j in range(len(res))])

file = open("./SurfacePressure_Mpa.json")  # /local_dev/Basel
pressureData = np.array(json.load(file))
file.close()

# true_sol=pressure(x[1],tt,Qinj)
fig, ax = plt.subplots()
ax.plot(tt[:], pw_t[:], "-b", label="Numerical")
plt.xlabel("Time (s)")
plt.ylabel("Well pressure (Pa)")
plt.legend()
plt.savefig(
    os.path.join(figure_dir, "well_pressure_vs_time.png"), dpi=200, bbox_inches="tight"
)
# plt.show()
fig, ax = plt.subplots()
ax.plot(pressureData[:, 0], pressureData[:, 1], "-r", label="Data")
ax.plot(tt[:], 1e-6 * pw_t[:], "-b", label="Numerical")
plt.xlabel("Time (s)")
plt.ylabel("Well pressure (MPa)")
plt.legend()
# plt.xlim([0,30000])
plt.savefig(
    os.path.join(figure_dir, "well_pressure_comparison.png"),
    dpi=200,
    bbox_inches="tight",
)
# plt.show()
# %% ##### rupture radius

file = open("./SeismicRadius.json")
seismicRadius = np.array(json.load(file))
file.close()

nyi = np.array([res[i].Nyielded for i in range(len(res))])
cr_front = nyi * (coor1D[1] - coor1D[0])
fig, ax = plt.subplots()
ax.plot(tt, cr_front, "-b", label="Numerical")
ax.plot(seismicRadius[:90, 0], seismicRadius[:90, 1], "-r", label="Seismic data")
plt.xlabel("Time (s)")
plt.ylabel("Rupture radius (m)")
plt.xlim([0.0, 800000.0])
plt.legend()
plt.savefig(
    os.path.join(figure_dir, "rupture_radius_vs_time.png"), dpi=200, bbox_inches="tight"
)
# plt.show()

# %%
#### slip at origin

slip_0 = np.abs(np.array([res[i].DDs[0] for i in range(len(res))]))
slip_p_0 = np.abs(np.array([res[i].DDs_plastic[0] for i in range(len(res))]))

fig, ax = plt.subplots()
ax.plot(tt, slip_0, "-r", label="Total slip")
ax.plot(tt, slip_p_0, "-b", label="Plastic slip")
plt.xlabel("Time (s)")
plt.ylabel("Slip at injection point (m)")
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
ax.plot(tt, width_0, "-r", label="Total width")
ax.plot(tt, width_p_0, "-b", label="Plastic width")
plt.xlabel("Time (s)")
plt.ylabel("Width at injection point (m)")
plt.legend()
plt.savefig(
    os.path.join(figure_dir, "width_at_injection_point.png"),
    dpi=200,
    bbox_inches="tight",
)
# plt.show()

# %%
fig, ax = plt.subplots()
ax.plot(slip_0, width_0, "-k", label="Total")
ax.plot(slip_p_0, width_p_0, "-b", label="Plastic")
plt.xlabel("Slip at injection point (m)")
plt.ylabel("Width at injection point (m)")
plt.legend()
plt.savefig(os.path.join(figure_dir, "width_vs_slip.png"), dpi=200, bbox_inches="tight")
# plt.show()

# %%
### profiles
timestep = -50
sol_to_p = res[timestep]
x = coor1D
num_pressure = sol_to_p.pressure
fig, ax = plt.subplots()
# ax.plot(x[1:],true_sol[1:],'r')
ax.plot(x[1:400], num_pressure[1:400], "-b", label="Numerical")
plt.xlabel("r (m)")
plt.ylabel("Fluid pressure (Pa)")
plt.legend()
plt.savefig(
    os.path.join(figure_dir, "pressure_profile.png"), dpi=200, bbox_inches="tight"
)
# plt.show()

# %%

wh_o = 1.0
# width profile
w_profile = np.array(sol_to_p.DDs[1::2])
w_profile_p = np.array(sol_to_p.DDs_plastic[1::2])
fig, ax = plt.subplots()
# ax.plot(x[1:],true_sol[1:],'r')
ax.plot((x[1:] + x[0::-2]) / 2.0, w_profile[:], "-k", label="Total width")
ax.plot((x[1:] + x[0::-2]) / 2.0, w_profile_p[:], "-b", label="Plastic width")
plt.xlabel("r (m)")
plt.ylabel("Width (m)")
plt.legend()
plt.savefig(os.path.join(figure_dir, "width_profile.png"), dpi=200, bbox_inches="tight")
# plt.show()

fig, ax = plt.subplots()
Tav = 2.0e-15
ax.plot(
    (x[1:] + x[0::-2]) / 2.0,
    1 + (w_profile[:] ** 3) / (12 * Tav),
    "-k",
    label="Numerical",
)
plt.xlabel("r (m)")
plt.ylabel("Hydraulic transmissibility increase (-)")
plt.legend()
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
ax.plot((x[1:] + x[0::-2]) / 2.0, -slip_profile[:], "-k", label="Total slip")
ax.plot((x[1:] + x[0::-2]) / 2.0, -slip_profile_p[:], "-b", label="Plastic slip")
plt.xlabel("r (m)")
plt.ylabel("Slip (m)")
plt.legend()
plt.savefig(os.path.join(figure_dir, "slip_profile.png"), dpi=200, bbox_inches="tight")
# plt.show()

# dilatancy profile
fig, ax = plt.subplots()
app_dila_c = -sol_to_p.DDs_rate[1::2] / sol_to_p.DDs_rate[0::2]
ax.plot((x[1:] + x[0::-2]) / 2.0, app_dila_c, "-b", label="Numerical")
plt.xlabel("r (m)")
plt.ylabel("Width/Slip (-)")
plt.legend()
plt.savefig(
    os.path.join(figure_dir, "dilatancy_profile.png"), dpi=200, bbox_inches="tight"
)
# plt.show()

# %%
# effective tracitons profile
tau = sol_to_p.effective_tractions[0::2]
sig_n = sol_to_p.effective_tractions[1::2]

fig, ax = plt.subplots()
# ax.plot(x[1:],true_sol[1:],'r')
ax.plot((x[1:] + x[0::-2]) / 2.0, -tau[:], "-k", label="Shear traction")
ax.plot((x[1:] + x[0::-2]) / 2.0, -sig_n[:], "-b", label="Normal traction")
plt.xlabel("r (m)")
plt.ylabel("Effective tractions (Pa)")
plt.legend()
plt.savefig(
    os.path.join(figure_dir, "effective_tractions_profile.png"),
    dpi=200,
    bbox_inches="tight",
)
# plt.show()


fric_c = [linearEvolution(slip_profile_p[i], d_c, f_p, f_r) for i in range(Nelts)]
dila_c = [linearEvolution(slip_profile_p[i], d_c, psi_p, 0.0) for i in range(Nelts)]
fig, ax = plt.subplots()
# ax.plot(x[1:],true_sol[1:],'r')
ax.plot((x[1:] + x[0::-2]) / 2.0, fric_c, "-b", label="Friction coefficient")
ax.plot((x[1:] + x[0::-2]) / 2.0, dila_c, "-r", label="Dilatancy coefficient")
plt.xlabel("r (m)")
plt.ylabel("Friction coef. and dilatancy coef. (-)")
plt.legend()
plt.savefig(
    os.path.join(figure_dir, "friction_dilatancy_coefficients.png"),
    dpi=200,
    bbox_inches="tight",
)
# plt.show()

###
fig, ax = plt.subplots()
# ax.plot(x[1:],true_sol[1:],'r')
ax.plot((x[1:] + x[0::-2]) / 2.0, np.abs(tau) + fric_c * sig_n, "-b", label="Numerical")
# ax.plot((x[1:]+x[0::-2])/2., dila_c,'-b')
plt.xlabel("r (m)")
plt.ylabel("F_mc (Pa)")
plt.legend()
plt.savefig(
    os.path.join(figure_dir, "mohr_coulomb_function.png"), dpi=200, bbox_inches="tight"
)
# plt.show()

# %%

# POST-Process stats

newton_its = np.array([res[i].stats["iteration"] for i in range(len(res))])
elap_time = np.array([res[i].stats["elapsed_time"] for i in range(len(res))])
step_time = np.diff(elap_time)
step_time = np.insert(step_time, 0, step_time[0])
n_yield_step = np.array([res[i].Nyielded for i in range(len(res))])

# %%
nmvt_step = 0.0 * newton_its

for i in range(len(res)):
    for k in range(newton_its[i] - 1):
        nmvt_step[i] += res[i].stats["jacobian_stat_list"][k][
            "Total number of A11 matvect: "
        ]

# %%
fig, ax = plt.subplots()
ax.plot(n_yield_step, nmvt_step, ".", label="Numerical")
plt.xlabel("Number of yielded elements")
plt.ylabel("Matrix-vector products")
plt.legend()
plt.savefig(
    os.path.join(figure_dir, "matvec_vs_yielded_elements.png"),
    dpi=200,
    bbox_inches="tight",
)
# plt.show()
# %%
# %%
fig, ax = plt.subplots()
ax.plot(newton_its, nmvt_step, ".", label="Numerical")
plt.xlabel("Newton iterations")
plt.ylabel("Matrix-vector products")
plt.legend()
plt.savefig(
    os.path.join(figure_dir, "matvec_vs_newton_iterations.png"),
    dpi=200,
    bbox_inches="tight",
)
# plt.show()

# %%
fig, ax = plt.subplots()
ax.plot(nmvt_step, step_time, ".", label="Numerical")
plt.xlabel("Matrix-vector products")
plt.ylabel("Step time (s)")
plt.legend()
plt.savefig(
    os.path.join(figure_dir, "steptime_vs_matvec.png"), dpi=200, bbox_inches="tight"
)
# plt.show()
# %%
# %%
fig, ax = plt.subplots()
ax.plot(newton_its, step_time, ".", label="Numerical")
plt.xlabel("Newton iterations")
plt.ylabel("Step time (s)")
plt.legend()
plt.savefig(
    os.path.join(figure_dir, "steptime_vs_newton_iterations.png"),
    dpi=200,
    bbox_inches="tight",
)
# plt.show()

# %%
fig, ax = plt.subplots()
ax.plot(nmvt_step / newton_its, step_time / newton_its, ".", label="Numerical")
plt.xlabel("Average matvec per iteration")
plt.ylabel("Average time per iteration (s)")
plt.legend()
plt.savefig(
    os.path.join(figure_dir, "avg_time_vs_avg_matvec.png"), dpi=200, bbox_inches="tight"
)
# plt.show()
# %%
