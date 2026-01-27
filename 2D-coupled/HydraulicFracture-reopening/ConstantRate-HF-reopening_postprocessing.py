#
# This file is part of PyFracX.
#
# Created by Brice Lecampion on 9.12.2021
# Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2021.  All rights reserved.
# See the LICENSE.TXT file for more details.
#
#
# %% Imports
from datetime import datetime
import re
import os
import sys
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "grid"])

from pyfracx.mesh.usmesh import UnstructuredMesh
from pyfracx.utils.json_dict_dataclass_utils import *
from pyfracx.hm.HMFsolver import HMFSolution
from pyfracx.loads.Injection import *

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from ReferenceSolutions.HF.KGDHFsolutions import *

# %% loading numerical results
# here you need to change basefolder to the folder name of your simulation
# set it to None to look for the most recent simulation
basefolder = None
basename = "2D-ConstantRate-HF-reopening"

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

mesh = json_read(os.path.join(basefolder, "Mesh"))
Nelts = mesh["Nelts"]
coor1D = np.array(mesh["Coordinates"])[:, 0]
mesh = UnstructuredMesh(2, np.array(mesh["Coordinates"]), np.array(mesh["Connectivity"]), 0)

Qinj = param["Injection"]["Constant Rate"]
wh_o = param["Flow"]["who"]
fluid_visc = param["Flow"]["Viscosity"]
c_f = param["Flow"]["Compressibility"]
sig_o_p = param["Initial stress"][1]
YoungM = param["Elasticity"]["Young"]
nu = param["Elasticity"]["Nu"]
E_prime = YoungM / (1 - nu * nu)


k_frac = wh_o**2 / 12
alpha_h = k_frac / (fluid_visc * c_f)

the_inj = Injection(np.array([0.0, 0.0]), np.array([[0.0, Qinj]]), "Rate")

# %% Setup figure directory
figure_dir = f"figures_{basename}"
os.makedirs(figure_dir, exist_ok=True)

# %%

n_k = []
n_k_nit = []
for i0 in range(len(res)):
    k = 0
    nit = len(res[i0].stats["jacobian_stat_list"])
    for i in range(nit):
        k = (
            k
            + res[i0].stats["jacobian_stat_list"][i]["Total number of A11 matvect: "]
            + nit
        )
    n_k.append(k)
    n_k_nit.append(k / nit)


# %%---------------------------------------------
# post-process to estimate crack front position
# sol
tts = np.array([res[i].time for i in range(len(res))])
nyi = np.array([res[i].Nyielded for i in range(len(res))])
cr_front = nyi / 2.0 * (coor1D[1] - coor1D[0])

kgd_m = KGD_Mvertex(Ep=E_prime, Qo=Qinj, mu=fluid_visc)

# analytical sol  M vertex
t_ = np.linspace(0.01, tts.max(), 1000)
y_ = kgd_m.half_length(t_)

fig, ax = plt.subplots()
ax.loglog(t_, y_, "r", label="Analytical solution")
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

half_length_true = kgd_m.half_length(tts)
rel_error = np.abs(cr_front - half_length_true) / half_length_true

fig, ax = plt.subplots()
ax.loglog(tts, rel_error, ".")
plt.xlabel("Time (s)")
plt.ylabel("Rel. error fracture length (-)")
plt.savefig(
    os.path.join(figure_dir, "relative_error_fracture_length.png"),
    dpi=200,
    bbox_inches="tight",
)
# plt.show()
#

half_length_true = kgd_m.half_length(tts)
lamb_ = np.abs(cr_front / half_length_true)

fig, ax = plt.subplots()
ax.plot(tts, lamb_, ".", label="Numerical")
ax.plot(tts, tts * 0 + 1.236, ".", label="Reference value")

plt.xlabel("Time (s)")
plt.ylabel(r"$\ell_s / \ell$ (-)")
ax.legend()
plt.savefig(os.path.join(figure_dir, "length_ratio.png"), dpi=200, bbox_inches="tight")
# plt.show()
#

pp_flow = (Qinj / wh_o) * np.sqrt(4.0 * alpha_h * tts) / (np.sqrt(4 * np.pi) * k_frac)
ppi = np.array([res[i].pressure[the_inj.locate_in_mesh(mesh)] for i in range(len(res))])
fig, ax = plt.subplots()
ax.loglog(tts, ppi * 1e-6, ".")
# ax.loglog(tts,pp_flow*1e-6,'.')
plt.xlabel("Time (s)")
plt.ylabel("Fluid pressure at injection (MPa)")
plt.savefig(
    os.path.join(figure_dir, "fluid_pressure_at_injection.png"),
    dpi=200,
    bbox_inches="tight",
)
# plt.show()


# ---- PLOTTING PROFILES ----
jj = len(res) - 1  # choosing the step to profile

xi_list = np.linspace(-0.9999, 0.9999, 60)
net_p_m = kgd_m.pressure(res[jj].time, np.abs(xi_list))
width_m = kgd_m.width(res[jj].time, np.abs(xi_list))
x_list = (kgd_m.half_length(res[jj].time)) * xi_list

fig, ax = plt.subplots()
# ax.plot(colPts,solp)
ax.plot(coor1D[:], (res[jj].pressure)[:], ".", label="Numerical")  # 900:1100
ax.plot(x_list, net_p_m + sig_o_p, ".r-", label="Analytical")  # add the initial sigma_o
plt.xlabel("x (m)")
plt.ylabel("Fluid pressure (Pa)")
ax.legend()
plt.savefig(
    os.path.join(figure_dir, "fluid_pressure_profile.png"), dpi=200, bbox_inches="tight"
)
# plt.show()

fig, ax = plt.subplots()
ax.plot((coor1D[1:] + coor1D[0:-1]) / 2.0, res[jj].DDs[1::2], label="Numerical")
ax.plot(x_list, width_m, ".r-", label="Analytical")
plt.xlabel("x (m)")
plt.ylabel("Fracture opening (m)")
ax.legend()
plt.savefig(
    os.path.join(figure_dir, "fracture_opening_profile.png"),
    dpi=200,
    bbox_inches="tight",
)
# plt.show()

#
# ax.plot(coor1D[:],(res[jj].res_flow)[:] ,'.-')
# plt.xlabel("x (M)")
# plt.ylabel(" res flow")
# # plt.show()
#
# fig, ax = plt.subplots()
# ax.plot((coor1D[1:]+coor1D[0:-1])/2.,res[jj].res_mech[1::2])
# plt.xlabel("x (m)")
# plt.ylabel("res mech ")
# # plt.show()


fig, ax = plt.subplots()
ax.plot((coor1D[1:] + coor1D[0:-1]) / 2.0, -(np.array(res[jj].DDs_plastic[0::2])))
plt.xlabel("x (m)")
plt.ylabel("Plastic slip (m)")
plt.savefig(
    os.path.join(figure_dir, "plastic_slip_profile.png"), dpi=200, bbox_inches="tight"
)
# plt.show()


fig, ax = plt.subplots()
ax.plot(
    (coor1D[1:] + coor1D[0:-1]) / 2.0,
    -(np.array(res[jj].DDs_plastic[0::2]) - np.array(res[jj - 1].DDs_plastic[0::2]))
    / res[jj].timestep,
)
plt.xlabel("x (m)")
plt.ylabel("Plastic slip rate (m/s)")
plt.savefig(
    os.path.join(figure_dir, "plastic_slip_rate_profile.png"),
    dpi=200,
    bbox_inches="tight",
)
# plt.show()


fig, ax = plt.subplots()
ax.plot(
    (coor1D[1:] + coor1D[0:-1]) / 2.0,
    -(np.array(res[jj].DDs[0::2]) - np.array(res[jj].DDs_plastic[0::2])),
)
plt.xlabel("x (m)")
plt.ylabel("Elastic slip (m)")
plt.savefig(
    os.path.join(figure_dir, "elastic_slip_profile.png"), dpi=200, bbox_inches="tight"
)
# plt.show()

# fig, ax = plt.subplots()
# ax.plot((coor1D[1:]+coor1D[0:-1])/2.,-(res[jj].DDs[0::2]-res[jj].DDs_plastic[0::2]))
# plt.xlabel("x (m)")
# plt.ylabel("elastic slip (m)")
# # plt.show()

# fig, ax = plt.subplots()
# ax.plot((coor1D[1:]+coor1D[0:-1])/2.,-res[jj].DDs[0::2])
# plt.xlabel("x (m)")
# plt.ylabel("Total slip (m)")
# # plt.show()

fig, ax = plt.subplots()
ax.plot(
    (coor1D[1:] + coor1D[0:-1]) / 2.0,
    (np.array(res[jj].DDs[1::2]) - np.array(res[jj].DDs_plastic[1::2])),
)
plt.xlabel("x (m)")
plt.ylabel("Elastic opening (m)")
plt.savefig(
    os.path.join(figure_dir, "elastic_opening_profile.png"),
    dpi=200,
    bbox_inches="tight",
)
# plt.show()


fig, ax = plt.subplots()
ax.plot((coor1D[1:] + coor1D[0:-1]) / 2.0, (np.array(res[jj].DDs[1::2]) + wh_o) ** 3)
ax.set_yscale("log")
plt.xlabel("x (m)")
plt.ylabel("Flow transmissivity (m³)")
plt.savefig(
    os.path.join(figure_dir, "flow_transmissivity.png"), dpi=200, bbox_inches="tight"
)
# plt.show()


fig, ax = plt.subplots()
ax.plot((coor1D[1:] + coor1D[0:-1]) / 2.0, res[jj].effective_tractions[1::2], ".")
plt.xlabel("x (m)")
plt.ylabel("Effective normal stress (Pa)")
plt.savefig(
    os.path.join(figure_dir, "effective_normal_stress.png"),
    dpi=200,
    bbox_inches="tight",
)
# plt.show()


fig, ax = plt.subplots()
ax.plot((coor1D[1:] + coor1D[0:-1]) / 2.0, res[jj].effective_tractions[0::2], ".")
plt.xlabel("x (m)")
plt.ylabel("Shear stress (Pa)")
plt.savefig(os.path.join(figure_dir, "shear_stress.png"), dpi=200, bbox_inches="tight")
# plt.show()


fig, ax = plt.subplots()
ax.plot((coor1D[1:] + coor1D[0:-1]) / 2.0, res[jj].yieldF, label="Yield function")
ax.plot(
    (coor1D[1:] + coor1D[0:-1]) / 2.0,
    res[jj].effective_tractions[1::2],
    ".",
    label="Effective normal stress",
)
ax.plot(
    (coor1D[1:] + coor1D[0:-1]) / 2.0,
    res[jj].effective_tractions[0::2],
    ".",
    label="Shear stress",
)

plt.xlabel("x (m)")
plt.ylabel("Stress (Pa)")
ax.legend()
plt.savefig(
    os.path.join(figure_dir, "yield_function_and_stresses.png"),
    dpi=200,
    bbox_inches="tight",
)
# plt.show()


# %%
