# %%
from matplotlib import pyplot as plt
import numpy as np
import re
import os
from pathlib import Path
import sys
from datetime import datetime
import scienceplots

plt.style.use(["science", "grid"])

file_path = os.path.realpath(__file__)
sys.path.append(os.path.dirname(file_path))
from my_utils import AxisymmHFShearSimulation


# %matplotlib inline
plt.rcParams["figure.figsize"] = [3.5, 3.5]
plt.rcParams["font.size"] = 12
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.1
plt.rcParams["savefig.dpi"] = 100

# %%

# %%
# here you need to change basefolder to the folder name of your simulation
# set it to None to look for the most recent simulation
basefolder = None
basename = "3DAxiSymmHF"

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


res = AxisymmHFShearSimulation(basefolder)

figure_dir = f"figures_{basename}"
os.makedirs(figure_dir, exist_ok=True)


# %%

###
# HF Zero toughness Regime
###

# %%
lw = 2
ms = 4
plt.rcParams["figure.figsize"] = [7, 5]
plt.figure()
color_list = ["sr", ".b", "*g", "ok"]
j = 0
for res in [res]:
    L, W, P = res.model.hf_scaling(res.tts)
    Ranal = res.model.hf_model.rupture_radius_at_time(res.tts)
    RsbyR = res.model.shear_by_open_ratio_axisym()
    Rsanal = RsbyR * Ranal
    skip = 5
    plt.plot(res.tts[::skip], res.open_front[::skip], "gs", label=r"$R(t)$", ms=ms)
    plt.plot(res.tts[::skip], res.shear_front[::skip], "mx", label=r"$R_s(t)$", ms=ms)

    plt.plot(res.tts[:], Ranal, "k--", lw=lw, label=r"M-vertex HF $\sim t^{4/9}$")

    plt.plot(res.tts[:], Rsanal, "b--", lw=lw, label=r"Dugdale model")
    j += 1

plt.semilogx()
plt.semilogy()
plt.xlabel(r"time [s]")
plt.ylabel(r"fronts radius [m]")
# plt.xlim([1, 2e2])
# plt.ylim([1e0, 3e1])
# plt.xticks([1e0,1e1,1e2], [r"$1$", r"$10$", r"$10^2$"])
# plt.yticks([1e0,1e1,2e1], [r"$1$", r"$10$", r"$20$"])
plt.legend()
# ax = plt.gca()
plt.legend(markerscale=2)
plt.savefig(
    os.path.join(figure_dir, "fracture_radius.png"), dpi=200, bbox_inches="tight"
)
# plt.show()

# %% Opening profile
color_list = ["r", "b", "g", "k"]
plt.rcParams["figure.figsize"] = [7, 5]
plt.figure()
i = 0

tlist = [-1, -2, -3]
for tind in tlist:
    time = res.soln[tind]["time"]
    l = res.open_front[tind]
    print(l)
    xi = res.colPts[:] / l
    L, W, P = res.model.hf_scaling(time)
    plt.plot(
        xi,
        (
            np.array(res.soln[tind]["DDs"][1::2])
            + res.model.w0 * 0
            - res.model.sig0p / res.model.kn
        )
        / W,
        "-",
        c=color_list[i],
        ms=0.5,
        label=r"$t=%d$ [s]" % (time),
    )
    i += 1

R = res.model.hf_model.rupture_radius_at_time(time)
width_m = (
    res.model.hf_model.opening_profile_at_time(res.col_pts, time) + res.model.w0 * 0
)
L, W, P = res.model.hf_scaling(time)
plt.plot(res.colPts / R, width_m / W, "--", label="M-vertex HF")
plt.xlabel(r"$r  / R(t)$")
plt.axvline(x=1, c="k", ls="--", lw=0.5)
plt.ylabel(r"$w(r / R(t), t)/ W(t)$")
# Shrink current axis by 20%
plt.xlim([0, 1.2])
plt.ylim([0, 1.3])
plt.legend(loc="upper right")
plt.savefig(os.path.join(figure_dir, "opening.png"), dpi=200, bbox_inches="tight")
# plt.show()

# %% Net Pressure / Stress
color_list = ["r", "b", "g", "k"]
plt.rcParams["figure.figsize"] = [7, 5]
plt.figure()
i = 0
for tind in tlist:
    time = res.soln[tind]["time"]
    L, W, P = res.model.hf_scaling(time)
    pres_colpts = np.array(
        [
            0.5 * (res.soln[tind]["pressure"][i] + res.soln[tind]["pressure"][i + 1])
            for i in range(res.colPts.shape[0])
        ]
    )
    stress = -np.array(res.soln[tind]["effective_tractions"][1::2]) + pres_colpts[:]
    l = res.open_front[tind]
    ls = res.shear_front[tind]
    plt.plot(
        res.colPts[:] / l,
        (stress - res.model.sig0) / P,
        "-",
        c=color_list[i],
        ms=0.5,
        label=r"$t=%d$ [s]" % (time),
    )
    i += 1

plt.xlim([0, 2])
# plt.semilogx()
plt.ylim([-2, 1.5])
net_press_anal = res.model.hf_model.pressure_profile_at_time(res.col_pts, time)
R = res.model.hf_model.rupture_radius_at_time(time)
L, W, P = res.model.hf_scaling(time)
plt.axhline(y=0, c="k", ls="--", lw=0.5)
fl = res.colPts[:] <= R
plt.plot(res.colPts[fl] / R, net_press_anal[fl] / P, "--", label="M-vertex HF")
plt.ylabel(r"$ (\sigma(r / R(t), t) - \sigma_o)/ P(t) $")
plt.xlabel(r"$r / R(t) $")
plt.axvline(x=1, c="k", ls="--", lw=0.5)
# Shrink current axis by 20%
plt.legend(loc="upper right")
plt.savefig(os.path.join(figure_dir, "net_pressure.png"), dpi=200, bbox_inches="tight")
# plt.show()

#
# Diffusuon late regime
#
# %%
lw = 1
ms = 4
plt.rcParams["figure.figsize"] = [7, 5]
plt.figure()
color_list = ["sr", ".b", "*g", "ok"]
j = 0
# for res in [res1, res2, res3, res4]:
for res in [res]:
    L, W, P = res.model.hf_scaling(res.tts)
    Ranal = res.model.hf_model.rupture_radius_at_time(res.tts)
    RsbyR = res.model.shear_by_open_ratio_axisym()
    Rsanal = RsbyR * Ranal
    skip = 2
    plt.plot(res.tts[::skip], res.open_front[::skip], "gs", label=r"$R(t)$", ms=ms)
    plt.plot(res.tts[::skip], res.shear_front[::skip], "mx", label=r"$R_s(t)$", ms=ms)
    plt.plot(res.tts, 1e-1 * res.tts ** (0.5), "b--", lw=lw, label=r"$\sim t^{1/2}$")

    plt.plot(res.tts[:], Ranal, "k--", lw=lw, label=r"M-vertex HF $\sim t^{4/9}$")
    j += 1

plt.semilogx()
plt.semilogy()
plt.xlabel(r"time [s]")
plt.ylabel(r"fronts radius [m]")
# plt.xlim([1, 2e2])
plt.xlim([1e0, 3e3])
plt.ylim([1e-2, 1e1])
plt.xticks([1e0, 1e1, 1e2, 1e3], [r"$1$", r"$10$", r"$10^2$", r"$10^3$"])
plt.yticks([1e-2, 1e-1, 1e0, 1e1], [r"$0.01$", r"$0.1$", r"$1$", r"$10$"])
plt.legend()
# ax = plt.gca()
plt.legend(markerscale=2)
# plt.show()

# %% Opening profile
color_list = ["r", "b", "g", "k"]
plt.rcParams["figure.figsize"] = [7, 5]
plt.figure()
i = 0
tlist = [-18, -25, -30]
for tind in tlist:
    time = res.soln[tind]["time"]
    l = res.open_front[tind]
    print(l)
    xi = res.colPts[:] / l
    L, W, P = res.model.hf_scaling(time)
    plt.plot(
        xi,
        (
            np.array(res.soln[tind]["DDs"][1::2])
            + res.model.w0 * 0
            - res.model.sig0p / res.model.kn
        )
        / W,
        "-",
        c=color_list[i],
        ms=0.5,
        label=r"$t=%d$ [s]" % (time),
    )
    i += 1

# plt.xlim([-0.01, 2.2])
R = res.model.hf_model.rupture_radius_at_time(time)
width_m = (
    res.model.hf_model.opening_profile_at_time(res.col_pts, time) + res.model.w0 * 0
)
L, W, P = res.model.hf_scaling(time)
plt.plot(res.colPts / R, width_m / W, "--", label="M-vertex HF")
plt.xlabel(r"$r  / R(t)$")
plt.axvline(x=1, c="k", ls="--", lw=0.5)
plt.ylabel(r"$w(r / R(t), t)/ W(t)$")
plt.xlim([0, 1.2])
plt.ylim([0, 1.3])
plt.legend(loc="upper right")
# plt.show()

# %% Net Pressure / Stress
color_list = ["r", "b", "g", "k"]
plt.rcParams["figure.figsize"] = [7, 5]
plt.figure()
i = 0
for tind in tlist:
    time = res.soln[tind]["time"]
    L, W, P = res.model.hf_scaling(time)
    pres_colpts = np.array(
        [
            0.5 * (res.soln[tind]["pressure"][i] + res.soln[tind]["pressure"][i + 1])
            for i in range(res.colPts.shape[0])
        ]
    )
    stress = -np.array(res.soln[tind]["effective_tractions"][1::2]) + pres_colpts[:]
    l = res.open_front[tind]
    ls = res.shear_front[tind]
    plt.plot(
        res.colPts[:] / l,
        (stress - res.model.sig0) / P,
        "-",
        c=color_list[i],
        ms=0.5,
        label=r"$t=%d$ [s]" % (time),
    )
    # plt.axvline(x=cohesive_length(time)  / lstar + 1, c=color_list[i], ls="--", lw=0.5)
    i += 1

plt.xlim([0, 2])
plt.ylim([-2, 1.5])
net_press_anal = res.model.hf_model.pressure_profile_at_time(res.col_pts, time)
R = res.model.hf_model.rupture_radius_at_time(time)
L, W, P = res.model.hf_scaling(time)
# tlist = np.int_(np.linspace(-1, -150, 4))
# xi_list = np.linspace(0 - 1e-
plt.axhline(y=0, c="k", ls="--", lw=0.5)
fl = res.colPts[:] <= R
plt.plot(res.colPts[fl] / R, net_press_anal[fl] / P, "--", label="M-vertex HF")
plt.ylabel(r"$ (\sigma(r / R(t), t) - \sigma_o)/ P(t) $")
plt.xlabel(r"$r / R(t) $")
plt.axvline(x=1, c="k", ls="--", lw=0.5)
# Shrink current axis by 20%
plt.legend(loc="upper right")
# plt.show()

# %%
