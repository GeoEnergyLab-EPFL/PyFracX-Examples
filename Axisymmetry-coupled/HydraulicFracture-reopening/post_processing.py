# %%
from matplotlib import pyplot as plt
import numpy as np

import os
home = os.environ['HOME'] 

from pathlib import Path
figfolder = "." # @INPUT

file_path = os.path.realpath(__file__)

import sys
sys.path.append(os.path.dirname(file_path))

# %matplotlib inline
plt.rcParams['figure.figsize'] = [3.5, 3.5]
plt.rcParams['font.size'] = 12
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.1
plt.rcParams["savefig.dpi"] = 100

#%%
from my_utils import AxisymmHFShearSimulation

# %%
## Specify folder name of simulation
directory_path = os.path.dirname(os.path.abspath(__file__))

folder = directory_path+'/3DAxiSymmHF-16-01-2025-13-58-11/'
res = AxisymmHFShearSimulation(folder)


# %%

###
# HF Zero toughness Regime
###

# %%
lw = 2
ms = 4
plt.rcParams['figure.figsize'] = [7, 5]
plt.figure()
color_list = ["sr", ".b", "*g", "ok"]
j=0
for res in [res]:
    L, W, P = res.model.hf_scaling(res.tts)
    Ranal = res.model.hf_model.rupture_radius_at_time(res.tts)
    RsbyR = res.model.shear_by_open_ratio_axisym()
    Rsanal = RsbyR * Ranal
    skip = 5
    plt.plot(res.tts[::skip], res.open_front[::skip], "gs", label=r"$R(t)$", ms=ms)
    plt.plot(res.tts[::skip], res.shear_front[::skip], "mx", label=r"$R_s(t)$", ms=ms)

    plt.plot(
        res.tts[:], Ranal,
        "k--",
        lw=lw, label=r"M-vertex HF $\sim t^{4/9}$"
    )

    plt.plot(
        res.tts[:], Rsanal,
        "b--",
        lw=lw, label=r"Dugdale model"
    )
    j+=1

plt.semilogx()
plt.semilogy()
plt.xlabel(r"time [s]")
plt.ylabel(r"fronts radius [m]")
#plt.xlim([1, 2e2])
#plt.ylim([1e0, 3e1])
#plt.xticks([1e0,1e1,1e2], [r"$1$", r"$10$", r"$10^2$"])
#plt.yticks([1e0,1e1,2e1], [r"$1$", r"$10$", r"$20$"])
plt.legend()
# ax = plt.gca()
plt.legend(markerscale=2)
#plt.savefig(figfolder + "frontradiusvstime-mvertex.png")

# %% Opening profile
color_list = ["r", "b", "g", "k"]
plt.rcParams['figure.figsize'] = [7, 5]
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
        (np.array(res.soln[tind]["DDs"][1::2]) + res.model.w0 * 0 - res.model.sig0p / res.model.kn) / W,
        "-",
        c=color_list[i],
        ms=0.5,
        label=r"$t=%d$ [s]" % (time),
    )
    i += 1

R = res.model.hf_model.rupture_radius_at_time(time)
width_m = res.model.hf_model.opening_profile_at_time(res.col_pts, time) + res.model.w0 * 0
L, W, P = res.model.hf_scaling(time)
plt.plot(res.colPts / R, width_m / W, "--", label="M-vertex HF")
plt.xlabel(r"$r  / R(t)$")
plt.axvline(x=1, c="k", ls="--", lw=0.5)
plt.ylabel(r"$w(r / R(t), t)/ W(t)$")
# Shrink current axis by 20%
plt.xlim([0, 1.2])
plt.ylim([0, 1.3])
plt.legend(loc="upper right")
#plt.savefig(figfolder + "openingprofile-mvertex.png")

# %% Net Pressure / Stress
color_list = ["r", "b", "g", "k"]
plt.rcParams['figure.figsize'] = [7, 5]
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
#plt.savefig(figfolder + "stress-mvertex.png")

#
# Diffusuon late regime
#
# %%
lw = 1
ms = 4
plt.rcParams['figure.figsize'] = [7, 5]
plt.figure()
color_list = ["sr", ".b", "*g", "ok"]
j=0
# for res in [res1, res2, res3, res4]:
for res in [res]:
    L, W, P = res.model.hf_scaling(res.tts)
    Ranal = res.model.hf_model.rupture_radius_at_time(res.tts)
    RsbyR = res.model.shear_by_open_ratio_axisym()
    Rsanal = RsbyR * Ranal
    skip = 2
    plt.plot(res.tts[::skip], res.open_front[::skip], "gs", label=r"$R(t)$", ms=ms)
    plt.plot(res.tts[::skip], res.shear_front[::skip], "mx", label=r"$R_s(t)$", ms=ms)
    plt.plot(res.tts, 1e-1 * res.tts**(0.5), "b--",lw=lw, label=r"$\sim t^{1/2}$")

    plt.plot(
        res.tts[:], Ranal,
        "k--",
        lw=lw, label=r"M-vertex HF $\sim t^{4/9}$"
    )
    j+=1

plt.semilogx()
plt.semilogy()
plt.xlabel(r"time [s]")
plt.ylabel(r"fronts radius [m]")
# plt.xlim([1, 2e2])
plt.xlim([1e0, 3e3])
plt.ylim([1e-2,1e1])
plt.xticks([1e0,1e1,1e2,1e3], [r"$1$", r"$10$", r"$10^2$", r"$10^3$"])
plt.yticks([1e-2, 1e-1,1e0,1e1], [r"$0.01$", r"$0.1$", r"$1$", r"$10$"])
plt.legend()
# ax = plt.gca()
plt.legend(markerscale=2)
plt.savefig(figfolder + "frontradiusvstime-diffusion.png")

# %% Opening profile
color_list = ["r", "b", "g", "k"]
plt.rcParams['figure.figsize'] = [7, 5]
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
        (np.array(res.soln[tind]["DDs"][1::2]) + res.model.w0 * 0 - res.model.sig0p / res.model.kn) / W,
        "-",
        c=color_list[i],
        ms=0.5,
        label=r"$t=%d$ [s]" % (time),
    )
    i += 1

# plt.xlim([-0.01, 2.2])
R = res.model.hf_model.rupture_radius_at_time(time)
width_m = res.model.hf_model.opening_profile_at_time(res.col_pts, time) + res.model.w0 * 0
L, W, P = res.model.hf_scaling(time)
plt.plot(res.colPts / R, width_m / W, "--", label="M-vertex HF")
plt.xlabel(r"$r  / R(t)$")
plt.axvline(x=1, c="k", ls="--", lw=0.5)
plt.ylabel(r"$w(r / R(t), t)/ W(t)$")
plt.xlim([0, 1.2])
plt.ylim([0, 1.3])
plt.legend(loc="upper right")
plt.savefig(figfolder + "openingprofile-diffusion.png")

# %% Net Pressure / Stress
color_list = ["r", "b", "g", "k"]
plt.rcParams['figure.figsize'] = [7, 5]
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
plt.savefig(figfolder + "stress-diffusion.png")


# %%
