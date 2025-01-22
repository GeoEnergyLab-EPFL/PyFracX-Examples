#
# This file is part of PyFracX-Examples
#
# POSTPROCESSING FILES FOR
# Fluid injection at constant rate into a frictional fault in 3D (modelled as axisymmetric problem).
# Coupled simulation.
# Reference results from Sáez & Lecampion (2022) 
#
# %%+
#Importing the necessary python libraries and managing the python path

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path
import json
from ReferenceSolutions.FDFR.frictional_ruptures_3D_constant_friction import *

from mesh.usmesh import usmesh

from utils.json_dict_dataclass_utils import *
from hm.HMFsolver import HMFSolution
from scipy import linalg

#%% loading numerical results 
# here you need to change basefolder to the folder name of your simulation
basefolder = './AxiSymm-ctRate-ctFriction-coupled-marpress-10-08-2024-20:21:23/'

#'./2D-ctP-LinearWeakening-MarginallyPressurized-21-07-2024-16-42-57/'
basename = "AxiSymm-ctRate-ctFriction-coupled-marpress"

# we load n_steps - please adjst as per your simulation 
res = []
n_steps = 38

for i in range(n_steps):
    tt = from_dict_to_dataclass(HMFSolution,json_read(basefolder+basename+'-'+str(i+1)))
    res.append(tt)

param = json_read(basefolder+'Parameters')

alpha_hyd = param["Flow"]["Hydraulic diffusivity"]
cond_hyd = param["Flow"]["Hydraulic conductivity"]
Qinj = param["Injection"]["Injection rate"]
wh = param["Hydraulic aperture"]
T = param["T parameter"]
f = param["Friction coefficient"]
E = param["Elasticity"]["Young"]
nu = param["Elasticity"]["Nu"]
G = E / (2 * (1 + nu))

mm =  json_read(basefolder+'Mesh')
Nelts = mm["Nelts"]
Nnodes = Nelts+1
coor1D = np.array(mm["Coordinates"])[:,0]
colPts = (coor1D[1:] + coor1D[0:-1]) / 2.0  # collocation points for P0

#radial coordinates of the nodes
r = np.array([linalg.norm(coor1D[i]) for i in range(Nnodes)])

#radial coordinates of the collocation points
r_col = np.array([linalg.norm(colPts[i]) for i in range(Nelts)])
#%%  
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
plt.plot(t_, R, "r")
plt.plot(tts, rmax, ".")
plt.semilogx()
plt.semilogy()
plt.xlabel("Time (s)")
plt.ylabel("Rupture radius (m)")
plt.legend(["Analytical solution", "Numerics"])
plt.grid()
plt.show()

plt.figure()
plt.plot(tts, lam * np.ones(len(tts)), "-r")
plt.plot(tts, rmax / np.sqrt(4 * alpha_hyd * tts), "ko")
plt.semilogx()
plt.grid()
plt.xlabel("Time (s)")
plt.ylabel(r"$\lambda$ = Rupture radius / $\sqrt{4 \alpha t}$")
plt.gca().legend(["Analytical solution", "Numerics"])
plt.title(r"T = %.3f, $\lambda = %.2f$" % (T, lam))
print(
    "Relative error from analytical prediction in Percentage : ",
    (np.abs(lam - rmax[-1] / np.sqrt(4 * alpha_hyd * tts[-1])) / lam) * 100.0,
)

# "Asymptotic solutions for self-similar fault slip induced by fluid injection at constant rate"
# by Viesca (2024)

import scipy.special

dp = Qinj/(4*np.pi*cond_hyd*wh)
alpha_new = 4 * alpha_hyd
lam = lambda_analytical(T)
rmax_analytical = lam * np.sqrt(4 * alpha_hyd * tts[-1])
t = tts[-1]

fig, ax = plt.subplots()
ax.plot(r_col, 
        -np.array(res[-1].DDs_plastic)[0:-1:2], 
        "--r", 
        label = "Numerical")

##########    ATTENTION    ##### MODIFY THE FUNCTION USED HERE #########
ax.plot(
    r_col,
    complete_slip_profile_mp(G, f, dp, alpha_hyd, T, tts[-1], lambda_analytical, r_col),
    "-k",
    ms=1.2,
    label="Analytical",
#for critically stressed case: use the function complete_slip_profile_cs 
# for marginally stressed case: use the function complete_slip_profile_mp)
)

plt.xlabel("r")
plt.ylabel("slip")
plt.grid()
plt.legend()

#%%

# VISULISATION OF THE PLASTIC AND ELASTIC DISPLACEMENT DISCONTINUITIES

jj=-1
fig, ax = plt.subplots()
ax.plot(r_col,-np.array(res[jj].DDs_plastic[0::2]))
plt.xlabel("x (m)")
plt.ylabel("Plastic slip profile (m) at time last step")
plt.grid()
plt.show()

fig, ax = plt.subplots()
ax.plot(r_col,-(np.array(res[jj].DDs)[0::2]-np.array(res[jj].DDs_plastic)[0::2]))
plt.xlabel("x (m)")
plt.ylabel("elastic slip profile (m) at time last step")
plt.grid()
plt.show()

fig, ax = plt.subplots()
ax.plot(r_col, res[jj].DDs_plastic[1::2])
plt.xlabel("x (m)")
plt.ylabel("Plastic opening profile (m) at time last step")
plt.grid()
plt.show()

fig, ax = plt.subplots()
ax.plot(r_col,-(np.array(res[jj].DDs)[1::2]-np.array(res[jj].DDs_plastic)[1::2]))
plt.xlabel("x (m)")
plt.ylabel("elastic Opg (m) at time last step")
plt.grid()
plt.show()

#%%
#
#
# VISULISATION OF THE STRESS PROFILES
#
#
fig, ax = plt.subplots()
ax.plot(r_col,res[jj].effective_tractions[1::2],'.-')
plt.xlabel("x (m)")
plt.ylabel("Effective normal stress (Pa) at time last step")
plt.grid()
plt.show()

fig, ax = plt.subplots()
ax.plot(r_col,res[jj].effective_tractions[0::2],'.-')
plt.xlabel("x (m)")
plt.ylabel("Shear stress (Pa) at time last step")
plt.grid()
plt.show()

fig, ax = plt.subplots()
ax.plot(r_col,res[jj].yieldF)
plt.xlabel("x (m)")
plt.ylabel("Yield function (Pa) at time last step")
plt.grid()
plt.show()


# %%
