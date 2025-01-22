#
# This file is part of PyFracX-Examples
#
# POSTPROCESSING FILES FOR
# Fluid injection at constant rate into a slip weakening frictional fault in 3D (modelled as axisymmetric problem).
# Reference results from Sáez & Lecampion (2023) 
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

from ReferenceSolutions.FDFR.frictional_ruptures_3D_constant_friction import *

from mesh.usmesh import usmesh

from utils.json_dict_dataclass_utils import *
from hm.HMFsolver import HMFSolution
from scipy import linalg

#%% loading numerical results 
# here you need to change basefolder to the folder name of your simulation
basefolder = '/home/sarma/Project_PhD/PyFracX-Examples/Axisymmetry-uncoupled/Constant-Friction-Constant-Permeability/AxiSymm-ctRate-NumFlow-lwfric-S_0.6-P_0.035-19-08-2024-21:08:40/'
#'./2D-ctP-LinearWeakening-MarginallyPressurized-21-07-2024-16-42-57/'
basename = "AxiSymm-ctRate-NumFlow-lwfric-S_0.6-P_0.035"

# we load n_steps - please adjst as per your simulation 
res = []
n_steps = 98

for i in range(n_steps):
    tt = from_dict_to_dataclass(HMFSolution,json_read(basefolder+basename+'-'+str((i+1)*10)))
    res.append(tt)

param = json_read(basefolder+'Parameters')

#%% 
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
f_r = F*f_p
E = param["Elasticity"]["Young"]
nu = param["Elasticity"]["Nu"]
G = E / (2 * (1 + nu))
Rw = param["Rupture length scale"]
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
import json

arg_fric_mod = 'lin' # 'lin' or 'exp' based on the friction model used

# Construct the file path (modify it according to your file structure)

base_path = f'/home/sarma/Project_PhD/PyFracX-Examples/ReferenceSolutions/FDFR/SlipWeakeningBenchmarks/{fric_model}/'
file_name = f'P{P}/S_{S}_P_{P}_F_{F}_{arg_fric_mod}_QD_pz50_L15_'
file_path = base_path + file_name

# Open and load the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)

# Example extraction of 'tlistNorm' and 'RfNorm' from the first entry (adjust according to your JSON structure)
tlistNorm = data['tlistNorm']
RfNorm = data['RfNorm']

#%%
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

x_cor = np.sqrt(4*alpha_hyd*tts)
x_cor = x_cor/Rw
y_cor = rmax/Rw

plt.figure()
plt.plot(x_cor, y_cor,'-.r', label = "Numerics")
plt.plot(tlistNorm, RfNorm, '-k', label = "Benchmark")
plt.xlabel(r"$\sqrt{4 \alpha t}/R_w$")
plt.ylabel(r"$R/R_w$")
plt.text(60, 8, f'(S = {S}, F = {F}, P = {P})', fontsize=12, ha='right')
plt.legend()
plt.grid()
plt.show()
#%%

# VISULISATION OF THE PLASTIC AND ELASTIC DISPLACEMENT DISCONTINUITIES
# slip plot

jj = -1
fig, ax = plt.subplots()
ax.plot(colPts, -np.array(res[jj].DDs[0:-1:2]), ".")
# ax.plot(coor1D[:],p_anal,'-g')
plt.xlabel("r (m)")
plt.ylabel(" slip ")
plt.title("Total slip at t = %.3f secs" % (res[-1].time))
plt.show()

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
