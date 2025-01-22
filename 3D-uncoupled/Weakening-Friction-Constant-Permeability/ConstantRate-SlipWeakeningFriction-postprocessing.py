#%%
#Importing the necessary python libraries and managing the python path
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path
import scipy
import matplotlib
from mesh.usmesh import usmesh

from utils.json_dict_dataclass_utils import *
from hm.HMFsolver import HMFSolution
from scipy import linalg

from ReferenceSolutions.FDFR.frictional_ruptures_3D_constant_friction import *
#%% loading numerical results 
# here you need to change basefolder to the folder name of your simulation
basefolder = '/home/sarma/Project_PhD/PyFracX-Examples/3D-uncoupled/3D-lwfric-oneway-S-0.6-P-0.05-20-08-2024-15:04:36/'
basename = "3D-lwfric-oneway-S-0.6-P-0.05"

# we load n_steps - please adjst as per your simulation 
res = []
n_steps = 95

for i in range(n_steps):
    tt = from_dict_to_dataclass(HMFSolution,json_read(basefolder+basename+'-'+str((i+1)*5)))
    res.append(tt)

param = json_read(basefolder+'Parameters')
#%%
alpha_hyd = param["Flow"]["Hydraulic diffusivity"]
cond_hyd = param["Flow"]["Hydraulic conductivity"]
Qinj = param["Injection"]["Injection rate"]
wh = param["Hydraulic aperture"]
T = param["T peak"]
S = param["Pre-stress ratio"]
P = param["Overpressure ratio"]
F = param["Peak to residual friction"]
f_p = param["Peak Friction"]
f_r = F*f_p
E = param["Elasticity"]["Young"]
nu = param["Elasticity"]["Nu"]
G = E / (2 * (1 + nu))
Rw = param["Rupture length scale"]


mm =  json_read(basefolder+'Mesh')
Nelts = mm["Nelts"]
conn = np.array(mm['Connectivity'])
coor = np.array(mm['Coordinates'])
Nnodes = len(coor)
mesh = usmesh(3, coor, conn, 0)

colPts= [(coor[conn[i][0]]+coor[conn[i][1]]+coor[conn[i][2]])/3. for i in range(Nelts)] # put it in usmesh
r=np.array([scipy.linalg.norm(coor[i]) for i in range(Nnodes)])
r_col=np.array([scipy.linalg.norm(colPts[i]) for i in range(Nelts)])

#%%
## plotting the unstructured mesh
triang=matplotlib.tri.Triangulation(coor[:,0], coor[:,1], triangles= conn, mask=None)
fig1, ax1 = plt.subplots()
ax1.set_aspect('equal')
ax1.triplot(triang, 'b-', lw=1)
ax1.plot(0.,0.,'ko')
plt.show()

#%%
#Create hmatrix for using the functions
from mechanics.H_Elasticity import *
from mechanics import H_Elasticity

kernel="3DT0"
elas_properties=np.array([E, nu])
elastic_m=Elasticity(kernel, elas_properties,max_leaf_size=64,eta=3.0,eps_aca=1.e-3)
# hmat creation
h1=elastic_m.constructHmatrix(mesh)
#%%

# Verification only if nu = 0

# post-processing
tts=np.array([res[i].time for i in range(len(res))])
nyi=np.array([res[i].Nyielded for i in range(len(res))])

rmax=0.*tts

for i in range(len(res)):
    aux_k = np.where(res[i].yieldedElts)[0]
    
    if len(aux_k) > 0:
        rmax[i] = r_col[aux_k].max()
    else:
        rmax[i] = 0.0  # or some other appropriate value

import json

arg_fric_mod = 'lin' # 'lin' or 'exp' based on the friction model used

# Construct the file path (modify it according to your file structure)

base_path = f'/home/sarma/Project_PhD/PyFracX-Examples/ReferenceSolutions/FDFR/SlipWeakeningBenchmarks/Linear/'
file_name = f'P{P}/S_{S}_P_{P}_F_{F}_{arg_fric_mod}_QD_L10_'
file_path = base_path + file_name

# Open and load the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)

# Example extraction of 'tlistNorm' and 'RfNorm' from the first entry (adjust according to your JSON structure)
tlistNorm = data['tlistNorm']
RfNorm = data['RfNorm']

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

# %%
solN = res[-1]

rcoor= (coor[:,0]**2+ coor[:,1]**2)**(0.5)

fig, ax = plt.subplots()
ax.plot(r_col, solN.pressure,'.')
plt.xlabel(" r (m)")
plt.ylabel(" Pressure (Pa)")
plt.show()

# %% slip 
global_dds = h1.convert_to_global(solN.DDs)

fig1, ax1 = plt.subplots()
tri=ax1.tripcolor(triang, global_dds[0::3],cmap = plt.cm.rainbow, 
                    alpha = 0.5)
ax1.axis('equal')
plt.colorbar(tri) 
plt.title('Slip along x (m)')
plt.show()

fig1, ax1 = plt.subplots()
tri=ax1.tripcolor(triang, global_dds[1::3],cmap = plt.cm.rainbow, 
                    alpha = 0.5)
ax1.axis('equal')
plt.colorbar(tri) 
plt.title('Slip along y (m)')
plt.show()

fig1, ax1 = plt.subplots()
tri=ax1.tripcolor(triang, global_dds[2::3],cmap = plt.cm.rainbow, 
                    alpha = 0.5)
ax1.axis('equal')
plt.colorbar(tri) 
plt.title('Opening (m)')
plt.show()

col_pts = np.asarray([np.mean(coor[conn[e,:],:],axis=0) for e in range(Nelts)])  #h1.getCollocationPoints()
rcoor_mid = np.sqrt(col_pts[:,0]**2+col_pts[:,1]**2)
fig, ax = plt.subplots()
ax.plot(rcoor_mid,  global_dds[2::3],'.')
plt.xlabel(" r (m)")
plt.ylabel(" Opening (m)")
plt.show()

fig, ax = plt.subplots()
ax.plot(rcoor_mid, global_dds[0::3],'.')
plt.xlabel(" r (m)")
plt.ylabel(" slip (m)")
plt.show()

# %%
plastic_dds = h1.convert_to_global(res[-1].DDs_plastic)

fig, ax = plt.subplots()
ax.plot(r_col,-plastic_dds[0::3] ,'.r', label='plastic')
ax.plot(r_col,-global_dds[0::3]+plastic_dds[0::3] ,'.b', label='elastic')
plt.xlabel('r')
plt.ylabel('Slip')
plt.legend()
#plt.xlim(0, 15)
plt.show()


fig, ax = plt.subplots()
ax.plot(r_col,plastic_dds[2::3] ,'.r', label='plastic')
ax.plot(r_col,global_dds[2::3]-plastic_dds[2::3] ,'.b', label='elastic')
plt.xlabel('r')
plt.ylabel('opening')
plt.legend()
plt.show()


# %% yield function 

fig1, ax1 = plt.subplots()
tri=ax1.tripcolor(triang, solN.yieldF,cmap = plt.cm.rainbow, 
                    alpha = 0.5)
ax1.axis('equal')
plt.colorbar(tri) 
plt.title('Yield function')
plt.show()

# %% traction

T_eff_n = h1.convert_to_global(solN.effective_tractions)

fig1, ax1 = plt.subplots()
tri=ax1.tripcolor(triang, T_eff_n[2::3],cmap = plt.cm.rainbow, 
                    alpha = 0.5)
ax1.axis('equal')
plt.colorbar(tri) 
plt.title('Normal effective traction')
plt.show()

fig1, ax1 = plt.subplots()
tri=ax1.tripcolor(triang, T_eff_n[0::3],cmap = plt.cm.rainbow, 
                    alpha = 0.5)
ax1.axis('equal')
plt.colorbar(tri) 
plt.title('Shear effective traction')
plt.show()

# %%
