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

#%% loading numerical results 
# here you need to change basefolder to the folder name of your simulation
basefolder = '/home/sarma/Project_PhD/PyFracX-Examples/3D-uncoupled/3D-ctFriction-ctperm-benchmark-20-08-2024-13-48-00/'
basename = "3D-ctFriction-ctperm-benchmark"

# we load n_steps - please adjst as per your simulation 
res = []
n_steps = 7

for i in range(n_steps):
    tt = from_dict_to_dataclass(HMFSolution,json_read(basefolder+basename+'-'+str(i+1)))
    res.append(tt)

param = json_read(basefolder+'Parameters')
#%%
alpha_hyd = param["Flow"]["fracture diffusivity"]
cond_hyd = param["Flow"]["Hydraulic conductivity"]
Qinj = param["Injection"]["Injection rate"]
wh = param["Flow"]["Initial aperture"]
T = param["T parameter"]
f = param["Friction coefficient"]
E = param["Elasticity"]["Young"]
nu = param["Elasticity"]["Nu"]
G = E / (2 * (1 + nu))

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
