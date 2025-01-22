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

#%% imports
import numpy as np
import matplotlib.pyplot as plt

from mesh.usmesh import usmesh

from utils.json_dict_dataclass_utils import *
from hm.HMFsolver import HMFSolution
from mechanics.evolutionLaws import linearEvolution
import json

#%% loading numerical results 
# here you need to change basefolder to the folder name of your simulation
import os
directory_path = os.path.dirname(os.path.abspath(__file__))
basefolder = directory_path + '/Axisymmetric-ctQ-LinearWeakening-VarPermeability-19-08-2024-13-30-29/'
basename="Axisymmetric-ctQ-LinearWeakening-VarPermeability"
# we load n_steps - please adjst as per your simulation 
res = []
n_steps =800 
for i in range(n_steps):
    sol = from_dict_to_dataclass(HMFSolution,json_read(basefolder+basename+'-'+str(i+1)))
    res.append(sol)

param = json_read(basefolder+'Parameters')

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

shear_prime=YoungM/(2*(1+nu)*2*(1-nu))

me =  json_read(basefolder+'Mesh')
Nelts = me["Nelts"] 
coor1D = np.array(me["Coordinates"])[:,0]

import matplotlib.pyplot as plt
#%% plots
timestep=-2
num_pressure=res[timestep].pressure
tt=np.array([res[j].time for j in range(len(res))])

####################################################################
####################################################################
# %% Overpressure at the well
pw_t=np.array([res[j].pressure[2] for j in range(len(res))])

fig, ax = plt.subplots()
ax.plot(tt[:], 1e-6*pw_t[:],'.b')
plt.xlabel("Time (s)")
plt.ylabel("Well Pressure MPa")
plt.xlim([0,res[-1].time])
plt.show()

#%% ##### rupture radius
nyi=np.array([res[i].Nyielded for i in range(len(res))])
cr_front = nyi*(coor1D[1]-coor1D[0])
fig, ax = plt.subplots()
ax.plot( tt ,cr_front ,'.y')

plt.xlabel("Time (s)")
plt.ylabel("rupture radius(m)")
plt.show()

#%%
#### slip at origin

slip_0 = np.abs(np.array([res[i].DDs[0] for i in range(len(res))]))
slip_p_0 = np.abs(np.array([res[i].DDs_plastic[0] for i in range(len(res))]))

fig, ax = plt.subplots()
ax.plot( tt ,slip_0 ,'.k', label='Total Slip')
ax.plot( tt ,slip_p_0 ,'.r', label='Plastic Slip')
plt.xlabel("Time (s)")
plt.ylabel("Slip @ injection point (m)")
plt.legend()
plt.show()

# %% width at origin
width_0 = np.abs(np.array([res[i].DDs[1] for i in range(len(res))]))
width_p_0 = np.abs(np.array([res[i].DDs_plastic[1] for i in range(len(res))]))

fig, ax = plt.subplots()
ax.plot( tt ,width_0 ,'.k', label='Total Opg')
ax.plot( tt ,width_p_0 ,'.b', label='Plastic Opg')
plt.xlabel("Time (s)")
plt.ylabel("Width @ injection point (m)")
plt.legend()
plt.show()

# %%
fig, ax = plt.subplots()
ax.plot( slip_0 ,width_0 ,'.k', label='Total slip (x) VS Total Opg (y)')
ax.plot( slip_p_0 ,width_p_0 ,'.b', label='Plastic slip (x) VS Plastic Opg (y)')
plt.xlabel("Slip @ injection point (m)")
plt.ylabel("Width @ injection point (m)")
plt.legend()
plt.show()

#%%
### profiles
timestep=-1
sol_to_p= res[timestep]
x=np.array(me["Coordinates"])[:,0]
num_pressure=sol_to_p.pressure
fig, ax = plt.subplots()
ax.plot(x[1:400], num_pressure[1:400],'.b')
plt.xlabel("  Radius (m)")
plt.ylabel("Fluid pressure (Pa)")
plt.show()

#%%
# width profile
w_profile=sol_to_p.DDs[1::2]
w_profile_p=sol_to_p.DDs_plastic[1::2]
fig, ax = plt.subplots()
ax.plot((x[1:]+x[0::-2])/2., w_profile[:],'.k')
ax.plot((x[1:]+x[0::-2])/2., w_profile_p[:],'.b')
plt.xlabel("  Radius (m)")
plt.ylabel(" Width (m)")
plt.show()

fig, ax = plt.subplots()

colPts = np.array(x[1:]+x[0::-2])/2.

ax.plot(colPts, (1+np.array(w_profile)[:]/who)**3,'.k')
plt.xlabel("  Radius (m)")
plt.ylabel(" Hydraulic transmissibility increase (-)")
plt.show()

#%%
# slip profile
slip_profile=sol_to_p.DDs[0::2]
slip_profile_p=sol_to_p.DDs_plastic[0::2]
fig, ax = plt.subplots()
# ax.plot(x[1:],true_sol[1:],'r')
ax.plot(colPts, -np.array(slip_profile)[:],'.k')
ax.plot(colPts, -np.array(slip_profile_p)[:],'.b')
plt.xlabel("  r (m)")
plt.ylabel("slip (m)")
plt.show()

# dilatancy profile
fig, ax = plt.subplots()
app_dila_c = -np.array(sol_to_p.DDs_rate)[1::2]/np.array(sol_to_p.DDs_rate)[0::2]
ax.plot(colPts,app_dila_c,'.b')
plt.xlabel("  r (m)")
plt.ylabel(" w/slip (-)")
plt.show()

#%%
# effective tracitons profile
tau = np.array(sol_to_p.effective_tractions)[0::2]
sig_n = np.array(sol_to_p.effective_tractions)[1::2]

fig, ax = plt.subplots()
# ax.plot(x[1:],true_sol[1:],'r')
ax.plot(colPts, -tau[:],'.k', label='Shear component')
ax.plot(colPts, -sig_n[:],'.b', label='Normal component')
plt.xlabel("  r (m)")
plt.ylabel("effective tractions (Pa)")
plt.legend()
plt.show()


fric_c= [linearEvolution(np.array(slip_profile_p)[i],d_c,f_p,f_r) for i in range(Nelts)]
dila_c = [linearEvolution(np.array(slip_profile_p)[i],d_c,psi_p,0.) for i in range(Nelts)]
fig, ax = plt.subplots()
ax.plot(colPts, fric_c,'.k', label='Friction evol.')
ax.plot(colPts, dila_c,'.b', label='Dilatancy evol.')

plt.xlabel("  r (m)")
plt.ylabel("friction coef. & dilatancy coef.")
plt.legend()
plt.show()

###
fig, ax = plt.subplots()
ax.plot(colPts, np.abs(tau)+fric_c*sig_n,'.b')

plt.xlabel("  r (m)")
plt.ylabel(" F_mc")
plt.show()